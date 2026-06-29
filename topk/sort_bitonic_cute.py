"""
SORT 策略 —— SonicMoE bitonic_topk 的 CuTeDSL Python 实现。

与 quack/sort/{utils.py, sorting_networks.py, bitonic_sort.py} + sonicmoe/topk.py
结构对齐。需要 NVIDIA GPU + cutlass.cute（CuTe DSL Python 绑定，PyPI: `nvidia-cutlass-dsl`）。

四层栈（文章 §2.6 图 C）：
  ① compare_and_swap  —— register 内无分支 fmin/fmax 交换
  ② optimal_sort / bitonic_sort —— 小块走最优网络，大块退回 bitonic
  ③ bitonic_topk_merge —— 反向配对取 max + 一次 bitonic_merge
  ④ bitonic_topk —— 主入口：register 折叠 + warp XOR butterfly

本版本不使用文章 §2.3 的 packed-key。score 和 expert id/index 分开维护：
  - values fragment 保存 Float32 score；
  - indices fragment 保存 Int32 expert id；
  - compare-and-swap 显式比较 (value, index)，value 更优先，value 相同则
    index 更小优先，用于对齐 torch.topk(sorted=True) 的稳定输出需求。

当前文件只做 CuTeDSL 4.5.2 + cu13 的工程适配：
  - 保留 SonicMoE/quack 的四层算法结构；
  - 用 make_topk_launcher(...) 生成 launcher，再由 launcher 调 kernel.launch；
  - kernel 内不用 early return，而是用 row_index < num_rows 包住主体。
"""

from __future__ import annotations

import math

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass.cute.runtime import from_dlpack
import torch

# ═══════════════════════════════════════════════════════════════════════════
# ① 原子层 —— 无分支 compare-and-swap（quack/sort/utils.py 同名函数）
# ═══════════════════════════════════════════════════════════════════════════
@cute.jit
def compare_and_swap(
    values,
    indices,
    left_index: Int32,
    right_index: Int32,
    ascending: cutlass.Constexpr = True,
):
    """register 内同步交换 value/index。

    不使用 packed-key 后，比较规则必须显式写出来：
      - descending/top-k 路径：value 大者优先；value 相等时 index 小者优先；
      - ascending 路径：value 小者优先；value 相等时 index 小者优先。

    bitonic 网络只负责决定 pair 是否交换；一旦交换，values 和 indices 必须
    同步搬运，否则输出 value/index 会错配。
    """
    left_value = values[left_index]
    right_value = values[right_index]
    left_column = indices[left_index]
    right_column = indices[right_index]

    if const_expr(ascending):
        should_swap = (
            right_value < left_value or
            (right_value == left_value and right_column < left_column)
        )
    else:
        should_swap = (
            right_value > left_value or
            (right_value == left_value and right_column < left_column)
        )

    if should_swap:
        values[left_index] = right_value
        values[right_index] = left_value
        indices[left_index] = right_column
        indices[right_index] = left_column


# ═══════════════════════════════════════════════════════════════════════════
# ② base sort —— 最优网络（n ≤ 16） + bitonic 后备（n ∈ {32,64,128}）
# ═══════════════════════════════════════════════════════════════════════════
# 一个内层 list 是一个 "level"，level 内的 compare-exchange 互不依赖，可并行。
# 来源：Bert Dobbelaere 最优网络表（quack/sort/sorting_networks.py 自动生成）。
OPTIMAL_NETWORKS = {
    2: [[(0, 1)]],
    4: [[(0, 2), (1, 3)], [(0, 1), (2, 3)], [(1, 2)]],
    8: [
        [(0, 2), (1, 3), (4, 6), (5, 7)],
        [(0, 4), (1, 5), (2, 6), (3, 7)],
        [(0, 1), (2, 3), (4, 5), (6, 7)],
        [(2, 4), (3, 5)],
        [(1, 4), (3, 6)],
        [(1, 2), (3, 4), (5, 6)],
    ],
    16: [
        [(0, 5), (1, 7), (2, 3), (4, 8), (6, 12), (9, 14), (10, 11), (13, 15)],
        [(0, 2), (1, 4), (3, 8), (5, 6), (7, 9), (10, 13), (11, 14), (12, 15)],
        [(0, 1), (2, 4), (3, 5), (6, 10), (7, 11), (8, 9), (12, 13), (14, 15)],
        [(0, 3), (1, 2), (4, 8), (5, 9), (6, 7), (11, 12), (13, 14)],
        [(1, 3), (2, 7), (4, 6), (5, 8), (9, 10), (12, 14)],
        [(2, 4), (3, 6), (5, 7), (8, 10), (9, 11), (13, 14)],
        [(3, 4), (5, 6), (7, 8), (9, 12), (10, 11)],
        [(6, 7), (8, 9), (10, 12)],
        [(4, 5), (6, 8), (7, 9), (10, 11)],
        [(5, 6), (7, 8), (9, 10)],
    ],
    # quack 里也有 32/64 的最优网络表；这里保持原文件策略：
    # 32/64/128 统一走 bitonic 递归，代码更短，结构仍对齐 SonicMoE。
}


@cute.jit
def optimal_sort(
    values,
    indices,
    size: cutlass.Constexpr,
    start_offset: cutlass.Constexpr = 0,
    ascending: cutlass.Constexpr = True,
):
    """对 values/indices 同步展开预生成最优网络。"""
    for parallel_level in const_expr(OPTIMAL_NETWORKS[size]):
        for left_index, right_index in const_expr(parallel_level):
            compare_and_swap(
                values,
                indices,
                start_offset + left_index,
                start_offset + right_index,
                ascending,
            )


@cute.jit
def bitonic_merge(
    values,
    indices,
    size: cutlass.Constexpr,
    start_offset: cutlass.Constexpr = 0,
    ascending: cutlass.Constexpr = True,
):
    """经典 bitonic merge：输入区间必须已经是 bitonic 序列。"""
    if const_expr(size <= 1):
        return

    half_size = const_expr(size // 2)
    for i in cutlass.range_constexpr(half_size):
        compare_and_swap(
            values,
            indices,
            start_offset + i,
            start_offset + i + half_size,
            ascending,
        )
    bitonic_merge(values, indices, half_size, start_offset, ascending)
    bitonic_merge(values, indices, half_size, start_offset + half_size, ascending)


# bitonic_sort 分为两个步骤
# Build Bitonic Sequence
# 1. bitonic_sort(left_half, ascending=True)
# 2. bitonic_sort(right_half, ascending=False)
# Merge Bitonic Sequence
# 3. bitonic_merge(whole_sequence, ascending)
@cute.jit
def bitonic_sort(
    values,
    indices,
    size: cutlass.Constexpr,
    start_offset: cutlass.Constexpr = 0,
    ascending: cutlass.Constexpr = True,
):
    """小块走 optimal_sort；大块拆半反向排序后 bitonic_merge。"""
    cutlass.const_expr(size <= 128 and (size & (size - 1)) == 0)

    if const_expr(size in (2, 4, 8, 16)):
        optimal_sort(values, indices, size, start_offset, ascending)
    else:
        half_size = const_expr(size // 2)
        bitonic_sort(values, indices, half_size, start_offset, True)
        bitonic_sort(values, indices, half_size, start_offset + half_size, False)
        bitonic_merge(values, indices, size, start_offset, ascending)


# ═══════════════════════════════════════════════════════════════════════════
# ③ 合并两个有序 top-k 成一个 top-k（quack/sort/bitonic_sort.py 同名函数）
# ═══════════════════════════════════════════════════════════════════════════
@cute.jit
def bitonic_topk_merge(
    left_values,
    left_indices,
    right_values,
    right_indices,
    k: cutlass.Constexpr,
    ascending: cutlass.Constexpr = False,
):
    """反向配对取 max/min，然后一次 bitonic_merge 得到合并后的 top-k。"""
    for i in cutlass.range_constexpr(k):
        left_value = left_values[i]
        left_index = left_indices[i]
        right_value = right_values[k - 1 - i]
        right_index = right_indices[k - 1 - i]

        if const_expr(ascending):
            should_take_right = (
                right_value < left_value or
                (right_value == left_value and right_index < left_index)
            )
        else:
            should_take_right = (
                right_value > left_value or
                (right_value == left_value and right_index < left_index)
            )

        if should_take_right:
            left_values[i] = right_value
            left_indices[i] = right_index

    bitonic_merge(left_values, left_indices, k, 0, ascending)


# ═══════════════════════════════════════════════════════════════════════════
# ④ 主入口 —— register 折叠 + warp XOR butterfly
# ═══════════════════════════════════════════════════════════════════════════
@cute.jit
def bitonic_topk(
    values,
    indices,
    array_size: cutlass.Constexpr,
    k: cutlass.Constexpr,
    ascending: cutlass.Constexpr = False,
    warp_width: cutlass.Constexpr = 32,
):
    """得到当前 warp 负责的一行的 top-k。

    输入含义：
      - values / indices 是“当前 lane 自己负责的那一段元素”，都在寄存器里；
      - array_size 是当前 lane 持有的元素个数，也就是 elements_per_thread；
      - k 是要选出的 top-k；
      - warp_width 是一行用了多少个 lane 协作处理。

    整体分成两级 reduce：

    1. lane 内 register-fold：
       当前 lane 只看自己的 values/indices。因为每个 lane 可能持有的元素数
       大于 k，所以先按 k 个元素一组切成多个 chunk。每个 chunk 内部先排序，
       然后和 current_* 做 bitonic_topk_merge。循环结束后，current_* 就是
       这个 lane 自己负责元素里的 lane-local top-k。

       例子：array_size=8, k=2 时，一个 lane 有 8 个元素：
         chunk0 = values[0:2] -> sort 得到初始 current top-2
         chunk1 = values[2:4] -> sort 后和 current merge
         chunk2 = values[4:6] -> sort 后和 current merge
         chunk3 = values[6:8] -> sort 后和 current merge
       最终 current_* 只剩这个 lane 的 top-2。

    2. warp 内 XOR butterfly：
       lane-local top-k 还不是整行 top-k，因为一行被多个 lane 分摊了。
       shuffle_sync_bfly 等价于 CUDA __shfl_xor_sync：当前 lane 从
       lane_id ^ offset 的伙伴 lane 读取数据。offset 每轮是 1, 2, 4, ...
       每一轮都把“当前 lane 已知的 top-k”和“伙伴 lane 的 top-k”合并，
       覆盖回 current_*。

       合并范围每轮翻倍：
         offset=1：合并相邻 2 个 lane 的结果；
         offset=2：合并 4 个 lane 的结果；
         offset=4：合并 8 个 lane 的结果；
         ...
       log2(warp_width) 轮后，每个 lane 都拥有整行的 row-level top-k。
       kernel 里只让 lane0 写回，是为了避免所有 lane 重复写同一份结果。
    """
    cutlass.const_expr(array_size % k == 0)

    # 初始化 lane-local top-k：
    # 先取当前 lane 持有的第一个 k-sized chunk，复制到 current_*。
    # 这里不用 fragment.load(start, len)，因为 CuTeDSL 4.5.2 的 fragment load
    # 接口不是切片语义；显式逐元素拷贝更稳定。
    current_values = cute.make_fragment(k, Float32)
    current_indices = cute.make_fragment(k, Int32)
    for i in cutlass.range_constexpr(k):
        current_values[i] = values[i]
        current_indices[i] = indices[i]

    # 把第一个 chunk 排成有序 top-k。descending 路径下，大 value 在前；
    # value 相等时 index 小的在前。
    bitonic_sort(current_values, current_indices, k, 0, ascending)

    # register-fold：
    # 当前 lane 剩余元素按 k 个一组处理。每个 chunk 先局部排序，再与 current_*
    # 合并。bitonic_topk_merge 的输入是两个已经有序的 top-k，输出仍写回
    # current_*，因此寄存器里始终只保留 k 个候选。
    num_chunks = const_expr(array_size // k)
    for chunk_index in cutlass.range_constexpr(1, num_chunks):
        chunk_values = cute.make_fragment(k, Float32)
        chunk_indices = cute.make_fragment(k, Int32)
        for i in cutlass.range_constexpr(k):
            chunk_values[i] = values[chunk_index * k + i]
            chunk_indices[i] = indices[chunk_index * k + i]
        bitonic_sort(chunk_values, chunk_indices, k, 0, ascending)
        bitonic_topk_merge(
            current_values,
            current_indices,
            chunk_values,
            chunk_indices,
            k,
            ascending,
        )

    # warp XOR butterfly：
    # 每轮从 lane_id ^ offset 的伙伴 lane 取出它当前的 top-k。
    # 注意 value 和 index 必须分别 shuffle，但两者的 slot 一一对应；
    # 之后一起传给 bitonic_topk_merge，保证 value/index 不错配。
    butterfly_levels = const_expr(int(math.log2(warp_width)))
    for butterfly_step in cutlass.range_constexpr(butterfly_levels):
        partner_values = cute.make_fragment(k, Float32)
        partner_indices = cute.make_fragment(k, Int32)
        for i in cutlass.range_constexpr(k):
            partner_values[i] = cute.arch.shuffle_sync_bfly(
                current_values[i],
                offset=1 << butterfly_step,
            )
            partner_indices[i] = cute.arch.shuffle_sync_bfly(
                current_indices[i],
                offset=1 << butterfly_step,
            )
        bitonic_topk_merge(
            current_values,
            current_indices,
            partner_values,
            partner_indices,
            k,
            ascending,
        )

    return current_values, current_indices


def make_topk_launcher(
    num_rows: int,
    padded_num_experts: int,
    top_k: int,
    threads_per_row: int,
    rows_per_block: int,
):
    """按 N/K/block 形状生成专用 launcher。

    这样做是为了对齐 CuTeDSL 4.5.2 的推荐调用形态：compile 的对象是
    launch_topk，而不是直接把 grid/block 作为 cute.compile 的关键字参数。
    算法本体仍然是下面的 bitonic_topk_kernel。
    """

    @cute.kernel
    def bitonic_topk_kernel(
        input_scores: cute.Tensor,
        output_values: cute.Tensor,
        output_indices: cute.Tensor,
    ):
        """每个 block 处理 rows_per_block 行，每行 threads_per_row 个 lane。"""
        block_id, _, _ = cute.arch.block_idx()
        thread_id_x, thread_id_y, _ = cute.arch.thread_idx()
        row_index = block_id * rows_per_block + thread_id_y

        if row_index < num_rows:
            elements_per_thread = const_expr(padded_num_experts // threads_per_row)
            row_values = cute.make_fragment(elements_per_thread, Float32)
            row_indices = cute.make_fragment(elements_per_thread, Int32)
            base_column = thread_id_x * elements_per_thread

            # ① 128-bit/vectorized load 的 Python DSL 表达：每个 lane 加载连续分片。
            for local_index in cutlass.range_constexpr(elements_per_thread):
                row_values[local_index] = input_scores[row_index, base_column + local_index]
                row_indices[local_index] = base_column + local_index

            # ② SonicMoE bitonic_topk：register-fold + warp butterfly。
            # 不使用 packed-key，所以 value/index 两个 fragment 一起参与合并。
            topk_values, topk_indices = bitonic_topk(
                row_values,
                row_indices,
                elements_per_thread,
                top_k,
                ascending=False,
                warp_width=threads_per_row,
            )

            # ③ butterfly 后所有 lane 都有 top-k，写一次即可。
            if thread_id_x == 0:
                for output_slot in cutlass.range_constexpr(top_k):
                    output_values[row_index, output_slot] = topk_values[output_slot]
                    output_indices[row_index, output_slot] = topk_indices[output_slot]

    @cute.jit
    def launch_topk(
        input_scores: cute.Tensor,
        output_values: cute.Tensor,
        output_indices: cute.Tensor,
    ):
        grid_x = const_expr((num_rows + rows_per_block - 1) // rows_per_block)
        bitonic_topk_kernel(
            input_scores,
            output_values,
            output_indices,
        ).launch(
            grid=(grid_x, 1, 1),
            block=(threads_per_row, rows_per_block, 1),
        )

    return launch_topk


# ═══════════════════════════════════════════════════════════════════════════
# 主机端 wrapper：cute.compile + cache + launch（对应 sonicmoe/topk.py）
# ═══════════════════════════════════════════════════════════════════════════
_compiled_kernel_cache = {}


def topk(scores: torch.Tensor, K: int):
    """T x E float32 -> top-K(values [T,K] float32, indices [T,K] int32)。

    门槛保持 SonicMoE topk 路径的典型约束：E <= 4096、K <= 16、E % 8 == 0。
    不满足时回退 torch.topk，避免在非目标形状上生成低效或非法 kernel。
    """
    assert scores.is_cuda, "scores must be on CUDA"
    assert scores.dtype == torch.float32 and scores.ndim == 2
    num_rows, num_experts = scores.shape
    assert 0 < K <= num_experts, "K must be in [1, num_experts]"

    if not (
        num_experts <= 4096
        and K <= 16
        and (K & (K - 1)) == 0
        and num_experts % 8 == 0
    ):
        values, indices = scores.topk(K, dim=-1)
        return values, indices.to(torch.int32)

    padded_num_experts = 1 << (num_experts - 1).bit_length()
    if padded_num_experts != num_experts:
        padding = torch.full(
            (num_rows, padded_num_experts - num_experts),
            float("-inf"),
            dtype=torch.float32,
            device=scores.device,
        )
        scores = torch.cat([scores, padding], dim=1)

    output_values = torch.empty((num_rows, K), dtype=torch.float32, device=scores.device)
    output_indices = torch.empty((num_rows, K), dtype=torch.int32, device=scores.device)

    threads_per_row = min(32, padded_num_experts // 8)
    rows_per_block = 4
    elements_per_thread = padded_num_experts // threads_per_row
    if elements_per_thread % K != 0:
        values, indices = scores.topk(K, dim=-1)
        return values, indices.to(torch.int32)

    cache_key = (
        num_rows,
        padded_num_experts,
        K,
        threads_per_row,
        rows_per_block,
        scores.device.type,
        scores.device.index,
    )
    if cache_key not in _compiled_kernel_cache:
        launch_topk = make_topk_launcher(
            num_rows,
            padded_num_experts,
            K,
            threads_per_row,
            rows_per_block,
        )
        _compiled_kernel_cache[cache_key] = cute.compile(
            launch_topk,
            from_dlpack(scores),
            from_dlpack(output_values),
            from_dlpack(output_indices),
        )

    _compiled_kernel_cache[cache_key](
        from_dlpack(scores),
        from_dlpack(output_values),
        from_dlpack(output_indices),
    )
    return output_values, output_indices


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("需要 CUDA GPU 才能跑 CuTe 版本。无 GPU 请用 torch.topk 或 CPU 参考实现。")
        raise SystemExit(0)

    sample_scores = torch.tensor(
        [[2.1, 0.5, 3.7, 1.2, 0.9, 4.5, 1.8, 0.3]],
        dtype=torch.float32,
        device="cuda",
    )
    values, indices = topk(sample_scores, K=2)
    print(f"输入:       {sample_scores[0].tolist()}")
    print(f"top-2 值:   {values[0].tolist()}  (期望 [4.5, 3.7])")
    print(f"top-2 索引: {indices[0].tolist()}  (期望 [5, 2])")
