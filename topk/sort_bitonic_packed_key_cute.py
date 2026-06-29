"""
SORT 策略 —— SonicMoE bitonic_topk 的 CuTeDSL Python 实现。

与 quack/sort/{utils.py, sorting_networks.py, bitonic_sort.py} + sonicmoe/topk.py
结构对齐。需要 NVIDIA GPU + cutlass.cute（CuTe DSL Python 绑定，PyPI: `nvidia-cutlass-dsl`）。

四层栈（文章 §2.6 图 C）：
  ① compare_and_swap  —— register 内无分支 fmin/fmax 交换
  ② optimal_sort / bitonic_sort —— 小块走最优网络，大块退回 bitonic
  ③ bitonic_topk_merge —— 反向配对取 max + 一次 bitonic_merge
  ④ bitonic_topk —— 主入口：register 折叠 + warp XOR butterfly

加上文章 §2.3 的 packed-key：FP32 score 尾数低 log2(E) 位编进 expert id，
一个 uint32 比较同时携带 value + tie-break，省掉双路搬运。

当前文件只做 CuTeDSL 4.5.2 + cu13 的工程适配：
  - 保留 SonicMoE/quack 的四层算法结构；
  - 用 make_topk_launcher(...) 生成 launcher，再由 launcher 调 kernel.launch；
  - kernel 内不用 early return，而是用 row_index < num_rows 包住主体。
"""

from __future__ import annotations

import math

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Uint32, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import from_dlpack
import torch


FULL_MASK = 0xFFFFFFFF


# ═══════════════════════════════════════════════════════════════════════════
# ① 原子层 —— 无分支 compare-and-swap（quack/sort/utils.py 同名函数）
# ═══════════════════════════════════════════════════════════════════════════
@cute.jit
def compare_and_swap(
    array,
    left_index: Int32,
    right_index: Int32,
    ascending: cutlass.Constexpr = True,
):
    """register array 内的原地 min/max 交换。

    FP32 走 cute.arch.fmin/fmax，编译后是 PTX min.f32 / max.f32 这类无分支
    指令。ascending=False 是 top-k 最大值路径：左边放 max，右边放 min。

    packed-key 的值仍以 Float32 形式参与 fmin/fmax，低位携带 expert id。
    这保持了原始 SonicMoE 路径的“一个 value 同时代表 score + tie-break”语义。
    """
    left_value, right_value = array[left_index], array[right_index]
    if const_expr(array.element_type == Float32):
        if const_expr(ascending):
            array[left_index], array[right_index] = (
                cute.arch.fmin(left_value, right_value),
                cute.arch.fmax(left_value, right_value),
            )
        else:
            array[left_index], array[right_index] = (
                cute.arch.fmax(left_value, right_value),
                cute.arch.fmin(left_value, right_value),
            )
    else:
        if const_expr(ascending):
            array[left_index], array[right_index] = (
                min(left_value, right_value),
                max(left_value, right_value),
            )
        else:
            array[left_index], array[right_index] = (
                max(left_value, right_value),
                min(left_value, right_value),
            )


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
    array,
    size: cutlass.Constexpr,
    start_offset: cutlass.Constexpr = 0,
    ascending: cutlass.Constexpr = True,
):
    """对 array[start_offset : start_offset + size] 展开预生成最优网络。"""
    for parallel_level in const_expr(OPTIMAL_NETWORKS[size]):
        for left_index, right_index in const_expr(parallel_level):
            compare_and_swap(
                array,
                start_offset + left_index,
                start_offset + right_index,
                ascending,
            )


@cute.jit
def bitonic_merge(
    array,
    size: cutlass.Constexpr,
    start_offset: cutlass.Constexpr = 0,
    ascending: cutlass.Constexpr = True,
):
    """经典 bitonic merge：输入区间必须已经是 bitonic 序列。"""
    if const_expr(size <= 1):
        return

    half_size = const_expr(size // 2)
    for i in cutlass.range_constexpr(half_size):
        compare_and_swap(array, start_offset + i, start_offset + i + half_size, ascending)
    bitonic_merge(array, half_size, start_offset, ascending)
    bitonic_merge(array, half_size, start_offset + half_size, ascending)


@cute.jit
def bitonic_sort(
    array,
    size: cutlass.Constexpr,
    start_offset: cutlass.Constexpr = 0,
    ascending: cutlass.Constexpr = True,
):
    """小块走 optimal_sort；大块拆半反向排序后 bitonic_merge。"""
    cutlass.const_expr(size <= 128 and (size & (size - 1)) == 0)

    if const_expr(size in (2, 4, 8, 16)):
        optimal_sort(array, size, start_offset, ascending)
    else:
        half_size = const_expr(size // 2)
        bitonic_sort(array, half_size, start_offset, True)
        bitonic_sort(array, half_size, start_offset + half_size, False)
        bitonic_merge(array, size, start_offset, ascending)


# ═══════════════════════════════════════════════════════════════════════════
# ③ 合并两个有序 top-k 成一个 top-k（quack/sort/bitonic_sort.py 同名函数）
# ═══════════════════════════════════════════════════════════════════════════
@cute.jit
def bitonic_topk_merge(
    left_topk,
    right_topk,
    k: cutlass.Constexpr,
    ascending: cutlass.Constexpr = False,
):
    """反向配对取 max/min，然后一次 bitonic_merge 得到合并后的 top-k。"""
    for i in cutlass.range_constexpr(k):
        if const_expr(not ascending):
            left_topk[i] = cute.arch.fmax(left_topk[i], right_topk[k - 1 - i])
        else:
            left_topk[i] = cute.arch.fmin(left_topk[i], right_topk[k - 1 - i])
    bitonic_merge(left_topk, k, 0, ascending)


# ═══════════════════════════════════════════════════════════════════════════
# ④ 主入口 —— register 折叠 + warp XOR butterfly
# ═══════════════════════════════════════════════════════════════════════════
@cute.jit
def bitonic_topk(
    array,
    array_size: cutlass.Constexpr,
    k: cutlass.Constexpr,
    ascending: cutlass.Constexpr = False,
    warp_width: cutlass.Constexpr = 32,
):
    """得到当前 warp 负责的一行的 top-k。

    第一段 register-fold：每个 lane 先在自己的寄存器数组里把多个 K-sized chunk
    折叠成 lane-local top-k。

    第二段 warp XOR butterfly：lane 与 lane ^ (1 << step) 交换局部 top-k，然后
    用 bitonic_topk_merge 合并。log2(warp_width) 轮后，所有 lane 都持有同一份
    row-level top-k，最终只让 lane0 写回。
    """
    cutlass.const_expr(array_size % k == 0)

    current_topk = cute.make_fragment(k, Float32)
    for i in cutlass.range_constexpr(k):
        current_topk[i] = array[i]
    bitonic_sort(current_topk, k, 0, ascending)

    num_chunks = const_expr(array_size // k)
    for chunk_index in cutlass.range_constexpr(1, num_chunks):
        chunk_topk = cute.make_fragment(k, Float32)
        for i in cutlass.range_constexpr(k):
            chunk_topk[i] = array[chunk_index * k + i]
        bitonic_sort(chunk_topk, k, 0, ascending)
        bitonic_topk_merge(current_topk, chunk_topk, k, ascending)

    butterfly_levels = const_expr(int(math.log2(warp_width)))
    for butterfly_step in cutlass.range_constexpr(butterfly_levels):
        partner_topk = cute.make_fragment(k, Float32)
        for i in cutlass.range_constexpr(k):
            partner_topk[i] = cute.arch.shuffle_sync_bfly(
                current_topk[i],
                offset=1 << butterfly_step,
            )
        bitonic_topk_merge(current_topk, partner_topk, k, ascending)

    return current_topk


# ═══════════════════════════════════════════════════════════════════════════
# Packed-key 编/解码（文章 §2.3）—— 对应 _TopK.kernel 七步主干的 ④⑥
# ═══════════════════════════════════════════════════════════════════════════
@cute.jit
def pack_expert_ids_into_scores(
    scores,
    num_scores: cutlass.Constexpr,
    base_column: Int32,
    expert_id_bits: cutlass.Constexpr,
):
    """把 expert id 编进 FP32 尾数低 expert_id_bits 位。

    非负 score 越大越优；score 相等时希望更小 expert id 更优。为了让一次
    Float32 比较同时携带 tie-break：
      - 非负值：写入 ~column 的低位，column 越小，encoded 越大；
      - 负值：写入 column 的低位，保持负数 FP32 bit 序下的比较方向。

    这里用 llvm.bitcast 做寄存器级 reinterpret，不通过内存搬运。
    """
    expert_id_mask = const_expr((1 << expert_id_bits) - 1)
    for local_index in cutlass.range_constexpr(num_scores):
        value = scores[local_index]
        column = base_column + local_index
        column_bits = column.to(Uint32)
        bits = llvm.bitcast(Uint32.mlir_type, value)
        encoded_id = Uint32(0)

        if value >= Float32(0):
            encoded_id = (~column_bits) & Uint32(expert_id_mask)
        else:
            encoded_id = column_bits & Uint32(expert_id_mask)

        bits = (bits & Uint32(~expert_id_mask & 0xFFFFFFFF)) | encoded_id
        scores[local_index] = llvm.bitcast(Float32.mlir_type, bits)


@cute.jit
def unpack_expert_ids_from_topk(
    topk_values,
    topk_indices,
    k: cutlass.Constexpr,
    expert_id_bits: cutlass.Constexpr,
):
    """从 packed key 低位取回 expert id，同时清零低位恢复近似原 score。"""
    expert_id_mask = const_expr((1 << expert_id_bits) - 1)
    for output_slot in cutlass.range_constexpr(k):
        value = topk_values[output_slot]
        bits = llvm.bitcast(Uint32.mlir_type, value)
        encoded_id = bits & Uint32(expert_id_mask)
        column = Uint32(0)

        if value >= Float32(0):
            column = (~encoded_id) & Uint32(expert_id_mask)
        else:
            column = encoded_id & Uint32(expert_id_mask)

        topk_values[output_slot] = llvm.bitcast(
            Float32.mlir_type,
            bits & Uint32(~expert_id_mask & 0xFFFFFFFF),
        )
        topk_indices[output_slot] = column.to(Int32)


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
            row_registers = cute.make_fragment(elements_per_thread, Float32)
            base_column = thread_id_x * elements_per_thread

            # ① 128-bit/vectorized load 的 Python DSL 表达：每个 lane 加载连续分片。
            for local_index in cutlass.range_constexpr(elements_per_thread):
                row_registers[local_index] = input_scores[row_index, base_column + local_index]

            # ② packed-key：把 expert id 写进 score 低位，后面只搬一份 Float32。
            expert_id_bits = const_expr(padded_num_experts.bit_length() - 1)
            pack_expert_ids_into_scores(
                row_registers,
                elements_per_thread,
                base_column,
                expert_id_bits,
            )

            # ③ SonicMoE bitonic_topk：register-fold + warp butterfly。
            topk_packed = bitonic_topk(
                row_registers,
                elements_per_thread,
                top_k,
                ascending=False,
                warp_width=threads_per_row,
            )

            # ④ unpack + lane0 写回。butterfly 后所有 lane 都有 top-k，写一次即可。
            topk_expert_ids = cute.make_fragment(top_k, Int32)
            unpack_expert_ids_from_topk(
                topk_packed,
                topk_expert_ids,
                top_k,
                expert_id_bits,
            )

            if thread_id_x == 0:
                for output_slot in cutlass.range_constexpr(top_k):
                    output_values[row_index, output_slot] = topk_packed[output_slot]
                    output_indices[row_index, output_slot] = topk_expert_ids[output_slot]

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
