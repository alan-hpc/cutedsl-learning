"""
CuTe DSL 版 radix-select top-k kernel。

目标运行环境：
  - nvidia-cutlass-dsl == 4.5.2
  - nvidia-cutlass-dsl-libs-cu13 == 4.5.2

整体算法参照 01_select_radix_cute.py：
  1. 每个 CTA 处理输入矩阵的一行，CTA 内固定 32 个线程，也就是一个 warp。
  2. 每个 lane 负责这一行里连续的 N / 32 个元素，把 float32 score 转成
     单调 uint32 key。这个 key 的无符号整数顺序等价于原始 float 的数值顺序。
  3. 对 32-bit key 做 4 轮 radix-select，每轮处理 8 bit，也就是 256 个桶。
     扫桶方向是 255 -> 0，因为我们要找第 K 大，而不是第 K 小。
  4. 每轮根据当前已经锁定的高位前缀 threshold_prefix，只统计仍可能包含
     第 K 大元素的候选集合。找到包含第 K 大 key 的桶后，把这个桶拼到
     threshold_prefix 上。
  5. 4 轮结束后，threshold_prefix 就是第 K 大元素对应的完整 32-bit key。
     之后先写出 key > threshold 的元素，再用 key == threshold 的元素补齐。

和 01_select_radix_cute.py 的主要实现差异：
  - 参考文件描述的是 shared-memory histogram + warp-aggregated atomicAdd。
  - 当前目标是 CuTe DSL 4.5.2 + cu13。这个版本里 shared-memory atomic/pointer
    相关接口更容易踩版本差异，所以这里保留 radix-select 语义，但用
    vote_ballot_sync + popc 统计每个 bucket 的数量。
  - 这种实现仍然是 warp 协作的 radix-select，只是 histogram 的表达方式
    从 shared-memory atomic 改成了 warp ballot 计数。
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Uint32, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import from_dlpack
import torch


WARP_SIZE = 32
RADIX_BITS = 8
NUM_BUCKETS = 1 << RADIX_BITS
FULL_MASK = 0xFFFFFFFF


@cute.jit
def float_to_monotone_key(value: Float32) -> Uint32:
    # radix-select 本质上只会按 bit/byte 比较整数 key。
    # 直接把 float32 的原始 bit 当 uint32 比较会出错：
    #   - 正负号会让所有负数排到正数之后；
    #   - 负数内部的 bit 序和数值序也是反的。
    #
    # 这里使用常见的 float flip 技巧：
    #   - 正数：翻转符号位，把它们移动到 uint32 的高半区；
    #   - 负数：所有 bit 取反，把负数内部顺序翻回来。
    #
    # 转换之后，key 的 uint32 大小关系就等价于 float 数值大小关系。
    bits = llvm.bitcast(Uint32.mlir_type, value)
    sign_bit = bits >> 31
    flip_mask = (Uint32(0) - sign_bit) | Uint32(0x80000000)
    return bits ^ flip_mask


def make_topk_launcher(num_elements: int, top_k: int):
    elements_per_lane = num_elements // WARP_SIZE

    @cute.jit
    def load_row_keys(scores, batch_idx, thread_idx, keys, columns):
        """把当前行分片加载到每个 lane 的寄存器 fragment 中。"""
        base_column = thread_idx * elements_per_lane

        # 每个 lane 负责这一行中的一段连续元素：
        #   lane 0 负责 [0, elements_per_lane)
        #   lane 1 负责 [elements_per_lane, 2 * elements_per_lane)
        #   ...
        #
        # keys[] 存转换后的 monotone uint32 key，后续 radix 只看 key。
        # columns[] 存原始列号，用于最后把 top-k 的原始 index 写回 output_indices。
        # 这里都放在 fragment 里，也就是每个 lane 自己的寄存器数组。
        for local_idx in cutlass.range_constexpr(elements_per_lane):
            column = base_column + local_idx
            keys[local_idx] = float_to_monotone_key(scores[batch_idx, column])
            columns[local_idx] = column

    @cute.jit
    def count_bucket(keys, threshold_prefix, prefix_mask, bit_shift, bucket):
        """统计当前 radix pass 中某个 bucket 的元素数量。"""
        bucket_count = Int32(0)

        # 参考实现里这里会写 shared-memory histogram，并用 atomicAdd 累加 bucket
        # count。当前实现改成 ballot：
        #   1. 每个 lane 针对自己的 local_idx 计算一个 in_bucket 谓词；
        #   2. vote_ballot_sync 把 32 个 lane 的谓词收集成一个 bit mask；
        #   3. popc(mask) 得到这一组 local_idx 在当前 bucket 中的数量；
        #   4. 对所有 local_idx 累加，得到整个 warp/整行在该 bucket 的数量。
        #
        # prefix_matches 用来过滤掉不属于当前候选前缀的元素。
        for local_idx in cutlass.range_constexpr(elements_per_lane):
            prefix_matches = (keys[local_idx] & Uint32(prefix_mask)) == threshold_prefix
            bucket_id = (keys[local_idx] >> bit_shift) & Uint32(0xFF)
            in_bucket = prefix_matches and (bucket_id == Uint32(bucket))
            bucket_mask = cute.arch.vote_ballot_sync(in_bucket, FULL_MASK)
            bucket_count += cute.arch.popc(bucket_mask)

        return bucket_count

    @cute.jit
    def select_threshold_key(keys):
        """执行 4 轮 8-bit radix-select，返回第 K 大元素的 monotone key。"""
        # threshold_prefix 表示“当前已经确定的第 K 大 key 的高位前缀”。
        #
        # 举例：第一轮选中了最高 8 bit 的 bucket=0xC1，则 threshold_prefix
        # 的最高 8 bit 会变成 0xC1，其余低位仍是 0。第二轮只会在最高 8 bit
        # 等于 0xC1 的候选元素里继续选下一个 byte。
        #
        # above_count 表示已经确定“严格大于当前候选前缀”的元素数量。
        # 当某一轮从高到低扫桶时，如果跳过了更大的桶，这些桶里的元素都一定
        # 大于最终 threshold，需要计入 above_count。下一轮要找的就不是全局
        # 第 K 个，而是候选集合里的第 (K - above_count) 个。
        threshold_prefix = Uint32(0)
        above_count = Int32(0)

        # 4 轮 radix-select，每轮处理 8 bit：
        #   pass 0: bits [31:24]
        #   pass 1: bits [23:16]
        #   pass 2: bits [15:8]
        #   pass 3: bits [7:0]
        #
        # 每一轮都只统计“高位前缀已经匹配 threshold_prefix”的元素。
        # 这样每轮都在逐步缩小候选集合，最终锁定完整 32-bit threshold key。
        for pass_index in cutlass.range_constexpr(4):
            # pass_index = 0, bit_shift = 24, prefix_mask = 0x00000000
            # pass_index = 1, bit_shift = 16, prefix_mask = 0xFF000000
            # pass_index = 2, bit_shift = 8,  prefix_mask = 0xFFFF0000
            # pass_index = 3, bit_shift = 0,  prefix_mask = 0xFFFFFF00
            bit_shift = const_expr(32 - RADIX_BITS * (pass_index + 1))
            prefix_mask = const_expr(((~((1 << (bit_shift + RADIX_BITS)) - 1))) & 0xFFFFFFFF)
            remaining_need = Int32(top_k) - above_count
            selected_bucket = Uint32(0)
            selected_found = Int32(0)
            cumulative = Int32(0)

            # 从大 bucket 到小 bucket 依次扫描。
            #
            # cumulative 表示已经扫过的更大 bucket 中元素数量之和。
            # 第一次满足 cumulative + bucket_count >= remaining_need 的 bucket，
            # 就是包含第 K 大 key 的 bucket。
            for reverse_offset in cutlass.range_constexpr(NUM_BUCKETS):
                bucket = const_expr(NUM_BUCKETS - 1 - reverse_offset)
                bucket_count = count_bucket(keys, threshold_prefix, prefix_mask, bit_shift, bucket)

                should_select = (
                    selected_found == 0 and
                    cumulative + bucket_count >= remaining_need
                )
                if should_select:
                    selected_bucket = Uint32(bucket)
                    above_count = cumulative
                    selected_found = Int32(1)
                if selected_found == 0:
                    cumulative += bucket_count

            # 把本轮选中的 8-bit bucket 拼进 threshold_prefix。
            # 4 轮之后，threshold_prefix 就是完整的第 K 大 key。
            threshold_prefix = threshold_prefix | (selected_bucket << bit_shift)

        return threshold_prefix

    @cute.jit
    def compact_greater(scores, output_values, output_indices, batch_idx, thread_idx, keys, columns, threshold_key):
        """先写出所有 key > threshold 的元素，返回已经写出的数量。"""
        output_count = Int32(0)

        # 这些元素一定属于 top-k，而且不会受到 threshold 重复值的影响。
        # output_count 记录已经写出的元素数量。
        # lane_prefix 是当前 lane 在本次 ballot 中的写入偏移，用于 warp 内 compact。
        for local_idx in cutlass.range_constexpr(elements_per_lane):
            emit_now = keys[local_idx] > threshold_key
            emit_mask = cute.arch.vote_ballot_sync(emit_now, FULL_MASK)
            lane_prefix = cute.arch.popc(emit_mask & ((1 << thread_idx) - 1))
            output_slot = output_count + lane_prefix

            if emit_now and output_slot < Int32(top_k):
                output_values[batch_idx, output_slot] = scores[batch_idx, columns[local_idx]]
                output_indices[batch_idx, output_slot] = columns[local_idx]

            output_count += cute.arch.popc(emit_mask)

        return output_count

    @cute.jit
    def compact_equal(
        scores,
        output_values,
        output_indices,
        batch_idx,
        thread_idx,
        keys,
        columns,
        threshold_key,
        output_count,
    ):
        """用 key == threshold 的元素补齐剩余 top-k slot。"""
        # 如果 threshold 对应的值出现多次，严格大于 threshold 的元素数量会小于 K，
        # 这时需要从等于 threshold 的元素里选出一部分补齐到 K。
        #
        # 注意：这里保证 values 和 indices 是自洽的；重复值之间的具体 index 顺序
        # 后面会通过一个小排序做稳定化。
        for local_idx in cutlass.range_constexpr(elements_per_lane):
            emit_now = keys[local_idx] == threshold_key
            emit_mask = cute.arch.vote_ballot_sync(emit_now, FULL_MASK)
            lane_prefix = cute.arch.popc(emit_mask & ((1 << thread_idx) - 1))
            output_slot = output_count + lane_prefix

            if emit_now and output_slot < Int32(top_k):
                output_values[batch_idx, output_slot] = scores[batch_idx, columns[local_idx]]
                output_indices[batch_idx, output_slot] = columns[local_idx]

            output_count += cute.arch.popc(emit_mask)

    @cute.jit
    def sort_outputs(output_values, output_indices, batch_idx):
        """把 radix-select 选出的 top-k 集合整理成 torch.topk(sorted=True) 风格。"""
        # 输出排序：
        #   - torch.topk(..., sorted=True) 会按 value 降序返回；
        #   - radix-select 的 gather 阶段只保证选出了 top-k 集合，不保证输出顺序；
        #   - 为了方便和 torch.topk 对齐，这里让 lane 0 对 K 个输出做一个小排序。
        #
        # 排序规则：
        #   1. value 大的排前面；
        #   2. value 相等时 index 小的排前面，作为稳定 tie-break。
        #
        # PyTorch CUDA 对重复值的 index 顺序不做跨版本承诺，所以这里不追求完全复制
        # PyTorch 内部 tie 行为，只保证输出稳定、value 对齐、index 指向正确元素。
        for outer in cutlass.range_constexpr(top_k):
            for inner in cutlass.range_constexpr(top_k - 1 - outer):
                current_value = output_values[batch_idx, inner]
                next_value = output_values[batch_idx, inner + 1]
                current_index = output_indices[batch_idx, inner]
                next_index = output_indices[batch_idx, inner + 1]
                should_swap = (
                    next_value > current_value or
                    (next_value == current_value and next_index < current_index)
                )
                if should_swap:
                    output_values[batch_idx, inner] = next_value
                    output_values[batch_idx, inner + 1] = current_value
                    output_indices[batch_idx, inner] = next_index
                    output_indices[batch_idx, inner + 1] = current_index

    @cute.kernel
    def radix_select_topk_kernel(
        scores: cute.Tensor,
        output_values: cute.Tensor,
        output_indices: cute.Tensor,
    ):
        block_idx, _, _ = cute.arch.block_idx()
        thread_idx, _, _ = cute.arch.thread_idx()
        batch_idx = block_idx

        keys = cute.make_fragment(elements_per_lane, Uint32)
        columns = cute.make_fragment(elements_per_lane, Int32)

        load_row_keys(scores, batch_idx, thread_idx, keys, columns)
        threshold_key = select_threshold_key(keys)
        output_count = compact_greater(
            scores,
            output_values,
            output_indices,
            batch_idx,
            thread_idx,
            keys,
            columns,
            threshold_key,
        )
        compact_equal(
            scores,
            output_values,
            output_indices,
            batch_idx,
            thread_idx,
            keys,
            columns,
            threshold_key,
            output_count,
        )
        cute.arch.barrier()

        if thread_idx == 0:
            sort_outputs(output_values, output_indices, batch_idx)

    @cute.jit
    def launch_topk(
        scores: cute.Tensor,
        output_values: cute.Tensor,
        output_indices: cute.Tensor,
    ):
        batch_size = scores.shape[0]
        radix_select_topk_kernel(
            scores,
            output_values,
            output_indices,
        ).launch(
            grid=(batch_size, 1, 1),
            block=(WARP_SIZE, 1, 1),
        )

    return launch_topk


_compiled_kernel_cache = {}


def topk(scores: torch.Tensor, k: int):
    """
    Args:
        scores: [B, N] float32 CUDA tensor.
        k: number of top elements to select per row.
    Returns:
        values: [B, K]
        indices: [B, K]
    """
    assert scores.is_cuda, "scores must be on CUDA"
    assert scores.dtype == torch.float32 and scores.ndim == 2
    batch_size, num_elements = scores.shape
    assert num_elements % WARP_SIZE == 0, "num_elements must be divisible by 32"
    assert 0 < k <= num_elements, "k must be in [1, num_elements]"

    output_values = torch.empty((batch_size, k), dtype=torch.float32, device=scores.device)
    output_indices = torch.empty((batch_size, k), dtype=torch.int32, device=scores.device)

    cache_key = (num_elements, k, scores.device.type)
    if cache_key not in _compiled_kernel_cache:
        launch_topk = make_topk_launcher(num_elements, k)
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
        print("需要 CUDA GPU 才能跑 CuTe 版本。")
        raise SystemExit(0)

    sample_scores = torch.tensor(
        [[2.1, 0.5, 3.7, 1.2, 0.9, 4.5, 1.8, 0.3] * 4],
        dtype=torch.float32,
        device="cuda",
    )
    values, indices = topk(sample_scores, 2)
    print(f"top-2 值:   {values[0].tolist()}")
    print(f"top-2 索引: {indices[0].tolist()}")
