"""
REDUCE 策略 —— K-pass max reduction 的 CuTeDSL Python 实现。

对应文章 §3.3，TileLang official top-k example 的 CuTe DSL 复刻。每个 thread
block 处理一行：K 轮里每轮做一次 warp-level reduce_max + argmax，把命中
位置 mask 成 -inf 后进入下一轮。工作量 ∝ K·E。
"""

from __future__ import annotations
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass.cute.runtime import from_dlpack
import torch

# ═══════════════════════════════════════════════════════════════════════════
# warp-level reduce_argmax：每个 lane 持一个 (value, index)，
# 经 5 步 XOR butterfly 把全局 max 及其 index 归约到所有 lane
# ═══════════════════════════════════════════════════════════════════════════
@cute.jit
def warp_reduce_argmax(local_value: Float32, local_index: Int32) -> tuple:
    """warp 内 32-lane argmax：XOR butterfly 5 步。

    每一步：从伙伴 lane 取 (partner_value, partner_index)，
    若 partner_value > local_value（平手取更小 index）则替换。
    """
    for butterfly_step in cutlass.range_constexpr(5):      # log2(32) = 5
        partner_offset = const_expr(1 << butterfly_step)
        partner_value = cute.arch.shuffle_sync_bfly(
            local_value, offset=partner_offset
        )
        partner_index = cute.arch.shuffle_sync_bfly(
            local_index, offset=partner_offset
        )
        # 平手用更小 index：稳定性
        should_take = (partner_value > local_value) | (
            (partner_value == local_value) & (partner_index < local_index)
        )
        local_value = partner_value if should_take else local_value
        local_index = partner_index if should_take else local_index
    return local_value, local_index

# ═══════════════════════════════════════════════════════════════════════════
# 顶层 kernel —— 一个 block 一行，K 轮 reduce
# ═══════════════════════════════════════════════════════════════════════════
@cute.kernel
def kpass_reduce_topk_kernel(
    input_scores:   cute.Tensor,
    output_values:  cute.Tensor,
    output_indices: cute.Tensor,
    num_elements:   cutlass.Constexpr,                     # E
    top_k:          cutlass.Constexpr,                     # K
):
    block_id, _, _ = cute.arch.block_idx()
    thread_id_x, _, _ = cute.arch.thread_idx()             # 0..31（warp 内）
    row_index = block_id

    # 把整行 load 进 register（per-thread 持 E/32 个元素）
    elements_per_thread = const_expr(num_elements // 32)
    working_scores = cute.make_fragment(elements_per_thread, Float32)
    base_column = thread_id_x * elements_per_thread
    for local_index in cutlass.range_constexpr(elements_per_thread):
        working_scores[local_index] = input_scores[row_index, base_column + local_index]

    # K 轮：每轮一次 warp argmax + mask
    for pass_index in cutlass.range_constexpr(top_k):
        # ── 1. 每个 thread 在自己的小块里求 local argmax
        local_max_value = working_scores[0]
        local_max_index = base_column
        for local_index in cutlass.range_constexpr(1, elements_per_thread):
            if working_scores[local_index] > local_max_value:
                local_max_value = working_scores[local_index]
                local_max_index = base_column + local_index

        # ── 2. warp 内归约到全局 argmax
        global_max_value, global_max_index = warp_reduce_argmax(
            local_max_value, local_max_index
        )

        # ── 3. lane 0 写结果
        if thread_id_x == 0:
            output_values [row_index, pass_index] = global_max_value
            output_indices[row_index, pass_index] = global_max_index

        # ── 4. mask：持有命中元素的那个 thread 把对应 slot 设 -inf
        if (
            base_column <= global_max_index
            and global_max_index < base_column + elements_per_thread
        ):
            working_scores[global_max_index - base_column] = Float32(-float("inf"))


# ═══════════════════════════════════════════════════════════════════════════
# 主机端 wrapper
# ═══════════════════════════════════════════════════════════════════════════
_compiled_kernel_cache = {}


def topk(scores: torch.Tensor, K: int):
    """T × E float32 → (values [T,K] float32, indices [T,K] int32)。"""
    assert scores.dtype == torch.float32 and scores.ndim == 2
    num_rows, num_elements = scores.shape
    assert num_elements % 32 == 0, "E 必须是 32 倍数（一个 warp 处理一行）"

    output_values  = torch.empty(num_rows, K, dtype=torch.float32, device=scores.device)
    output_indices = torch.empty(num_rows, K, dtype=torch.int32, device=scores.device)

    cache_key = (num_elements, K, scores.device.type)
    if cache_key not in _compiled_kernel_cache:
        _compiled_kernel_cache[cache_key] = cute.compile(
            kpass_reduce_topk_kernel,
            from_dlpack(scores),
            from_dlpack(output_values), from_dlpack(output_indices),
            num_elements, K,
            grid=(num_rows, 1, 1), block=(32, 1, 1),
        )
    _compiled_kernel_cache[cache_key](
        from_dlpack(scores),
        from_dlpack(output_values), from_dlpack(output_indices),
    )
    return output_values, output_indices


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("需要 CUDA GPU 才能跑 CuTe 版本。无 GPU 请用 02_reduce_kpass.py 的 numpy 版本。")
        import sys; sys.exit(0)
    sample_scores = torch.tensor(
        [[2.1, 0.5, 3.7, 1.2, 0.9, 4.5, 1.8, 0.3] * 4],   # 凑到 32
        dtype=torch.float32, device="cuda",
    )
    values, indices = topk(sample_scores, K=2)
    print(f"top-2 值:   {values[0].tolist()}")
    print(f"top-2 索引: {indices[0].tolist()}")