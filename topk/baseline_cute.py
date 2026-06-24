"""
Reference CuTe DSL top-k kernel.

Target runtime:
  - nvidia-cutlass-dsl == 4.5.2
  - nvidia-cutlass-dsl-libs-cu13 == 4.5.2

Design:
  - One CTA handles one row.
  - Lane 0 performs a serial top-k insertion pass for correctness and API stability.
  - This is intentionally a minimal CuTe DSL implementation, not a tuned radix-select kernel.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32
from cutlass.cute.runtime import from_dlpack
import torch


def make_topk_launcher(num_elements: int, top_k: int):
    @cute.kernel
    def topk_kernel(
        scores: cute.Tensor,
        output_values: cute.Tensor,
        output_indices: cute.Tensor,
    ):
        block_idx, _, _ = cute.arch.block_idx()
        thread_idx, _, _ = cute.arch.thread_idx()
        batch_idx = block_idx

        top_values = cute.make_fragment(top_k, Float32)
        top_indices = cute.make_fragment(top_k, Int32)

        if thread_idx == 0:
            neg_inf = Float32(float("-inf"))
            for slot in cutlass.range_constexpr(top_k):
                top_values[slot] = neg_inf
                top_indices[slot] = Int32(-1)

            for column in cutlass.range_constexpr(num_elements):
                candidate_value = scores[batch_idx, column]
                candidate_index = Int32(column)

                for slot in cutlass.range_constexpr(top_k):
                    if candidate_value > top_values[slot]:
                        displaced_value = top_values[slot]
                        displaced_index = top_indices[slot]
                        top_values[slot] = candidate_value
                        top_indices[slot] = candidate_index
                        candidate_value = displaced_value
                        candidate_index = displaced_index

            for slot in cutlass.range_constexpr(top_k):
                output_values[batch_idx, slot] = top_values[slot]
                output_indices[batch_idx, slot] = top_indices[slot]

    @cute.jit
    def launch_topk(
        scores: cute.Tensor,
        output_values: cute.Tensor,
        output_indices: cute.Tensor,
    ):
        batch_size = scores.shape[0]
        topk_kernel(
            scores,
            output_values,
            output_indices,
        ).launch(
            grid=(batch_size, 1, 1),
            block=(32, 1, 1),
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
