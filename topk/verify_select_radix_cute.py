from __future__ import annotations

import argparse
import time

import torch

import baseline_cute
import select_radix_cute


def assert_topk_matches(scores: torch.Tensor, k: int, *, check_indices: bool = False) -> None:
    actual_values, actual_indices = select_radix_cute.topk(scores, k)
    expected_values, expected_indices = torch.topk(scores, k, dim=1, largest=True, sorted=True)

    torch.cuda.synchronize()

    if not torch.equal(actual_values, expected_values):
        raise AssertionError(
            "value mismatch\n"
            f"actual={actual_values.cpu()}\n"
            f"expected={expected_values.cpu()}"
        )

    gathered_values = scores.gather(1, actual_indices.to(torch.long))
    if not torch.equal(actual_values, gathered_values):
        raise AssertionError(
            "index mismatch\n"
            f"values={actual_values.cpu()}\n"
            f"indices={actual_indices.cpu()}\n"
            f"gathered={gathered_values.cpu()}"
        )

    if check_indices and not torch.equal(actual_indices.to(expected_indices.dtype), expected_indices):
        raise AssertionError(
            "strict index mismatch\n"
            f"actual={actual_indices.cpu()}\n"
            f"expected={expected_indices.cpu()}"
        )


def run_correctness() -> None:
    torch.manual_seed(0)
    cases = [
        (1, 32, 1),
        (1, 32, 2),
        (4, 32, 4),
        (2, 64, 8),
        (3, 128, 16),
    ]

    sample = torch.tensor(
        [[2.1, 0.5, 3.7, 1.2, 0.9, 4.5, 1.8, 0.3] * 4],
        dtype=torch.float32,
        device="cuda",
    )
    assert_topk_matches(sample, 2)

    for batch_size, num_elements, k in cases:
        scores = torch.randn((batch_size, num_elements), dtype=torch.float32, device="cuda")
        offsets = torch.arange(num_elements, dtype=torch.float32, device="cuda") * 1.0e-6
        assert_topk_matches(scores + offsets, k, check_indices=True)

    tie_scores = torch.tensor(
        [[1.0, 4.0, 4.0, 3.0, 2.0, 4.0, 0.0, -1.0] * 4],
        dtype=torch.float32,
        device="cuda",
    )
    assert_topk_matches(tie_scores, 6)

    print("correctness: ok")


def time_call(fn, scores: torch.Tensor, k: int, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn(scores, k)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        fn(scores, k)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / iters


def run_benchmark(batch_size: int, num_elements: int, k: int, iters: int) -> None:
    scores = torch.randn((batch_size, num_elements), dtype=torch.float32, device="cuda")
    assert_topk_matches(scores, k)

    radix_ms = time_call(select_radix_cute.topk, scores, k, warmup=3, iters=iters)
    baseline_ms = time_call(baseline_cute.topk, scores, k, warmup=3, iters=iters)

    print(f"benchmark: B={batch_size} N={num_elements} K={k} iters={iters}")
    print(f"radix-select: {radix_ms:.4f} ms")
    print(f"baseline:     {baseline_ms:.4f} ms")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-elements", type=int, default=128)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.num_elements % 32 != 0:
        raise ValueError("--num-elements must be divisible by 32")

    run_correctness()
    if args.bench:
        run_benchmark(args.batch_size, args.num_elements, args.k, args.iters)


if __name__ == "__main__":
    main()
