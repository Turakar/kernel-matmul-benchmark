import torch
from torch import Tensor

from kernel_matmul_experiments.benchmark.implementations import MatmulBase


def is_oom_exception(exception: Exception) -> bool:
    return (
        isinstance(exception, torch.cuda.OutOfMemoryError)
        or ("CUDA out of memory" in str(exception))
        or ("[KeOps] Cuda error." in str(exception))
    )


def timeit(impl: MatmulBase, rhs: Tensor, iterations: int) -> Tensor:
    # Warmup (compiling, autotuning, ...)
    impl(rhs)

    # Benchmark
    timings = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iterations):
        start.record()
        impl(rhs)
        end.record()
        torch.cuda.synchronize()
        timings.append(start.elapsed_time(end))
    timings = torch.tensor(timings)
    return timings
