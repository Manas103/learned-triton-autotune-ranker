"""Benchmark one (config, shape) pair: compile, run once to check it does
not crash, time it. A config that fails to compile or launch (out of
shared memory, an illegal num_stages for a short K loop, etc.) is a real,
expected outcome of a 512-point grid search, not a bug; it is recorded as
invalid rather than silently excluded or treated as a crash."""
import time

import torch

from ranker.gemm_kernel import run_matmul

N_WARMUP = 2
N_TIMED = 10
SEED = 1234


def make_inputs(M, K, N, device="cuda", dtype=torch.float16):
    g = torch.Generator(device="cpu").manual_seed(SEED)
    a = (torch.rand((M, K), generator=g, dtype=torch.float32) * 2 - 1).to(device=device, dtype=dtype)
    b = (torch.rand((K, N), generator=g, dtype=torch.float32) * 2 - 1).to(device=device, dtype=dtype)
    return a, b


def time_config(cfg, a, b):
    """Returns (elapsed_seconds, error) -- error is None on success."""
    try:
        for _ in range(N_WARMUP):
            run_matmul(a, b, cfg)
        torch.cuda.synchronize()
    except Exception as e:
        return None, str(e)[:300]

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    try:
        for _ in range(N_TIMED):
            start.record()
            run_matmul(a, b, cfg)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
    except Exception as e:
        return None, str(e)[:300]
    times.sort()
    return times[len(times) // 2] / 1000.0, None


def wall_clock(fn, *args):
    t0 = time.perf_counter()
    result = fn(*args)
    return result, time.perf_counter() - t0
