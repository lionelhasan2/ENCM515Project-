"""
Profiler: kernel performance metrics for IoT matmul benchmarks.

Measures:
  - Latency: wall-clock execution time per matmul
  - Throughput: GFLOPS (floating-point ops per second)
  - Memory: total working set (A + B + C matrices)

All timing uses time.perf_counter() (standard when hardware counters unavailable).
"""

import time
import numpy as np
from dataclasses import dataclass
from typing import Callable


# ── IoT target hardware parameters ──────────────────────────────────────────
# ARM Cortex-M4 (STM32F407 — common IoT/wearable MCU)
CORTEX_M4_SRAM_KB = 192       # KB — hard memory limit



@dataclass
class ProfileResult:
    """Structured container for a single kernel benchmark result."""
    kernel_name: str
    matrix_shape: tuple          # (M, K, N)
    dtype: str
    runs: int

    # Timing
    latency_ms: float            # Mean wall-clock time per run (ms)
    latency_std_ms: float        # Std dev across runs
    latency_min_ms: float
    latency_max_ms: float

    # Compute throughput
    flops: int                   # Total floating-point ops for one matmul
    gflops: float                # GFLOPS = flops / latency_s / 1e9

    # Memory
    memory_A_kb: float
    memory_B_kb: float
    memory_C_kb: float
    memory_total_kb: float
    fits_cortex_m4: bool         # Whether total fits in 192KB SRAM

    # Arithmetic intensity (FLOPS per byte)
    arithmetic_intensity: float  # Higher AI = better power efficiency
    
    # Number of operations performed (+, *, etc)
    num_operations: int

    # Speedup (filled in post-hoc relative to baseline)
    speedup: float = 1.0


def measure_matmul(
    kernel_fn: Callable,
    A: np.ndarray,
    B: np.ndarray,
    kernel_name: str,
    runs: int = 5,
) -> ProfileResult:
    """
    Profile a matmul kernel by timing multiple runs.

    Parameters
    ----------
    kernel_fn : callable
        (A, B) -> C kernel to benchmark
    A, B : np.ndarray
        Input matrices (A: MxK, B: KxN)
    kernel_name : str
        Label for this kernel in output
    runs : int
        Number of timed repetitions (mean latency computed)

    Returns
    -------
    ProfileResult
        Latency, GFLOPS, memory, and speedup metrics
    """
    M, K = A.shape
    _, N = B.shape

    # Warmup run (not timed) — fills caches, JIT warms up
    _ = kernel_fn(A, B)

    # Timed runs
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        C, numop = kernel_fn(A, B)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times = np.array(times)
    latency_s = float(np.mean(times))
    latency_ms = latency_s * 1000

    # ── Compute metrics ──────────────────────────────────────────────────────
    # Matmul FLOP count: 2*M*N*K (one multiply + one add per inner loop step)
    flops = 2 * M * N * K
    gflops = (flops / latency_s) / 1e9 if latency_s > 0 else 0.0


    # Memory footprint
    mem_A = A.nbytes / 1024
    mem_B = B.nbytes / 1024
    mem_C = (M * N * np.dtype(A.dtype).itemsize) / 1024
    mem_total = mem_A + mem_B + mem_C
    fits_m4 = mem_total <= CORTEX_M4_SRAM_KB

    # Arithmetic intensity
    # Bytes accessed: read A (M*K), read B (K*N), write C (M*N)
    bytes_accessed = A.nbytes + B.nbytes + (M * N * np.dtype(A.dtype).itemsize)
    arith_intensity = flops / bytes_accessed if bytes_accessed > 0 else 0.0

    return ProfileResult(
        kernel_name=kernel_name,
        matrix_shape=(M, K, N),
        dtype=str(A.dtype),
        runs=runs,
        latency_ms=latency_ms,
        latency_std_ms=float(np.std(times) * 1000),
        latency_min_ms=float(np.min(times) * 1000),
        latency_max_ms=float(np.max(times) * 1000),
        flops=flops,
        gflops=gflops,
        memory_A_kb=mem_A,
        memory_B_kb=mem_B,
        memory_C_kb=mem_C,
        memory_total_kb=mem_total,
        fits_cortex_m4=fits_m4,
        arithmetic_intensity=arith_intensity,
        num_operations=numop,
    )


def compute_speedups(results: list[ProfileResult], baseline_name: str) -> list[ProfileResult]:
    """
    Fill in speedup field for each result relative to a named baseline kernel.
    Modifies results in-place.
    """
    baseline = next((r for r in results if r.kernel_name == baseline_name), None)
    if baseline is None:
        raise ValueError(f"Baseline kernel '{baseline_name}' not found in results")

    for r in results:
        r.speedup = baseline.latency_ms / r.latency_ms if r.latency_ms > 0 else 0.0

    return results


def print_results_table(results: list[ProfileResult]):
    """Print results table to console (latency, GFLOPS, speedup, memory, AI)."""
    header = (
        f"\n{'Kernel':<22} {'Shape':>14} {'Latency(ms)':>12} {'GFLOPS':>8} "
        f"{'Speedup':>8} {'Mem(KB)':>8} {'FitsM4':>7} {'AI':>8} {'NumOp':>10}"
    )
    print(header)
    print("─" * len(header))
    for r in results:
        shape_str = f"{r.matrix_shape[0]}x{r.matrix_shape[1]}x{r.matrix_shape[2]}"
        fits = "✓" if r.fits_cortex_m4 else "✗"
        print(
            f"{r.kernel_name:<22} {shape_str:>14} {r.latency_ms:>12.4f} {r.gflops:>8.4f} "
            f"{r.speedup:>8.2f}x {r.memory_total_kb:>8.1f} {fits:>7} {r.arithmetic_intensity:>8.2f} {r.num_operations:>10}"
        )
