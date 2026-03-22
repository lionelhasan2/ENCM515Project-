"""
benchmark.py — Matrix multiplication kernel benchmark suite.

Tests multiple matmul implementations on:
  1. Speed sweep: performance vs matrix size (32x32 to 512x512)
  2. Inference workload: 3-layer MLP chain (784→256→128→10)

To add a new kernel, add it to the 'kernels' dict in both
benchmark_speed_sweep() and benchmark_inference_workload().
"""

import numpy as np
import os
import sys

#import example

sys.path.insert(0, os.path.dirname(__file__))

from profiler import measure_matmul, compute_speedups, print_results_table
from kernels import matmul_naive, matmul_simd, matmul_scalar_cpu, matmul_true_simd

# ── IoT-representative layer shapes (MLP for 28x28 image) ────
IOT_LAYER_SHAPES = [
    (1,   784, 256),   # Input layer:   784 features -> 256 hidden
    (1,   256, 128),   # Hidden layer:  256 -> 128
    (1,   128,  10),   # Output layer:  128 -> 10 classes
]

# Matrix sizes for sweep benchmarks
BENCHMARK_SIZES = [32, 64, 128, 256, 512]


def benchmark_speed_sweep():
    """
    Benchmark all kernels across matrix sizes (32x32 to 512x512).
    Prints latency, throughput, and speedup for each size.
    """
    print("\n" + "═"*70)
    print("  WORKLOAD 1: SPEED SWEEP — Kernel Performance vs Matrix Size")
    print("═"*70)

    all_results = []

    # Kernel registry: add new kernels here as they're implemented
    kernels = {
        "Naive": matmul_naive,
        # TODO: team members add more kernels
        "Numpy Scalar": matmul_scalar_cpu,
        "NumPy SIMD": matmul_simd,
        "True SIMD Cython": matmul_true_simd
        # "Cython": matmul_cython,
        # "int8 Quant": matmul_quantized_int8,
    }
    
    #result = example.fibonacci(35)
    #print("Result: ", result)
    #exit(1)

    for size in BENCHMARK_SIZES:
        print(f"\n  Matrix size: {size}x{size}")
        rng = np.random.default_rng(seed=size)
        A = rng.standard_normal((size, size)).astype(np.float32)
        B = rng.standard_normal((size, size)).astype(np.float32)

        size_results = []

        for name, fn in kernels.items():
            r = measure_matmul(fn, A, B, name, runs=5)
            size_results.append(r)
            
        # Compute speedups vs naive
        compute_speedups(size_results, "Naive")
        print_results_table(size_results)
        all_results.extend(size_results)

    return all_results


def benchmark_inference_workload():
    """
    Benchmark kernels on the 3-layer MLP inference workload.
    Measures per-layer and total latency and memory.
    """
    print("\n" + "═"*70)
    print("  WORKLOAD 2: INFERENCE — 3-Layer MLP (784→256→128→10)")
    print("═"*70)

    # Kernel registry: add new kernels here as they're implemented
    kernels = {
        "Naive": matmul_naive,
        "Numpy Scalar": matmul_scalar_cpu,
        "NumPy SIMD": matmul_simd,
        "True SIMD Cython": matmul_true_simd
        # TODO: team members add more kernels
    }

    workload_results = []

    for name, fn in kernels.items():
        layer_results = []
        total_mem_kb = 0

        for M, K, N in IOT_LAYER_SHAPES:
            
            rng = np.random.default_rng(seed=42)
            A = rng.standard_normal((M, K)).astype(np.float32)
            B = rng.standard_normal((K, N)).astype(np.float32)
            
            print("A Matrix Rows: ", len(A), "Matrix Colums: ", len(A[0]))
            print("B Matrix: ", len(B), "Matrix Colums: ", len(B[0]))


            r = measure_matmul(fn, A, B, name, runs=5)
            layer_results.append(r)
            total_mem_kb += r.memory_total_kb

        # Print per-layer results in table format
        print(f"\n  Kernel: {name}")
        print_results_table(layer_results)
        
        # Print total summary
        total_latency = sum(r.latency_ms for r in layer_results)
        avg_gflops = np.mean([r.gflops for r in layer_results])
        print(f"  Total: {total_latency:.4f} ms | Avg GFLOPS: {avg_gflops:.2f} | Memory: {total_mem_kb:.1f} KB\n")

    return workload_results


def benchmark_precision():
    """Placeholder for precision analysis (TBD)."""
    return []


if __name__ == "__main__":

    speed_results  = benchmark_speed_sweep()
    workload_res   = benchmark_inference_workload()
    benchmark_precision()


