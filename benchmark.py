import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from config import WORKLOAD, WORKLOAD_NAME, KERNEL_PROPS
from profiler import measure_matmul, compute_speedups, print_results_table


def benchmark():
    print(f"\n{WORKLOAD_NAME}\n")
    
    # Collect all results across all kernels
    all_results = []
    
    for fn, props in KERNEL_PROPS.items():
        kernel_name = props['name']
        for M, K, N in WORKLOAD:
            rng = np.random.default_rng(seed=42)
            A = rng.standard_normal((M, K)).astype(np.float32)
            B = rng.standard_normal((K, N)).astype(np.float32)
            r = measure_matmul(fn, A, B, kernel_name, runs=5)
            all_results.append(r)
    
    # Compute speedups relative to Scalar baseline
    compute_speedups(all_results, "Scalar")
    
    # Print results grouped by kernel
    for fn, props in KERNEL_PROPS.items():
        kernel_name = props['name']
        print(f"Kernel: {kernel_name}")
        kernel_results = [r for r in all_results if r.kernel_name == kernel_name]
        print_results_table(kernel_results)
        total = sum(r.latency_ms for r in kernel_results)
        print(f"Total: {total:.4f} ms\n")


if __name__ == "__main__":
    benchmark()


