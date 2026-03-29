"""
Shared configuration for benchmark, codesign, and visualization scripts.
"""

from kernels import (
    matmul_scalar_cpu, matmul_simd, matmul_true_simd, 
    matmul_true_simd_offset, matmul_true_simd_membank, 
    matmul_quantized_int8, matmul_simd_quantized_int8
)

# Unified workload: Boundary layer matrices (512→256→128→64)
# Small enough to fit in M4 cache with int8, large enough to show real benefits
WORKLOAD = [(1, 512, 256), (1, 256, 128), (1, 128, 64)]
WORKLOAD_NAME = "Boundary Layers (512→256→128→64)"

# Kernel properties: map kernel functions to their metadata
KERNEL_PROPS = {
    matmul_scalar_cpu: {'name': 'Scalar', 'uses_int8': False},
    matmul_simd: {'name': 'SIMD', 'uses_int8': False},
    matmul_true_simd: {'name': 'True SIMD', 'uses_int8': False},
    matmul_true_simd_offset: {'name': 'True SIMD Offset', 'uses_int8': False},
    matmul_true_simd_membank: {'name': 'True SIMD MemBank', 'uses_int8': False},
    matmul_quantized_int8: {'name': 'Int8 Quantized', 'uses_int8': True},
    matmul_simd_quantized_int8: {'name': 'SIMD + Int8', 'uses_int8': True},
}
