"""
Matmul kernel implementations.

Each kernel must have signature: (A: ndarray, B: ndarray) -> ndarray
representing C = A @ B where:
  - A: (M, K) matrix
  - B: (K, N) matrix
  - C: (M, N) result
"""

import numpy as np
import simd


def matmul_naive(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Naive baseline: numpy's @ operator (compiled C code, no hand tuning).
    """
    # M, K = A.shape
    # K2, N = B.shape
    # assert K == K2, f"Inner dimensions must match: {K} vs {K2}"
    
    return A @ B
  
def matmul_simd(A: np.ndarray, B: np.ndarray) -> np.ndarray:
  """
  Simulated SIMD Matrix Multiply using Vectorized Loops.
  """
  return simd.matmul_simd(A, B)

def matmul_scalar_cpu(A: np.ndarray, B: np.ndarray) -> np.ndarray:
  """
  Scalar/CPU Matrix Multiply.
  """
  return simd.matmul_scalar(A, B)

def matmul_true_simd(A: np.ndarray, B: np.ndarray) -> np.ndarray:
  """
  Implementing true SIMD matrix multiplicaion using AVX Family + Cython.
  """
  return simd.matmul_true_simd(A, B)

def matmul_true_simd_offset(A: np.ndarray, B: np.ndarray) -> np.ndarray:
  """
  Implements true SIMD matrix multiplicaion using AVX Family + Cython.
  Implements a simulated offset to observe performance with memeory access misalignment.
  """
  return simd.matmul_true_simd_offset(A, B)
    
def matmul_true_simd_membank(A: np.ndarray, B: np.ndarray) -> np.ndarray:
  """
  Implementing true SIMD matrix multiplicaion using AVX Family + Cython.
  Simulates a memory access offset and "memory bank" to detect misalignment.
  """
  return simd.matmul_true_simd_membank(A, B)

def matmul_quantized_int8(A: np.ndarray, B: np.ndarray) -> np.ndarray:
  """
  Int8 Quantized Matrix Multiply: Simulates quantization for embedded deployment.
  """
  return simd.matmul_quantized_int8(A, B)