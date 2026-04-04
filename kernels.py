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


def matmul_naive(A: np.ndarray, B: np.ndarray) -> tuple:
    """
    Truly naive baseline: Pure Python triple-nested loop, no optimization.
    """
    M, K = A.shape
    K2, N = B.shape    
    result = [[0.0 for _ in range(N)] for _ in range(M)]
    
    numop = 0
    for i in range(M):
        for j in range(N):
            for k in range(K):
                result[i][j] += A[i, k] * B[k, j]
                numop += 1
    
    return np.array(result, dtype=np.float32), numop
  
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

def matmul_simd_quantized_int8(A: np.ndarray, B: np.ndarray) -> np.ndarray:
  """
  SIMD + Int8 Quantized Matrix Multiply: Combines vectorization with quantization.
  """
  return simd.matmul_simd_quantized_int8(A, B)