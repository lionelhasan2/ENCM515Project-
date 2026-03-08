"""
Matmul kernel implementations.

Each kernel must have signature: (A: ndarray, B: ndarray) -> ndarray
representing C = A @ B where:
  - A: (M, K) matrix
  - B: (K, N) matrix
  - C: (M, N) result
"""

import numpy as np


def matmul_naive(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Naive baseline: numpy's @ operator (compiled C code, no hand tuning).
    """
    # M, K = A.shape
    # K2, N = B.shape
    # assert K == K2, f"Inner dimensions must match: {K} vs {K2}"
    
    return A @ B
