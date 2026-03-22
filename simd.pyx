cimport numpy as np
import numpy as np

"""
External call to generate real SIMD instructions.
"""
cdef extern from "immintrin.h":
    ctypedef float __m256 # Declares the datatype for SIMD vectors of 8 elements

    __m256 _mm256_loadu_ps(float*) # Loads multiple data points at once

    __m256 _mm256_fmadd_ps(__m256, __m256, __m256) #  Vectoized multiply and add

    __m256 _mm256_setzero_ps() # Creates vectors set with zeroes

    void _mm256_storeu_ps(float*, __m256) # Stores vector data as a scalar value


def matmul_simd(np.ndarray[np.float32_t, ndim=2] A, 
                np.ndarray[np.float32_t, ndim=2] B):
"""
Simulated SIMD Matrix Multiply using Vectorized Loops.
"""
    
    result = np.zeros((A.shape[0], B.shape[1]), dtype=np.float32) # Declare numpy array for results

    # Typed Memory Views: Efficient access to memory buffers, such as those underlying NumPy arrays, without incurring any Python overhead.
    cdef float[:, :] reg1_view = A
    cdef float[:, :] reg2_view = B
    cdef float[:, :] result_view = result

    # Define 8 element wide vecotrs (registers) to store data
    cdef float reg1[8]
    cdef float reg2[8]

    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            accumulate = 0.0
            for k in range(0, A.shape[1], 8):

                # Fill the reigsters with data from the memory view
                for t in range(8):
                    reg1[t] = reg1_view[i, k + t]
                    reg2[t] = reg2_view[k + t, j]

                # Multiply and accumualate
                for t in range(8):
                    accumulate += reg1[t] * reg2[t]

            result_view[i, j] = accumulate

    return result


def matmul_scalar(np.ndarray[np.float32_t, ndim=2] A, 
                  np.ndarray[np.float32_t, ndim=2] B):
    """
    Scalar/CPU Matrix Multiply.
    """

    result = np.zeros((A.shape[0], B.shape[1]), dtype=np.float32) # Declare numpy array for results

    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            for k in range(len(B)):
                result[i, j] += A[i, k] * B[k, j]

    return result

def matmul_true_simd(np.ndarray[np.float32_t, ndim=2] A, 
                     np.ndarray[np.float32_t, ndim=2] B):
    """
    Implementing true SIMD matrix multiplicaion using AVX Family + Cython
    """

    result = np.zeros((A.shape[0], B.shape[1]), dtype=np.float32)

    cdef float[:, :] reg3_view = A
    cdef float[:, :] reg4_view = B
    cdef float[:, :] result_view = result

    cdef __m256 reg3 = _mm256_setzero_ps()
    cdef __m256 reg4 = _mm256_setzero_ps()
    cdef __m256 accumulate = _mm256_setzero_ps()

    # In order to store the resulting scalar value after SIMD operations, 
    cdef float tmp_reg[8]

    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            for k in range(0, A.shape[1], 8):
                # k in being indexed by 8 to ensure we are not using redundant data

                # Load all 8 datapoints in a single instruction
                reg3 = _mm256_loadu_ps(&reg3_view[i, k]) # Loads A[i, k] -> A[i, k+7]
                reg4 = _mm256_loadu_ps(&reg4_view[k, j]) # Loads B[i, k] -> B[i, k+7]
                
                # Multiply and add both registers
                accumulate = _mm256_fmadd_ps(reg3, reg4, accumulate)

            # Store data into a temporary vector so it can be changed into a scalar value later
            _mm256_storeu_ps(tmp_reg, accumulate)
            result_view[i, j] = (tmp_reg[0] + tmp_reg[1] + tmp_reg[2] + tmp_reg[3] +
                                 tmp_reg[4] + tmp_reg[5] + tmp_reg[6] + tmp_reg[7])

    return result

def matmul_true_simd_membank(np.ndarray[np.float32_t, ndim=2] A, 
                             np.ndarray[np.float32_t, ndim=2] B):
    """
    Implementing true SIMD matrix multiplicaion using AVX Family + Cython. 
    This version includes a memory bank that corrects misaligned memory offsets.
    """

    result = np.zeros((A.shape[0], B.shape[1]), dtype=np.float32)

    cdef bool offset = false

    cdef float[:, :] reg3_view = A
    cdef float[:, :] reg4_view = B
    cdef float[:, :] result_view = result

    cdef __m256 reg3 = _mm256_setzero_ps()
    cdef __m256 reg4 = _mm256_setzero_ps()
    cdef __m256 accumulate = _mm256_setzero_ps()

    # In order to store the resulting scalar value after SIMD operations, 
    cdef float tmp_reg[8]

    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            for k in range(0, A.shape[1], 8):

                # check for offset and shuffle to the correct lanes
                # the way the loops are being done right now, we are guarunteed an
                # offset nearly every time, how can we correct this (mem bank)

                # Load all 8 datapoints in a single instruction
                reg3 = _mm256_loadu_ps(&reg3_view[i, k])
                reg4 = _mm256_loadu_ps(&reg4_view[k, j])
                
                # Multiply and add both registers using single instruction
                accumulate = _mm256_fmadd_ps(reg3, reg4, accumulate)

            # Store data into a temporary vector so it can be changed into a scalar value later
            _mm256_storeu_ps(tmp_reg, accumulate)
            result_view[i, j] = (tmp_reg[0] + tmp_reg[1] + tmp_reg[2] + tmp_reg[3] +
                                 tmp_reg[4] + tmp_reg[5] + tmp_reg[6] + tmp_reg[7])

    return result