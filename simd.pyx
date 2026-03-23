cimport numpy as np
import numpy as np

"""
The matrix multiply formula: result[i,j] = Sum(A[i,k] * B[k,j])
"""

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

    # Define 8 element wide vectors (registers) to store data
    cdef float reg1[8]
    cdef float reg2[8]

    cdef int i, j, k, y, t

    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            accumulate = 0.0
            for k in range(0, A.shape[1], 8):

                # Fill the reigsters with data from the memory view
                for y in range(8):
                    reg1[y] = reg1_view[i, k + y]
                    reg2[y] = reg2_view[k + y, j]

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

    cdef int i, j, k

    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            for k in range(len(B)):
                result[i, j] += A[i, k] * B[k, j]

    return result


def matmul_true_simd(np.ndarray[np.float32_t, ndim=2] A, 
                     np.ndarray[np.float32_t, ndim=2] B):
    """
    Implements true SIMD matrix multiplicaion using AVX Family + Cython
    """

    result = np.zeros((A.shape[0], B.shape[1]), dtype=np.float32)
    # Transpose B array to avoid row/colum misalignment
    Bt = np.transpose(B)

    cdef float[:, :] reg3_view = A
    cdef float[:, :] reg4_view = Bt
    cdef float[:, :] result_view = result

    cdef __m256 reg3 = _mm256_setzero_ps()
    cdef __m256 reg4 = _mm256_setzero_ps()
    cdef __m256 accumulate = _mm256_setzero_ps()

    # In order to store the resulting scalar value after SIMD operations, 
    cdef float tmp_reg[8]

    cdef int i, j, k

    for i in range(A.shape[0]):
        for j in range(Bt.shape[0]):
            accumulate = _mm256_setzero_ps()
            for k in range(0, A.shape[1], 8):
                # k in being indexed by 8 to ensure we are not using redundant data

                # Load all 8 datapoints in a single instruction
                reg3 = _mm256_loadu_ps(&reg3_view[i, k]) # Loads A[i, k] -> A[i, k+7]
                reg4 = _mm256_loadu_ps(&reg4_view[j, k]) # Loads Bt[j, k] -> Bt[j, k+7]
                
                # Multiply and add both registers
                accumulate = _mm256_fmadd_ps(reg3, reg4, accumulate)

            # Store data into a temporary vector so it can be changed into a scalar value later
            _mm256_storeu_ps(tmp_reg, accumulate)
            result_view[i, j] = (tmp_reg[0] + tmp_reg[1] + tmp_reg[2] + tmp_reg[3] +
                                 tmp_reg[4] + tmp_reg[5] + tmp_reg[6] + tmp_reg[7])

    return result


def matmul_true_simd_offset(np.ndarray[np.float32_t, ndim=2] A, 
                             np.ndarray[np.float32_t, ndim=2] B):
    """
    Implements true SIMD matrix multiplicaion using AVX Family + Cython.
    Implements a simulated offset to observe performance with memeory access misalignment.
    """

    cdef int offset = 4

    result = np.zeros((A.shape[0], B.shape[1]), dtype=np.float32)
    # Transpose B array to avoid row/colum misalignment
    Bt = np.transpose(B)

    cdef float[:, :] reg3_view = A
    cdef float[:, :] reg4_view = Bt
    cdef float[:, :] result_view = result

    cdef __m256 reg3 = _mm256_setzero_ps()
    cdef __m256 reg4 = _mm256_setzero_ps()
    cdef __m256 accumulate = _mm256_setzero_ps()

    # In order to store the resulting scalar value after SIMD operations, 
    cdef float tmp_reg[8]

    cdef int i, j, k

    for i in range(A.shape[0]):
        for j in range(Bt.shape[0]):
            accumulate = _mm256_setzero_ps()
            for k in range(0, (A.shape[1] - offset), 8):
                # k in being indexed by 8 to ensure we are not using redundant data

                # Load all 8 datapoints in a single instruction
                reg3 = _mm256_loadu_ps(&reg3_view[i, k + offset]) # Loads A[i, k] -> A[i, k+7]
                reg4 = _mm256_loadu_ps(&reg4_view[j, k + offset])
                
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
    Simulates a memory access offset and "memory bank" to detect misalignment.
    """

    cdef int offset = 4

    result = np.zeros((A.shape[0], B.shape[1]), dtype=np.float32)
    # Transpose B array to avoid row/colum misalignment
    Bt = np.transpose(B)

    cdef float[:, :] reg3_view = A
    cdef float[:, :] reg4_view = Bt
    cdef float[:, :] result_view = result

    cdef __m256 reg3 = _mm256_setzero_ps()
    cdef __m256 reg4 = _mm256_setzero_ps()
    cdef __m256 accumulate = _mm256_setzero_ps()

    # In order to store the resulting scalar value after SIMD operations, 
    cdef float tmp_reg[8]

    cdef int i, j, k, indx

    for i in range(A.shape[0]):
        for j in range(Bt.shape[0]):
            accumulate = _mm256_setzero_ps()
            for k in range(0, (A.shape[1] - offset), 8):
                # k in being indexed by 8 to ensure we are not using redundant data

                # Check if the offset is divisible by 32
                if (k * 4) % 32 != 0:
                    indx = k
                else:
                    indx = k

                # Load all 8 datapoints in a single instruction
                reg3 = _mm256_loadu_ps(&reg3_view[i, indx + offset])
                reg4 = _mm256_loadu_ps(&reg4_view[j, indx + offset])
                
                # Multiply and add both registers
                accumulate = _mm256_fmadd_ps(reg3, reg4, accumulate)

            # Store data into a temporary vector so it can be changed into a scalar value later
            _mm256_storeu_ps(tmp_reg, accumulate)
            result_view[i, j] = (tmp_reg[0] + tmp_reg[1] + tmp_reg[2] + tmp_reg[3] +
                                 tmp_reg[4] + tmp_reg[5] + tmp_reg[6] + tmp_reg[7])

    return result