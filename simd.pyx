cimport numpy as np
import numpy as np
import random

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
    cdef int numop = 0

    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            accumulate = 0.0
            for k in range(0, A.shape[1], 8):

                # Fill the reigsters with data from the memory view
                for y in range(8):
                    reg1[y] = reg1_view[i, k + y] # row A
                    reg2[y] = reg2_view[k + y, j] # colum B

                for t in range(8): # Tried np.sum(np.mul()) and it was significantly slower
                    accumulate += reg1[t] * reg2[t]
                    numop = numop + 1

            result_view[i, j] = accumulate

    #print("Number of Operations for Vectoized MatMul: ", numop)
    #print(result)
    return result, numop


def matmul_scalar(np.ndarray[np.float32_t, ndim=2] A, 
                  np.ndarray[np.float32_t, ndim=2] B):
    """
    Scalar/CPU Matrix Multiply.
    """

    result = np.zeros((A.shape[0], B.shape[1]), dtype=np.float32) # Declare numpy array for results

    cdef int i, j, k
    cdef int numop = 0

    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            for k in range(len(B)):
                result[i, j] += A[i, k] * B[k, j]
                numop = numop + 1

    #print("Number of Operations for Scalar MatMul: ", numop)
    #print(result)
    return result, numop


def matmul_true_simd(np.ndarray[np.float32_t, ndim=2] A, 
                     np.ndarray[np.float32_t, ndim=2] B):
    """
    Implements true SIMD matrix multiplicaion using AVX Family + Cython
    """

    result = np.zeros((A.shape[0], B.shape[1]), dtype=np.float32)
    Bt = np.transpose(B) # Transpose B array to avoid row/colum misalignment
    Bt = np.ascontiguousarray(Bt) # Ensures the array is stored as a continuous block after transpose

    cdef float[:, :] reg3_view = A
    cdef float[:, :] reg4_view = Bt
    cdef float[:, :] result_view = result

    cdef __m256 reg3 = _mm256_setzero_ps()
    cdef __m256 reg4 = _mm256_setzero_ps()
    cdef __m256 accumulate = _mm256_setzero_ps()

    # Vector to store the resulting scalar value after SIMD operations 
    cdef float tmp_reg[8]

    cdef int i, j, k
    cdef int numop = 0

    for i in range(A.shape[0]):
        for j in range(Bt.shape[0]):
            accumulate = _mm256_setzero_ps()
            for k in range(0, Bt.shape[1], 8):
                # k in being indexed by 8 to ensure we are not using redundant data

                # Load all 8 datapoints in a single instruction
                reg3 = _mm256_loadu_ps(&reg3_view[i, k]) # Loads A[i, k] -> A[i, k+7]
                reg4 = _mm256_loadu_ps(&reg4_view[j, k]) # Loads Bt[j, k] -> Bt[j, k+7]
                
                # Multiply and add both registers
                accumulate = _mm256_fmadd_ps(reg3, reg4, accumulate)
                numop = numop + 1

            # Store data into a temporary vector so it can be changed into a scalar value later
            _mm256_storeu_ps(tmp_reg, accumulate)
            result_view[i, j] = (tmp_reg[0] + tmp_reg[1] + tmp_reg[2] + tmp_reg[3] +
                                 tmp_reg[4] + tmp_reg[5] + tmp_reg[6] + tmp_reg[7])

    #print("Number of Operations for SIMD MatMul: ", numop)
    #print(result)
    return result, numop


def matmul_true_simd_offset(np.ndarray[np.float32_t, ndim=2] A, 
                             np.ndarray[np.float32_t, ndim=2] B):
    """
    Implements true SIMD matrix multiplicaion using AVX Family + Cython.
    Implements a simulated offset to observe performance with memeory access misalignment.
    """

    result = np.zeros((A.shape[0], B.shape[1]), dtype=np.float32)
    # Transpose B array to avoid row/colum misalignment
    Bt = np.transpose(B)
    Bt = np.ascontiguousarray(Bt)

    cdef float[:, :] reg3_view = A
    cdef float[:, :] reg4_view = Bt
    cdef float[:, :] result_view = result

    cdef __m256 reg3 = _mm256_setzero_ps()
    cdef __m256 reg4 = _mm256_setzero_ps()
    cdef __m256 accumulate = _mm256_setzero_ps()

    # Vector to store the resulting scalar value after SIMD operations 
    cdef float tmp_reg[8]

    cdef int i, j, k
    cdef int numop = 0
    cdef int offset = random.randint(1, 7)

    for i in range(A.shape[0]):
        for j in range(Bt.shape[0]):
            accumulate = _mm256_setzero_ps()
            for k in range(0, (Bt.shape[1] - offset - 7), 8):
                # k in being indexed by 8 to ensure we are not using redundant data

                # Load all 8 datapoints in a single instruction
                reg3 = _mm256_loadu_ps(&reg3_view[i, k + offset]) # Loads A[i, k] -> A[i, k+7]
                reg4 = _mm256_loadu_ps(&reg4_view[j, k + offset])
                
                # Multiply and add both registers
                accumulate = _mm256_fmadd_ps(reg3, reg4, accumulate)
                numop = numop + 1

            # Store data into a temporary vector so it can be changed into a scalar value later
            _mm256_storeu_ps(tmp_reg, accumulate)
            result_view[i, j] = (tmp_reg[0] + tmp_reg[1] + tmp_reg[2] + tmp_reg[3] +
                                 tmp_reg[4] + tmp_reg[5] + tmp_reg[6] + tmp_reg[7])

    #print("Number of Operations for Offset SIMD MatMul: ", numop)
    #print(result)
    return result, numop


def matmul_true_simd_membank(np.ndarray[np.float32_t, ndim=2] A, 
                             np.ndarray[np.float32_t, ndim=2] B):
    """
    Implementing true SIMD matrix multiplicaion using AVX Family + Cython.
    Simulates a memory access offset and "memory bank" to detect misalignment.
    """

    result = np.zeros((A.shape[0], B.shape[1]), dtype=np.float32)
    # Transpose B array to avoid row/colum misalignment
    Bt = np.transpose(B)
    Bt = np.ascontiguousarray(Bt)

    cdef float[:, :] reg3_view = A
    cdef float[:, :] reg4_view = Bt
    cdef float[:, :] result_view = result

    cdef __m256 reg3 = _mm256_setzero_ps()
    cdef __m256 reg4 = _mm256_setzero_ps()
    cdef __m256 accumulate = _mm256_setzero_ps()

    # Vector to store the resulting scalar value after SIMD operations 
    cdef float tmp_reg[8]

    cdef int i, j, indx
    cdef int numop = 0
    cdef int offset = random.randint(1, 7)
    cdef int k = offset

    for i in range(A.shape[0]):
        for j in range(Bt.shape[0]):
            accumulate = _mm256_setzero_ps()
            for k in range(0, (Bt.shape[1] - offset), 8):
                # k in being indexed by 8 to ensure we are not using redundant data

                # Check if the offset is divisible by 32
                if (k * 4) % 32 != 0:
                    indx = k - offset
                else:
                    indx = k
                
                # Load all 8 datapoints in a single instruction
                reg3 = _mm256_loadu_ps(&reg3_view[i, indx]) # Loads A[i, k] -> A[i, k+7]
                reg4 = _mm256_loadu_ps(&reg4_view[j, indx])
                
                # Multiply and add both registers
                accumulate = _mm256_fmadd_ps(reg3, reg4, accumulate)
                numop = numop + 1

            # Store data into a temporary vector so it can be changed into a scalar value later
            _mm256_storeu_ps(tmp_reg, accumulate)
            result_view[i, j] = (tmp_reg[0] + tmp_reg[1] + tmp_reg[2] + tmp_reg[3] +
                                 tmp_reg[4] + tmp_reg[5] + tmp_reg[6] + tmp_reg[7])

    #print("Number of Operations for MemBank SIMD MatMul: ", numop)
    #print(result)
    return result, numop


def matmul_quantized_int8(np.ndarray[np.float32_t, ndim=2] A, 
                          np.ndarray[np.float32_t, ndim=2] B):
    """
    Int8 Quantized Matrix Multiply
    """

    # get scale to map float to int8
    scale_a = np.max(np.abs(A)) / 127.0
    scale_b = np.max(np.abs(B)) / 127.0

    if scale_a == 0:
        scale_a = 1.0
    if scale_b == 0:
        scale_b = 1.0

    # quantize
    A_quant = np.clip(np.round(A / scale_a), -128, 127).astype(np.int8)
    B_quant = np.clip(np.round(B / scale_b), -128, 127).astype(np.int8)

    result_int = np.zeros((A.shape[0], B.shape[1]), dtype=np.int32)

    # use memory views for speed
    cdef char[:, :] a_view = A_quant
    cdef char[:, :] b_view = B_quant
    cdef int[:, :] result_view = result_int

    cdef int i, j, k
    cdef int numop = 0
    cdef int accumulate

    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            accumulate = 0
            for k in range(len(B)):
                accumulate += a_view[i, k] * b_view[k, j]
                numop = numop + 1
            
            result_view[i, j] = accumulate

    result = result_int.astype(np.float32) * (scale_a * scale_b)

    #print("Number of Operations for Quantized MatMul: ", numop)
    #print(result)
    return result, numop


def matmul_simd_quantized_int8(np.ndarray[np.float32_t, ndim=2] A, 
                               np.ndarray[np.float32_t, ndim=2] B):
    """
    SIMD + Quantized Int8 Matrix Multiply (AVX).
    Combines quantization (reduced memory) with SIMD vectorization.
    """

    scale_a = np.max(np.abs(A)) / 127.0
    scale_b = np.max(np.abs(B)) / 127.0

    if scale_a == 0:
        scale_a = 1.0
    if scale_b == 0:
        scale_b = 1.0

    A_quant = np.clip(np.round(A / scale_a), -128, 127).astype(np.int8)
    B_quant = np.clip(np.round(B / scale_b), -128, 127).astype(np.int8)

    result_int = np.zeros((A.shape[0], B.shape[1]), dtype=np.int32)

    cdef char[:, :] a_view = A_quant
    cdef char[:, :] b_view = B_quant
    cdef int[:, :] result_view = result_int

    cdef int reg1[8]
    cdef int reg2[8]

    cdef int i, j, k, y, t
    cdef int numop = 0

    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            accumulate_int = 0
            for k in range(0, A.shape[1], 8):
                for y in range(8):
                    reg1[y] = a_view[i, k + y]
                    reg2[y] = b_view[k + y, j]
                
                for t in range(8):
                    accumulate_int += reg1[t] * reg2[t]
                    numop = numop + 1
            
            result_view[i, j] = accumulate_int

    result = result_int.astype(np.float32) * (scale_a * scale_b)
    return result, numop
