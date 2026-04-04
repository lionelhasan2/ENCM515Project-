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

    ctypedef float __m512

    __m512 _mm512_loadu_ps(__m512*)

    __m512 _mm512_setzero_ps()

    __m512 _mm512_fmadd_ps(__m512, __m512, __m512)

    void _mm512_storeu_ps(float*, __m512) # Stores vector data as a scalar value   

    ctypedef struct __m128i:
        pass

    __m128i _mm_loadu_si128(void * addr)  

    __m128i _mm_add_epi32(__m128i, __m128i)   

    __m128i _mm_cvtepi16_epi32(__m128i)

    __m128i _mm_cvtepi8_epi16(__m128i)

    __m128i _mm_mullo_epi16(__m128i, __m128i)

    __m128i _mm_setzero_si128()

    __m128i _mm_storeu_si128(void * addr, __m128i)

    __m128i _mm_srli_si128(__m128i, int)

    ctypedef struct __m512i:
        pass

    __m512i _mm512_setzero_si512()

    __m512i _mm512_loadu_si512(void * mem_addr)

    void _mm512_storeu_si512(void * mem_addr, __m512i a)

    __m512i _mm512_madd_epi16(__m512i a, __m512i b)

    __m512i _mm512_add_epi32(__m512i, __m512i)


def matmul_simd(np.ndarray[np.float32_t, ndim=2] A, 
                np.ndarray[np.float32_t, ndim=2] B):
    """
    Simulated SIMD Matrix Multiply using Vectorized Loops.
    """

    # Add accumulate
    # time the inner and outer loops to identify timing
    # make functions as similar as possible
    
    result = np.zeros((A.shape[0], B.shape[1]), dtype=np.float32) # Declare numpy array for results

    # Typed Memory Views: Efficient access to memory buffers, such as those underlying NumPy arrays, without incurring any Python overhead.
    cdef float[:, :] reg1_view = A
    cdef float[:, :] reg2_view = B
    cdef float[:, :] result_view = result

    # Define 8 element wide vectors (registers) to store data
    cdef float reg1[16]
    cdef float reg2[16]

    cdef int i, j, k, y, t
    cdef int numop = 0

    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            accumulate = 0.0
            for k in range(0, A.shape[1] - (A.shape[1] % 16), 16):

                # Fill the reigsters with data from the memory view
                for y in range(16):
                    reg1[y] = reg1_view[i, k + y] # row A
                    reg2[y] = reg2_view[k + y, j] # colum B
                    numop = numop + 1

                for t in range(16): # Tried np.sum(np.mul()) and it was significantly slower
                    accumulate += reg1[t] * reg2[t]
                    numop = numop + 1

            # tail (<16 leftover)
            for k in range(A.shape[1] - (A.shape[1] % 16), A.shape[1]):
                accumulate += reg1_view[i, k] * reg2_view[k, j]
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

    cdef float[:, :] A_view = A
    cdef float[:, :] B_view = B
    cdef float[:, :] result_view = result

    cdef int i, j, k
    cdef int numop = 0

    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            accumulate = 0.0
            for k in range(len(B)):
                accumulate += A_view[i, k] * B_view[k, j]
                numop = numop + 1
            result_view[i, j] = accumulate

    #print("Number of Operations for Scalar MatMul: ", numop)
    #print(result)
    return result, numop


def matmul_true_simd(np.ndarray[np.float32_t, ndim=2] A, 
                     np.ndarray[np.float32_t, ndim=2] B):
    """
    Implements true SIMD matrix multiplicaion using AVX + 512 Family + Cython
    """

    result = np.zeros((A.shape[0], B.shape[1]), dtype=np.float32)
    Bt = np.transpose(B) # Transpose B array to avoid row/colum misalignment
    Bt = np.ascontiguousarray(Bt) # Ensures the array is stored as a continuous block after transpose

    cdef float[:, :] reg3_view = A
    cdef float[:, :] reg4_view = Bt
    cdef float[:, :] result_view = result

    cdef __m512 reg3 = _mm512_setzero_ps()
    cdef __m512 reg4 = _mm512_setzero_ps()
    cdef __m512 accumulate = _mm512_setzero_ps()

    # Vector to store the resulting scalar value after SIMD operations 
    cdef float tmp_reg[16]

    cdef int i, j, k
    cdef int numop = 0

    for i in range(A.shape[0]):
        for j in range(Bt.shape[0]):
            accumulate = _mm512_setzero_ps()
            for k in range(0, Bt.shape[1], 16):
                # k in being indexed by 8 to ensure we are not using redundant data

                # Load all 8 datapoints in a single instruction
                reg3 = _mm512_loadu_ps(&reg3_view[i, k]) # Loads A[i, k] -> A[i, k+7]
                reg4 = _mm512_loadu_ps(&reg4_view[j, k]) # Loads Bt[j, k] -> Bt[j, k+7]
                
                # Multiply and add both registers
                accumulate = _mm512_fmadd_ps(reg3, reg4, accumulate)
                numop = numop + 1

            # Store data into a temporary vector so it can be changed into a scalar value later
            _mm512_storeu_ps(tmp_reg, accumulate)
            result_view[i, j] = (tmp_reg[0] + tmp_reg[1] + tmp_reg[2] + tmp_reg[3] +
                                 tmp_reg[4] + tmp_reg[5] + tmp_reg[6] + tmp_reg[7] +
                                 tmp_reg[8] + tmp_reg[9] + tmp_reg[10] + tmp_reg[11] +
                                 tmp_reg[12] + tmp_reg[13] + tmp_reg[14] + tmp_reg[15])

    #print("Number of Operations for SIMD MatMul: ", numop)
    #print(result)
    return result, numop


def matmul_true_simd_offset(np.ndarray[np.float32_t, ndim=2] A, 
                             np.ndarray[np.float32_t, ndim=2] B):
    """
    Implements true SIMD matrix multiplicaion using AVX + 512 Family + Cython.
    Implements a simulated offset to observe performance with memeory access misalignment.
    """

    result = np.zeros((A.shape[0], B.shape[1]), dtype=np.float32)
    # Transpose B array to avoid row/colum misalignment
    Bt = np.transpose(B)
    Bt = np.ascontiguousarray(Bt)

    cdef float[:, :] reg3_view = A
    cdef float[:, :] reg4_view = Bt
    cdef float[:, :] result_view = result

    cdef __m512 reg3 = _mm512_setzero_ps()
    cdef __m512 reg4 = _mm512_setzero_ps()
    cdef __m512 accumulate = _mm512_setzero_ps()

    # Vector to store the resulting scalar value after SIMD operations 
    cdef float tmp_reg[16]

    cdef int i, j, k
    cdef int numop = 0
    cdef int offset = random.randint(1, 7)

    for i in range(A.shape[0]):
        for j in range(Bt.shape[0]):
            accumulate = _mm512_setzero_ps()
            for k in range(0, (Bt.shape[1] - offset - 16), 16):
                # k in being indexed by 8 to ensure we are not using redundant data

                # Load all 8 datapoints in a single instruction
                reg3 = _mm512_loadu_ps(&reg3_view[i, k + offset]) # Loads A[i, k] -> A[i, k+7]
                reg4 = _mm512_loadu_ps(&reg4_view[j, k + offset])
                
                # Multiply and add both registers
                accumulate = _mm512_fmadd_ps(reg3, reg4, accumulate)
                numop = numop + 1

            # Store data into a temporary vector so it can be changed into a scalar value later
            _mm512_storeu_ps(tmp_reg, accumulate)
            result_view[i, j] = (tmp_reg[0] + tmp_reg[1] + tmp_reg[2] + tmp_reg[3] +
                                 tmp_reg[4] + tmp_reg[5] + tmp_reg[6] + tmp_reg[7] +
                                 tmp_reg[8] + tmp_reg[9] + tmp_reg[10] + tmp_reg[11] +
                                 tmp_reg[12] + tmp_reg[13] + tmp_reg[14] + tmp_reg[15])

    #print("Number of Operations for Offset SIMD MatMul: ", numop)
    #print(result)
    return result, numop


def matmul_true_simd_membank(np.ndarray[np.float32_t, ndim=2] A, 
                             np.ndarray[np.float32_t, ndim=2] B):
    """
    Implementing true SIMD matrix multiplicaion using AVX + 512 Family + Cython.
    Simulates a memory access offset and "memory bank" to detect misalignment.
    """

    result = np.zeros((A.shape[0], B.shape[1]), dtype=np.float32)
    # Transpose B array to avoid row/colum misalignment
    Bt = np.transpose(B)
    Bt = np.ascontiguousarray(Bt)

    cdef float[:, :] reg3_view = A
    cdef float[:, :] reg4_view = Bt
    cdef float[:, :] result_view = result

    cdef __m512 reg3 = _mm512_setzero_ps()
    cdef __m512 reg4 = _mm512_setzero_ps()
    cdef __m512 accumulate = _mm512_setzero_ps()

    # Vector to store the resulting scalar value after SIMD operations 
    cdef float tmp_reg[16]

    cdef int i, j, indx
    cdef int numop = 0
    cdef int offset = random.randint(1, 7)
    cdef int k = offset

    for i in range(A.shape[0]):
        for j in range(Bt.shape[0]):
            accumulate = _mm512_setzero_ps()
            for k in range(0, (Bt.shape[1] - offset), 16):
                # k in being indexed by 8 to ensure we are not using redundant data

                # Check if the offset is divisible by 32
                if (k * 4) % 32 != 0:
                    indx = k - offset
                else:
                    indx = k
                
                # Load all 8 datapoints in a single instruction
                reg3 = _mm512_loadu_ps(&reg3_view[i, indx]) # Loads A[i, k] -> A[i, k+7]
                reg4 = _mm512_loadu_ps(&reg4_view[j, indx])
                
                # Multiply and add both registers
                accumulate = _mm512_fmadd_ps(reg3, reg4, accumulate)
                numop = numop + 1

            # Store data into a temporary vector so it can be changed into a scalar value later
            _mm512_storeu_ps(tmp_reg, accumulate)
            result_view[i, j] = (tmp_reg[0] + tmp_reg[1] + tmp_reg[2] + tmp_reg[3] +
                                 tmp_reg[4] + tmp_reg[5] + tmp_reg[6] + tmp_reg[7] +
                                 tmp_reg[8] + tmp_reg[9] + tmp_reg[10] + tmp_reg[11] +
                                 tmp_reg[12] + tmp_reg[13] + tmp_reg[14] + tmp_reg[15])

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
    SIMD + Quantized Int8 Matrix Multiply (SSE).
    Combines quantization (reduced memory) with SIMD vectorization.
    """
    result_int = np.zeros((A.shape[0], B.shape[1]), dtype=np.int32)

    Bt = np.transpose(B)
    Bt = np.ascontiguousarray(Bt)

    scale_a = np.max(np.abs(A)) / 127.0
    scale_b = np.max(np.abs(Bt)) / 127.0

    if scale_a == 0:
        scale_a = 1.0
    if scale_b == 0:
        scale_b = 1.0

    A_quant = np.clip(np.round(A / scale_a), -128, 127).astype(np.int8)
    B_quant = np.clip(np.round(Bt / scale_b), -128, 127).astype(np.int8)

    cdef char[:, :] a_view = A_quant
    cdef char[:, :] b_view = B_quant
    cdef int[:, :] result_view = result_int

    cdef __m128i a_reg = _mm_setzero_si128()
    cdef __m128i b_reg = _mm_setzero_si128()

    cdef __m128i acc_lo = _mm_setzero_si128()
    cdef __m128i acc_hi = _mm_setzero_si128()
    
    cdef __m128i a_lo
    cdef __m128i a_hi

    cdef __m128i b_lo
    cdef __m128i b_hi

    cdef __m128i a_16_lo
    cdef __m128i b_16_hi

    cdef __m128i prod16_lo
    cdef __m128i prod16_hi

    cdef __m128i prod32_lo
    cdef __m128i prod32_hi

    cdef int tmp_lo[8]
    cdef int tmp_hi[8]

    cdef int i, j, k
    cdef int numop = 0

    for i in range(A.shape[0]):
        for j in range(Bt.shape[0]):
            acc_lo = _mm_setzero_si128()
            acc_hi = _mm_setzero_si128()
            for k in range(0, Bt.shape[1], 16):
            
                a_reg = _mm_loadu_si128(<void*> &a_view[i, k])
                b_reg = _mm_loadu_si128(<void*> &b_view[j, k])

                # split into low/high halves
                a_lo = a_reg
                a_hi = _mm_srli_si128(a_reg, 8)

                b_lo = b_reg
                b_hi = _mm_srli_si128(b_reg, 8)

                # widen
                a_16_lo = _mm_cvtepi8_epi16(a_lo)
                a_16_hi = _mm_cvtepi8_epi16(a_hi)

                b_16_lo = _mm_cvtepi8_epi16(b_lo)
                b_16_hi = _mm_cvtepi8_epi16(b_hi)

                # multiply
                prod16_lo = _mm_mullo_epi16(a_16_lo, b_16_lo)
                prod16_hi = _mm_mullo_epi16(a_16_hi, b_16_hi)

                # widen to int32
                prod32_lo = _mm_cvtepi16_epi32(prod16_lo)
                prod32_hi = _mm_cvtepi16_epi32(prod16_hi)

                # accumulate
                acc_lo = _mm_add_epi32(acc_lo, prod32_lo)
                acc_hi = _mm_add_epi32(acc_hi, prod32_hi)

        _mm_storeu_si128(<void*>tmp_lo, acc_lo)
        _mm_storeu_si128(<void*>tmp_hi, acc_hi)

        result_view[i,j] = (tmp_lo[0] + tmp_lo[1] + tmp_lo[2] + tmp_lo[3] +
                            tmp_hi[0] + tmp_hi[1] + tmp_hi[2] + tmp_hi[3])

    result = result_int.astype(np.float32) * (scale_a * scale_b)
    return result, numop


def matmul_simd_quantized_int16(np.ndarray[np.float32_t, ndim=2] A, 
                                np.ndarray[np.float32_t, ndim=2] B):
    """
    SIMD + Quantized Int16 Matrix Multiply (AVX).
    Combines quantization (reduced memory) with SIMD vectorization.
    """
    result_int = np.zeros((A.shape[0], B.shape[1]), dtype=np.int32)

    Bt = np.transpose(B)
    Bt = np.ascontiguousarray(Bt)

    scale_a = np.max(np.abs(A)) / 32767
    scale_b = np.max(np.abs(Bt)) / 32767

    if scale_a == 0:
        scale_a = 1.0
    if scale_b == 0:
        scale_b = 1.0

    A_quant = np.clip(np.round(A / scale_a), -32768, 32767).astype(np.int16)
    B_quant = np.clip(np.round(Bt / scale_b), -32768, 3276).astype(np.int16)

    cdef short[:, :] a_view = A_quant
    cdef short[:, :] b_view = B_quant
    cdef int[:, :] result_view = result_int

    cdef int i, j, k
    cdef int numop = 0

    cdef __m512i reg3 = _mm512_setzero_si512()
    cdef __m512i reg4 = _mm512_setzero_si512()
    cdef __m512i accumulate = _mm512_setzero_si512()
    cdef __m512i reg5 = _mm512_setzero_si512()
    
    cdef int tmp_reg[16]

    for i in range(A.shape[0]):
        for j in range(Bt.shape[0]):
            accumulate = _mm512_setzero_si512()
            for k in range(0, Bt.shape[1], 32):

                reg3 = _mm512_loadu_si512(&a_view[i, k])
                reg4 = _mm512_loadu_si512(&b_view[j, k])

                reg5 = _mm512_madd_epi16(reg3, reg4)
                accumulate = _mm512_add_epi32(accumulate, reg5)
            
            _mm512_storeu_si512(tmp_reg, accumulate)
            result_view[i, j] = (tmp_reg[0] + tmp_reg[1] + tmp_reg[2] + tmp_reg[3] +
                                 tmp_reg[4] + tmp_reg[5] + tmp_reg[6] + tmp_reg[7] +
                                 tmp_reg[8] + tmp_reg[9] + tmp_reg[10] + tmp_reg[11] +
                                 tmp_reg[12] + tmp_reg[13] + tmp_reg[14] + tmp_reg[15])

    result = result_int.astype(np.float32) * (scale_a * scale_b)
    return result, numop