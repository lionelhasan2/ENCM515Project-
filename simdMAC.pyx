cimport numpy as np
import numpy as np
import random

"""
The matrix multiply formula: result[i,j] = Sum(A[i,k] * B[k,j])
"""

"""
External call to generate real SIMD instructions (ARM NEON).
"""
cdef extern from "arm_neon.h":
    ctypedef float float32x4_t # Declares the datatype for SIMD vectors of 4 elements (128-bit)

    float32x4_t vld1q_f32(float*) # Loads 4 floats at once

    float32x4_t vfmaq_f32(float32x4_t, float32x4_t, float32x4_t) # Fused multiply-add: a + b*c

    float32x4_t vmovq_n_f32(float) # Set all lanes to a scalar value

    void vst1q_f32(float*, float32x4_t) # Stores 4 floats to memory


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
    Implements true SIMD matrix multiplication using ARM NEON + Cython
    """
    result = np.zeros((A.shape[0], B.shape[1]), dtype=np.float32)
    Bt = np.transpose(B)
    Bt = np.ascontiguousarray(Bt)

    cdef float[:, :] reg3_view = A
    cdef float[:, :] reg4_view = Bt
    cdef float[:, :] result_view = result

    cdef float32x4_t reg3
    cdef float32x4_t reg4
    cdef float32x4_t accumulate

    cdef float tmp_reg[4]
    cdef int i, j, k
    cdef int numop = 0

    for i in range(A.shape[0]):
        for j in range(Bt.shape[0]):
            accumulate = vmovq_n_f32(0.0)
            for k in range(0, Bt.shape[1], 4):  # Process 4 at a time (NEON is 128-bit)
                reg3 = vld1q_f32(&reg3_view[i, k])
                reg4 = vld1q_f32(&reg4_view[j, k])
                accumulate = vfmaq_f32(accumulate, reg3, reg4)
                numop = numop + 1

            vst1q_f32(tmp_reg, accumulate)
            result_view[i, j] = (tmp_reg[0] + tmp_reg[1] + tmp_reg[2] + tmp_reg[3])

    #print("Number of Operations for SIMD MatMul: ", numop)
    #print(result)
    return result, numop


def matmul_true_simd_offset(np.ndarray[np.float32_t, ndim=2] A, 
                             np.ndarray[np.float32_t, ndim=2] B):
    """
    Implements SIMD matrix multiplication using ARM NEON.
    Includes an offset to observe performance with memory access misalignment.
    """
    result = np.zeros((A.shape[0], B.shape[1]), dtype=np.float32)
    Bt = np.transpose(B)
    Bt = np.ascontiguousarray(Bt)

    cdef float[:, :] reg3_view = A
    cdef float[:, :] reg4_view = Bt
    cdef float[:, :] result_view = result

    cdef float32x4_t reg3
    cdef float32x4_t reg4
    cdef float32x4_t accumulate

    cdef float tmp_reg[4]
    cdef int i, j, k
    cdef int numop = 0
    cdef int offset = random.randint(1, 3)  # Smaller offset for 128-bit vectors

    for i in range(A.shape[0]):
        for j in range(Bt.shape[0]):
            accumulate = vmovq_n_f32(0.0)
            for k in range(0, (Bt.shape[1] - offset - 3), 4):
                reg3 = vld1q_f32(&reg3_view[i, k + offset])
                reg4 = vld1q_f32(&reg4_view[j, k + offset])
                accumulate = vfmaq_f32(accumulate, reg3, reg4)
                numop = numop + 1

            vst1q_f32(tmp_reg, accumulate)
            result_view[i, j] = (tmp_reg[0] + tmp_reg[1] + tmp_reg[2] + tmp_reg[3])

    #print("Number of Operations for Offset SIMD MatMul: ", numop)
    #print(result)
    return result, numop


def matmul_true_simd_membank(np.ndarray[np.float32_t, ndim=2] A, 
                             np.ndarray[np.float32_t, ndim=2] B):
    """
    Implements SIMD matrix multiplication using ARM NEON.
    Simulates memory bank conflicts based on alignment.
    """
    result = np.zeros((A.shape[0], B.shape[1]), dtype=np.float32)
    Bt = np.transpose(B)
    Bt = np.ascontiguousarray(Bt)

    cdef float[:, :] reg3_view = A
    cdef float[:, :] reg4_view = Bt
    cdef float[:, :] result_view = result

    cdef float32x4_t reg3
    cdef float32x4_t reg4
    cdef float32x4_t accumulate

    cdef float tmp_reg[4]
    cdef int i, j, k, indx
    cdef int numop = 0
    cdef int offset = random.randint(1, 3)

    for i in range(A.shape[0]):
        for j in range(Bt.shape[0]):
            accumulate = vmovq_n_f32(0.0)
            for k in range(0, (Bt.shape[1] - offset), 4):
                if (k * 4) % 16 != 0:  # 16-byte bank boundary for NEON
                    indx = k - offset
                else:
                    indx = k
                
                reg3 = vld1q_f32(&reg3_view[i, indx])
                reg4 = vld1q_f32(&reg4_view[j, indx])
                accumulate = vfmaq_f32(accumulate, reg3, reg4)
                numop = numop + 1

            vst1q_f32(tmp_reg, accumulate)
            result_view[i, j] = (tmp_reg[0] + tmp_reg[1] + tmp_reg[2] + tmp_reg[3])

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