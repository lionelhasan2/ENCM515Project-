import numpy
from math import ceil
# from config import WORKLOAD
from profiler import CORTEX_M4_SRAM_KB, ProfileResult, print_results_table

# The goal here is to simulate a simple RISC CPU (no complex features like pipelining or branching)
# to use for matrix multiplication and compare the performance of different accelerators and
# CPU optimizations such as quantization and SIMD.

# For quantization, the program will receive the quantized matrixes, as opposed to quantizing
# them in the program. If the program has to quantize the matrixes then just as much if not
# more memory is used storing the original input matrixes and creating new quantized matrixes.
# The actual required size of each word depends on the implementation. For floating point, 32-bit
# wide words are required for calculation, but for 8-bit quantization each memory word only
# needs to be 8 bits wide, which should lead to a little under 75% less memory usage as long
# as the majority of the data memory usage is from storing the matrixes.

# While real instructions are sets of bits, I'll be working with instructions
# as objects for simplicity of implementation and decoding here.

# In reality floating point and integer registers are usually seperated, but here we
# will treat each register as if it can accept either floating numbers or integers
# to reduce the number of instructions that would need to be implemented which
# do not reflect anything significant in the performance comparisons this is designed for.

# If the instruction were actual bits it would be partitioned as so:
# 5 bits |    3 bits each     | 32 bits
# op     | reg1 | reg2 | reg3 | imm
# the imm portion could be made significantly shorter by loading it in two parts for some instructions
# but this longer implementation is simpler, easier to write, and easier to understand.

class Instruction:
    # op is the instruction to execute (ie: ADD)
    # reg1/2 tell the cpu which registers to read from
    # reg3 is the typically the register to write results to
    # imm is immediate value
    def __init__(self, op, reg1, reg2, reg3, imm):
        self.op = op
        self.reg1 = reg1
        self.reg2 = reg2
        self.reg3 = reg3
        self.imm = imm

# Simulation constants
CLK_SPD = 1e9 # Assumed clock speed only for estimating program latency
DMA_BANDWIDTH = 10e6
DMA_SETUP_LATENCY = 5e-6
REG_COUNT = 8
VREG_COUNT = 4
MEM_CYCLE_COUNT = 1 # Number of cycles to stall when accessing memory
DMEM_SIZE = 250000 # 1MB / 32-bit words = 250000 addresses

DMA_SETUP_COST = 5000 # We'll assume the DMA takes 5000 cycles (5μs with 1GHz) to set up.
# Assume that the DMA is more efficient on a per-cycle basis for larger amounts of data,
# the latency for the first bits of data is amortized in the setup cost.
DMA_COST_PER_BIT = 1 

addresses = {} # symbol : address map
registers = numpy.zeros(REG_COUNT, dtype=numpy.int32) # 8 register array, each register can handle integers/floating point values.
# numpy goes up to int64 while python has arbitrary length integers, it's a bit overkill but we use that here instead.
v_registers = [0 for i in range(VREG_COUNT)] # 256-bit Vector register array for SIMD operations

# For memory the highest workload modeled is 1x512x256 matrix multiplication, so an absolute
# minimum memory of ~0.5MB is required (mostly for storing the input 512x256 matrix). We'll work with
# 1MB as it will be sufficient for the benchmarks, though of course not larger matrixes.

# In the quantized implementation each word in data memory will only be 8 bits wide, so only a quarter
# of the memory has to be used (4 8-bit integers per 32-bit word).

data_memory = numpy.zeros(DMEM_SIZE, dtype=numpy.float32) # 2MB / 32-bit words = 500000 unique indexes for data

pc = 0      # Program counter
cycles = 0  # we check the total cycles for performance comparison
flop_total = 0   # flops count
prog_enabled = 0 # Program enabled
bytes_accessed = 0

# DMA address controls and matrix specifications
dma_input_1 = 0
dma_input_2 = 0
dma_output = 0 # Writes after address until finished

# Executes a given instruction
def exec(instr: Instruction):
    global data_memory, pc, cycles, flop_total, prog_enabled, bytes_accessed
    global dma_input_1, dma_input_2, dma_output, addresses, registers, v_registers
    next_pc = pc+1
    match instr.op:
        case 0: # NOP
            pass
        case 1: # ADD R1 R2 R3
            registers[instr.reg3] = registers[instr.reg1] + registers[instr.reg2]
            flop_total += 1
            cycles += 1
        case 2: # ADDI R1 imm R3
            registers[instr.reg3] = registers[instr.reg1] + instr.imm
            flop_total += 1
            cycles += 1
        case 3: # MUL R1 R2 R3
            registers[instr.reg3] = registers[instr.reg1] * registers[instr.reg2]
            flop_total += 1
            cycles += 1
        case 4: # MULI R1 R2 R3
            registers[instr.reg3] = registers[instr.reg1] * instr.imm
            flop_total += 1
            cycles += 1
        # Quantization gets an advantage here as the memory cycle stalls become more significant,
        # but adds an overhead in decoding the 32-bit words into 4 8-bit integers.
        case 5: # LW R3 imm(R1)
            registers[instr.reg3] = data_memory[instr.reg1+instr.imm]
            bytes_accessed += 1
            cycles += 1 + MEM_CYCLE_COUNT
        case 6: # SW R2 imm(R1)
            data_memory[instr.reg1+instr.imm] = registers[instr.reg3]
            bytes_accessed += 1
            cycles += 1 + MEM_CYCLE_COUNT
        case 7: # JMP imm
            next_pc = instr.imm
            cycles += 1
        # Branching can take longer depending on control hazards, but we'll assume branch
        # predictions are always correct, so branching always takes 1 cycle.
        case 8: # BEQ R1 R2 imm
            if registers[instr.reg1] == registers[instr.reg2]:
                next_pc = instr.imm
            cycles += 1 
        case 9: # HALT
            prog_enabled = 0
            cycles += 1
        case 10: # VADD VR1 VR2 VR3 (32-bit)
            # As long as an overflow doesn't occur on one of the 32-bit numbers adding the 256-bit vectors is
            # actually equivalent. Since overflow isn't being modelled in this implementation, this is sufficient.
            v_registers[instr.reg3] = v_registers[instr.reg1] + v_registers[instr.reg2]
            flop_total += 8 # 8 total addition operations for each 32-bit word
            cycles += 1
        
        case 11: # VMUL VR1 VR2 VR3 (32-bit)
            # Here each 32-bit number needs to be multiplied individually and added back together in VR3
            v1 = v_registers[instr.reg1]
            v2 = v_registers[instr.reg2]
            v3 = 0
            bitMask = 0xFFFFFFFF # 32-bit wide mask
            # Add each 32-bit integer to v3
            for i in range(0, 128, 32):
                # Read lowest 32 bits of each vector register
                t1 = (v1 & (0xFFFFFFFF << i)) >> i # Get 32-bit integer
                t2 = (v2 & (0xFFFFFFFF << i)) >> i 
                t3 = t1*t2
                v3 = v3 + (t3 << i) # Add back to correct bit shift
            v_registers[instr.reg3] = v3 # Write back result
            flop_total += 8
            cycles += 1
        
        case 12: # VLW VR3, (R1) (32-bit)
            acc = 0 # Accumulator to write to VR3
            for i in range(8):
                acc <<= 32 # Taking advantage of 0 << 32 = 0 to shift all after the first integer
                acc += data_memory[instr.reg1+i]
            v_registers[instr.reg3] = acc
            bytes_accessed += 8
            cycles += 1 + MEM_CYCLE_COUNT
        
        case 13: # VSW VR2, (R1) (32-bit)
            # Write to the array from LSB to MSB
            t = v_registers[instr.reg2]
            for i in range(8-1, -1, -1):
                data_memory[i] = t & 0xFFFFFFFF # Write 32 LSB to DMEM
                t >>= 32 # shift temp to next word
            bytes_accessed += 8
            cycles += 1 + MEM_CYCLE_COUNT
        
        case 14: # LA R3 imm
            registers[instr.reg3] = addresses[instr.imm]
            cycles += 1
        
        case 20: # MAC R1 R2 R3
            registers[instr.reg3] = registers[instr.reg3] + registers[instr.reg1]*registers[instr.reg2]
            flop_total += 2 # one for add, one for multiply
            cycles += 1
        
        case 21: # DMAS1 R1 R2 (DMA input address range #1)
            dma_input_1 = registers[instr.reg1]
            cycles += 1
        case 22: # DMAS2 R1 R2 (DMA input address range #2)
            dma_input_2 = registers[instr.reg1]
            cycles += 1
        case 23: # DMASO R1 (DMA output address range #3)
            dma_output = registers[instr.reg1]
            cycles += 1
        case 24: # MMUL (simulated MMUL unit using DMA information)
            # We'll assume the MMUL unit has 256 parallel MAC units, so it only takes 1 cycle per every
            # 256 MAC operations. The total MAC operations required for multiplying matrixes with shape
            # m x k x n is actually m x k x n (which is why it's 2 x m x k x n FLOPS for add and multiply)
            # easier to calculate this after the matrixes are retrieved from memory

            # Get left and right matrixes from data memory using DMA variables
            left_matrix = read_DMEM_to_matrix(dma_input_1)
            right_matrix = read_DMEM_to_matrix(dma_input_2)

            # Calculate total required MAC operations
            t = len(left_matrix) * len(left_matrix[0]) * len(right_matrix[0]) # m x k x n
            cycles += ceil(t/256) + 1 # divided by 256 assuming all MAC units were consuming data roughly equally
            cycles += DMA_SETUP_COST # account for DMA setup cost here
            flop_total += t * 2 # 2 x m x k x n flops

            # Run matrix multiplication on retrieved input matrixes
            product_matrix = multiply_matrixes(left_matrix, right_matrix)

            # Write product matrix to memory at DMEM[dma_output:]
            write_matrix_to_DMEM(product_matrix, dma_output)
    
    pc = next_pc
            
# MMUL simulated unit helper function, generates matrix from DMEM
def read_DMEM_to_matrix(adr1):
    global data_memory
    rows = data_memory[adr1+0].astype(numpy.int32)
    cols = data_memory[adr1+1].astype(numpy.int32)
    m = numpy.array([[data_memory[adr1+2+i+j*cols] for i in range(cols)] for j in range(rows)], dtype=numpy.float32)
    return m

# MMUL helper function, multiplies matrixes m1 and m2
def multiply_matrixes(m1, m2):
    rows = len(m1)
    cols = len(m2[0])
    return_matrix = numpy.zeros((rows, cols), dtype=numpy.float32)
    for row in range(rows):
        for col in range(cols):
            acc = 0
            for i in range(len(m2)):
                acc += m1[row][i] * m2[i][col]
            return_matrix[row][col] = acc
    return return_matrix

def run_program(instruction_memory):
    global registers, v_registers, data_memory, pc, cycles, flop_total, prog_enabled, dma_input_1, dma_input_2, dma_output, bytes_accessed

    # Reset program variables / metrics
    registers = numpy.zeros(REG_COUNT, dtype=numpy.int32)
    v_registers = [0 for i in range(VREG_COUNT)]
    # Don't reset data memory as it needs to be written to for input before the program runs!
    # data_memory = numpy.zeros(DMEM_SIZE, dtype=numpy.float32)
    pc = 0
    cycles = 0
    flop_total = 0
    bytes_accessed = 0
    prog_enabled = 1

    dma_input_1 = 0
    dma_input_2 = 0
    dma_output = 0

    # Run program until halted by CPU
    while prog_enabled:
        exec(instruction_memory[pc])
    
def write_matrix_to_DMEM(matrix, address):
    global data_memory
    dp = 0
    data_memory[dp+address] = len(matrix) # row count
    data_memory[dp+1+address] = len(matrix[0]) # col count
    dp += 2
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            data_memory[dp+address] = matrix[i][j]
            dp += 1

def reset_DMEM():
    global data_memory, DMEM_SIZE
    data_memory = numpy.zeros(DMEM_SIZE, dtype=numpy.float32)

# run benchmark with specific program and specified element width
def benchmark(program_name, program, A: numpy.ndarray, B: numpy.ndarray, elem_width=4):
    global addresses, data_memory
    rng = numpy.random.default_rng(seed=42)
    dp = 0
    r = []
    M = len(A)
    K = len(A[0])
    N = len(B[0])
    reset_DMEM()
    write_matrix_to_DMEM(A, 0)
    addresses['A'] = 0
    matrix_2_pos = 2+M*K # address of second matrix information
    write_matrix_to_DMEM(B, matrix_2_pos)
    addresses['B'] = matrix_2_pos
    matrix_3_pos = matrix_2_pos + 2 + K*N
    addresses['C'] = matrix_3_pos
    run_program(program)

    approximate_latency = cycles / CLK_SPD #  Approximate latency from cycle count and clock speed assumptions
    # Memory used (KB) = (total elements in array + 2) * x bytes/elem / 1024 bytes/KB
    mem_A = ((M*K)+2) * elem_width / 1024
    mem_B = ((K*N)+2) * elem_width / 1024
    mem_C = ((M*N)+2) * elem_width / 1024

    arith_intensity = flop_total / bytes_accessed if bytes_accessed > 0 else 0.0

    ref = numpy.matmul(A.astype(numpy.float64), B.astype(numpy.float64))
    actual_output = read_DMEM_to_matrix(matrix_3_pos)

    result = ProfileResult(
        kernel_name=program_name,
        matrix_shape=(M, K, N),
        dtype=str(A.dtype),
        runs=1, # The hardware sim is deterministic, no need for multiple runs
        latency_ms=approximate_latency * 1000,
        latency_std_ms=approximate_latency * 1000,
        latency_min_ms=approximate_latency * 1000,
        latency_max_ms=approximate_latency * 1000,
        flops= flop_total / approximate_latency,
        gflops= (flop_total / approximate_latency) / 1e9,
        memory_A_kb= mem_A,
        memory_B_kb= mem_B,
        memory_C_kb= mem_C,
        memory_total_kb=mem_A+mem_B+mem_C,
        fits_cortex_m4= (mem_A+mem_B+mem_C) <= CORTEX_M4_SRAM_KB,
        arithmetic_intensity=arith_intensity,
        num_operations=cycles,
        error = numpy.linalg.norm(actual_output - ref) / numpy.linalg.norm(ref),
    )
    return result

def matmul(A: numpy.ndarray, B: numpy.ndarray):
    return benchmark("MMUL Accelerator", mmul_accelerator_program, A, B)

"""
Implemented instructions:
0: NOP
1: ADD R1, R2, R3 (R1+R2 -> R3)
2: ADDI R1, imm, R3 (R1+imm -> R3)
3: MUL R1, R2, R3 (R1*R2 -> R3) (For this simulation we assume the CPU )
4: MULI R1, imm, R3 (R1*imm -> R3)
5: LW R3, imm(R1) (load word from address with imm offset)
6: SW R2, imm(R1) (store word in address with imm offset)
7: JMP imm (jump to IMEM[imm])
8: BEQ R1, R2, imm (branch to IMEM[imm] if equal)
9: HALT (finish the program)
14: LA R3, imm (load imm key address to register)

SIMD instructions, using vector registers:
10: VADD VR1, VR2, VR3 (VR1+VR2 -> VR3)
11: VMUL VR1, VR2, VR3 (VR1*VR2 -> VR3 for each element (not a dot product))
12: VLW VR3, (R1) (load 8 32-bit words into VR3 starting from R1)
13: VSW VR2, (R1) (write 8 32-bit words into memory at DMEM[R1:R1+7] from VR2)

Originally I planned to add SIMD instructions for 8-bits but realized that the very expected
overflows would be unmanageable as the result could only be represented in a larger vector than available.

These last instructions simulate the use of a dedicated accelerator.
20: MAC R1, R2, R3 (R3 = R3 + R1*R2)

In reality control of the DMA is MUCH more complicated, but this is most analagous to a burst mode DMA
that takes control of the system bus until all data has been sent before transferring control back to the CPU.
Here, we can approximate the number of cycles it will take as the setup cost ()
21: DMAS1 R1 (Setup DMA to address DMEM[R1:] for first input matrix)
22: DMAS2 R1 (Setup DMA to address DMEM[R1:] for second input matrix)
23: DMASO R1 (Setup DMA to address DMEM[R1:] for output matrix (variable size depending on matrix))
24: MMUL (Orders MMUL unit to run using DMA information)
"""

mmul_accelerator_program = [
    # Load matrix addresses to registers, write to DMA settings, and run MMUL
    Instruction(14, 0, 0, 1, 'A'), # la x1 'A'
    Instruction(14, 0, 0, 2, 'B'), # la x2 'B'
    Instruction(14, 0, 0, 3, 'C'), # la x3 'C'
    Instruction(21, 1, 0, 0, 0), # dmas x1
    Instruction(22, 2, 0, 0, 0), # dmas x2
    Instruction(23, 3, 0, 0, 0), # dmas x3
    Instruction(24, 0, 0, 0, 0), # MMUL
    Instruction(9, 0, 0, 0, 0) # halt
]

# if __name__ == "main":
#     rng = numpy.random.default_rng(seed=42)
#     for (M, K, N) in WORKLOAD:
#         A = rng.standard_normal((M, K)).astype(numpy.float32)
#         B = rng.standard_normal((K, N)).astype(numpy.float32)
#         profile_result = matmul(A, B)

#     # Calculate speedup based on existing benchmarks
#     baseline_latencies = (34.3796e-3, 9.4571e-3, 2.1826e-3)

#     for i in range(3):
#         profile_result[i].speedup = baseline_latencies[i] / profile_result[i].latency_ms * 1000

#     print_results_table(profile_result)