import numpy
from math import ceil

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

DMA_SETUP_COST = 5000 # We'll assume the DMA takes 5000 cycles (5μs with 1GHz) to set up.
# Assume that the DMA is more efficient on a per-cycle basis for larger amounts of data,
# the latency for the first bits of data is amortized in the setup cost.
DMA_COST_PER_BIT = 1 

registers = numpy.zeros(REG_COUNT, dtype=numpy.int32) # 8 register array, each register can handle integers/floating point values.
# numpy goes up to int64 while python has arbitrary length integers, it's a bit overkill but we use that here instead.
v_registers = [0 for i in range(VREG_COUNT)] # 256-bit Vector register array for SIMD operations

# For memory the highest workload modeled is 1x512x256 matrix multiplication, so an absolute
# minimum memory of ~0.5MB is required (mostly for storing the input 512x256 matrix). We'll work with
# 1MB as it will be sufficient for the benchmarks, though of course not larger matrixes.

# In the quantized implementation each word in data memory will only be 8 bits wide, so only a quarter
# of the memory has to be used (4 8-bit integers per 32-bit word).

data_memory = numpy.array(250000, dtype=numpy.float32) # 8 megabits / 32-bit words = 250000 unique indexes for data

pc = 0      # Program counter
cycles = 0  # we check the total cycles for performance comparison
flops = 0   # flops count
prog_enabled = 0 # Program enabled

# DMA address controls and matrix specifications
dma_input_1 = [0, 0]
dma_input_2 = [0, 0]
dma_output = 0 # Writes after address until finished
input_1_rows = 0
input_2_rows = 0

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
21: DMAS1 R1 R2 imm (Setup DMA to address DMEM[R1:R2] for first input matrix with imm rows)
22: DMAS2 R1 R2 imm (Setup DMA to address DMEM[R1:R2] for second input matrix with imm rows)
23: DMASO R1 (Setup DMA to address DMEM[R1:] for output matrix (variable size depending on matrix))
24: MMUL (Orders MMUL unit to run using DMA information)
"""

# Executes a given instruction
def exec(instr: Instruction):
    next_pc = pc+1
    match instr.op:
        case 0: # NOP
            pass
        case 1: # ADD R1 R2 R3
            registers[instr.reg3] = registers[instr.reg1] + registers[instr.reg2]
            flops += 1
            cycles += 1
        case 2: # ADDI R1 imm R3
            registers[instr.reg3] = registers[instr.reg1] + instr.imm
            flops += 1
            cycles += 1
        case 3: # MUL R1 R2 R3
            registers[instr.reg3] = registers[instr.reg1] * registers[instr.reg2]
            flops += 1
            cycles += 1
        case 4: # MULI R1 R2 R3
            registers[instr.reg3] = registers[instr.reg1] * instr.imm
            flops += 1
            cycles += 1
        # Quantization gets an advantage here as the memory cycle stalls become more significant,
        # but adds an overhead in decoding the 32-bit words into 4 8-bit integers.
        case 5: # LW R3 imm(R1)
            registers[instr.reg3] = data_memory[instr.reg1+instr.imm]
            cycles += 1 + MEM_CYCLE_COUNT
        case 6: # SW R2 imm(R1)
            data_memory[instr.reg1+instr.imm] = instr.reg3
            cycles += 1 + MEM_CYCLE_COUNT
        case 7: # JMP imm
            next_pc = instr.imm
            cycles += 1
        # Branching can take longer depending on control hazards, but we'll assume branch
        # predictions are always correct, so branching always takes 1 cycle.
        case 8: # BEQ R1 R2 imm
            if instr.reg1 == instr.reg2:
                next_pc = instr.imm
            cycles += 1 
        case 9: # HALT
            prog_enabled = 0
            cycles += 1
        case 10: # VADD VR1 VR2 VR3 (32-bit)
            # As long as an overflow doesn't occur on one of the 32-bit numbers adding the 256-bit vectors is
            # actually equivalent. Since overflow isn't being modelled in this implementation, this is sufficient.
            v_registers[instr.reg3] = v_registers[instr.reg1] + v_registers[instr.reg2]
            flops += 8 # 8 total addition operations for each 32-bit word
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
            flops += 8
            cycles += 1
        
        case 12: # VLW VR3, (R1) (32-bit)
            acc = 0 # Accumulator to write to VR3
            for i in range(8):
                acc <<= 32 # Taking advantage of 0 << 32 = 0 to shift all after the first integer
                acc += data_memory[instr.reg1+i]
            v_registers[instr.reg3] = acc
            cycles += 1 + MEM_CYCLE_COUNT
        
        case 13: # VSW VR2, (R1) (32-bit)
            # Write to the array from LSB to MSB
            t = v_registers[instr.reg2]
            for i in range(8-1, -1, -1):
                data_memory[i] = t & 0xFFFFFFFF # Write 32 LSB to DMEM
                t >>= 32 # shift temp to next word
            cycles += 1 + MEM_CYCLE_COUNT
        
        case 20: # MAC R1 R2 R3
            registers[instr.reg3] = registers[instr.reg3] + registers[instr.reg1]*registers[instr.reg2]
            flops += 2 # one for add, one for multiply
            cycles += 1
        
        case 21: # DMAS1 R1 R2 imm (DMA input address range #1)
            dma_input_1 = [instr.reg1, instr.reg2]
            input_1_rows = instr.imm
            cycles += 1
        case 22: # DMAS2 R1 R2 imm (DMA input address range #2)
            dma_input_2 = [instr.reg1, instr.reg2]
            input_2_rows = instr.imm
            cycles += 1
        case 23: # DMASO R1 (DMA output address range #3)
            dma_output = instr.reg1
            cycles += 1
        case 24: # MMUL (simulated MMUL unit using DMA information)
            # We'll assume the MMUL unit has 256 parallel MAC units, so it only takes 1 cycle per every
            # 256 MAC operations. The total MAC operations required for multiplying matrixes with shape
            # m x k x n is actually m x k x n (which is why it's 2 x m x k x n FLOPS for add and multiply)
            # easier to calculate this after the matrixes are retrieved from memory

            # Get left and right matrixes from data memory using DMA variables
            left_matrix = read_DMEM_to_matrix(dma_input_1[0], dma_input_1[1], input_1_rows)
            right_matrix = read_DMEM_to_matrix(dma_input_2[0], dma_input_2[1], input_2_rows)

            # Calculate total required MAC operations
            t = len(left_matrix) * len(left_matrix[0]) * len(right_matrix[0]) # m x k x n
            cycles += ceil(t/256) + 1 # divided by 256 assuming all MAC units were consuming data roughly equally
            flops += t * 2 # 2 x m x k x n flops

            # Run matrix multiplication on retrieved input matrixes
            product_matrix = multiply_matrixes(left_matrix, right_matrix)

            # Write product matrix to memory at DMEM[dma_output:]
            data_pointer = dma_output
            for i in range(len(product_matrix)):
                for j in range(len(product_matrix[0])):
                    data_memory[data_pointer] = product_matrix[i][j]
                    data_pointer += 1
    
    pc = next_pc
            
# MMUL simulated unit helper function, generates matrix from DMEM
def read_DMEM_to_matrix(adr1, adr2, rows):
    if (adr2-adr1+1) % rows != 0: raise ValueError("Number of columns does not correspond with rows and total elements")
    cols = (adr2-adr1+1) // rows
    m = numpy.array([[data_memory[i+j*cols] for i in range(cols)] for j in range(rows)], dtype=numpy.float32)
    return m

# MMUL helper function, multiplies matrixes m1 and m2
def multiply_matrixes(m1, m2):
    rows = len(m1)
    cols = len(m2[0])
    return_matrix = numpy.zeros((cols, rows), dtype=numpy.float32)
    for row in range(rows):
        for col in range(cols):
            acc = 0
            for i in range(len(m2)):
                acc += m1[row][i] * m2[i][col]
            return_matrix[row][col] = acc
    return return_matrix

def run_program(instruction_memory):
    global registers, v_registers, data_memory, pc, cycles, flops, prog_enabled, dma_input_1, dma_input_2, dma_output, input_1_rows, input_2_rows

    # Reset program variables / metrics
    registers = numpy.zeros(REG_COUNT, dtype=numpy.int32)
    v_registers = [0 for i in range(VREG_COUNT)]
    data_memory = numpy.array(250000, dtype=numpy.float32)
    pc = 0
    cycles = 0
    flops = 0
    prog_enabled = 1

    dma_input_1 = [0, 0]
    dma_input_2 = [0, 0]
    dma_output = 0
    input_1_rows = 0
    input_2_rows = 0

    # Run program until halted by CPU
    while prog_enabled:
        exec(instruction_memory[pc])