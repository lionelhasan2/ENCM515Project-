# IoT ML Inference Benchmark

## What We're Testing

A **3-layer neural network inference** (MLP for image classification):
- Layer 1: 784 → 256 (input processing)
- Layer 2: 256 → 128 (hidden layer)
- Layer 3: 128 → 10 (output classes)

This represents a typical edge ML workload: small batch (1 sample), modest layer widths, running on resource-constrained devices.

## Target System

**ARM Cortex-M4 @ 168 MHz** (STM32F407 MCU)
- Common in wearables, IoT sensors, industrial controllers
- 192 KB SRAM — tight memory constraint
- No SIMD in baseline; team can add optimized kernels

## Running the Benchmark

```bash
python benchmark.py
```

Outputs to terminal:
- **Speed sweep** — performance across matrix sizes (32×32 to 512×512)
- **Inference workload** — total latency for the 3-layer chain
- Metrics: latency (ms), throughput (GFLOPS), speedup vs baseline

## Architecture

**Three modules:**
- `kernels.py` — matmul implementations (you add your optimized kernels here)
- `benchmark.py` — test driver (registers kernels, runs both workloads, prints results)
- `profiler.py` — measurement engine: takes a kernel function (lambda or regular function), profiles it

**How profiling works:**
1. Benchmark passes a kernel function reference to `measure_matmul(kernel_fn, A, B, ...)`
2. One warmup run (not timed, fills CPU cache)
3. Five timed runs using `time.perf_counter()` 
4. Report mean latency + compute GFLOPS, memory, arithmetic intensity
5. Compare speedup vs Naive baseline

## Adding a New Kernel

### Python Implementation

Add to `kernels.py`:

```python
def matmul_optimizationname(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Your optimized matmul implementation."""
    # Your code here
    return C
```

Then import and register in `benchmark.py`:

```python
from kernels import matmul_optimizationname

kernels = {
    "Naive": matmul_naive,
    "Your Kernel": matmul_optimizationname,  # <-- add here
}
```


## Metrics Explained

### Performance Metrics

- **Latency (ms)** — wall-clock execution time (lower is better)
- **GFLOPS** — compute throughput in billions of floating-point operations per second
  - A FLOP = one floating-point arithmetic operation (multiply, add, etc.)
  - For matmul: each element requires M×K multiplies + M×K adds = 2×M×K×N total FLOPs
  - Formula: `(2 × M × N × K) / (time_seconds × 1e9)`
  - Higher GFLOPS = kernel is doing more operations per second = better performance
- **Speedup** — relative performance vs Naive baseline (e.g., 2.5x = 2.5 times faster)

### IoT-Relevant Metrics

- **Mem (KB)** — memory footprint (A + B + C matrices)
  - **FitsM4**: ✓ if ≤192 KB (Cortex-M4 SRAM), ✗ if exceeds (uses slower external memory)

- **AI (Arithmetic Intensity)** — FLOPS per byte of memory accessed
  - Formula: `(2 × M × N × K) / (bytes_read + bytes_written)`
  - **Correlates with power efficiency**: Memory access costs ~10× more power than arithmetic on embedded CPUs
  - High AI (>1.0) = compute-bound = efficient; Low AI (<0.5) = memory-bound = wastes energy on data movement
