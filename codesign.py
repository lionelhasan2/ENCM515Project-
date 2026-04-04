import numpy as np
import time
from config import WORKLOAD, KERNEL_PROPS
from profiler import measure_matmul
from kernels import matmul_scalar_cpu, matmul_simd_quantized_int8


class MLPCodesignSimulator:
    
    # Convert WORKLOAD format (M, K, N) to layer dicts (input=K, weight=N)
    LAYERS = [{'input': K, 'weight': N} for M, K, N in WORKLOAD]
    
    RUNS = 3
    
    # Transfer latency per layer (ms) - overhead for moving data to/from accelerator
    TRANSFER_LATENCY_MS = 0.01
    
    def _measure_layer_matmul(self, in_dim, weight_dim, kernel_fn):
        A = np.random.randn(1, in_dim).astype(np.float32)
        B = np.random.randn(in_dim, weight_dim).astype(np.float32)
        kernel_name = KERNEL_PROPS[kernel_fn]['name']
        result = measure_matmul(kernel_fn, A, B, kernel_name, runs=self.RUNS)
        return result
    
    def benchmark_cpu_only(self):
        print("SCENARIO 1: CPU-ONLY (Scalar MatMul)")
        
        results = []
        total_output_elements = 0
        for layer in self.LAYERS:
            r = self._measure_layer_matmul(layer['input'], layer['weight'], matmul_scalar_cpu)
            results.append(r)
            total_output_elements += layer['weight']
        
        total_time_ms = sum(r.latency_ms for r in results)
        throughput_m_elems_sec = (total_output_elements / total_time_ms) / 1e3
        avg_gflops = sum(r.gflops for r in results) / len(results)
        total_mem = sum(r.memory_total_kb for r in results)
        
        print(f"Throughput: {throughput_m_elems_sec:.4f} M Elements/sec | Avg GFLOPS: {avg_gflops:.4f} | Memory: {total_mem:.1f}KB")
        return {'results': results, 'throughput': throughput_m_elems_sec, 'avg_gflops': avg_gflops, 'memory': total_mem, 'total': total_time_ms}
    
    def benchmark_cpu_accelerator(self):
        print("SCENARIO 2: CPU+ACCELERATOR (SIMD + Int8 Quantization)")
        
        results = []
        total_output_elements = 0
        for layer in self.LAYERS:
            r = self._measure_layer_matmul(layer['input'], layer['weight'], matmul_simd_quantized_int8)
            results.append(r)
            total_output_elements += layer['weight']
        
        total_compute_ms = sum(r.latency_ms for r in results)
        # Add transfer latency: one per layer (upload + download)
        transfer_overhead_ms = len(self.LAYERS) * self.TRANSFER_LATENCY_MS
        total_time_ms = total_compute_ms + transfer_overhead_ms
        
        throughput_m_elems_sec = (total_output_elements / total_time_ms) / 1e3
        avg_gflops = sum(r.gflops for r in results) / len(results)
        total_mem = sum(r.memory_total_kb for r in results)
        
        print(f"Throughput: {throughput_m_elems_sec:.4f} M Elements/sec | Avg GFLOPS: {avg_gflops:.4f} | Memory: {total_mem:.1f}KB | Transfer Overhead: {transfer_overhead_ms:.4f}ms")
        return {'results': results, 'throughput': throughput_m_elems_sec, 'avg_gflops': avg_gflops, 'memory': total_mem, 'total': total_time_ms}


def main():
    simulator = MLPCodesignSimulator()
    cpu = simulator.benchmark_cpu_only()
    accel = simulator.benchmark_cpu_accelerator()
    
    throughput_speedup = accel['throughput'] / cpu['throughput']
    mem_saved = (1 - accel['memory'] / cpu['memory']) * 100
    
    print(f"\nThroughput Speedup: {throughput_speedup:.2f}x")
    print(f"Memory Reduction: {mem_saved:.1f}%")


if __name__ == '__main__':
    main()
