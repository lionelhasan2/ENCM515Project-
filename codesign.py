import numpy as np
import time
from config import WORKLOAD, KERNEL_PROPS
from profiler import measure_matmul
from kernels import matmul_scalar_cpu, matmul_true_simd, matmul_quantized_int8


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
    
    def benchmark_cpu_accelerator_simd(self):
        print("SCENARIO 2A: CPU+TRUE SIMD (Float32)")
        
        results = []
        total_output_elements = 0
        for layer in self.LAYERS:
            r = self._measure_layer_matmul(layer['input'], layer['weight'], matmul_true_simd)
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
    
    def benchmark_cpu_accelerator_quantization(self):
        print("SCENARIO 2B: CPU+QUANTIZATION (Int8 Scalar)")
        
        results = []
        total_output_elements = 0
        for layer in self.LAYERS:
            r = self._measure_layer_matmul(layer['input'], layer['weight'], matmul_quantized_int8)
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
    
    print("\n" + "="*80)
    simd = simulator.benchmark_cpu_accelerator_simd()
    
    print("\n" + "="*80)
    quant = simulator.benchmark_cpu_accelerator_quantization()
    
    print("\n" + "="*80)
    print("SUMMARY: CPU vs Accelerator Strategies")
    print("="*80)
    
    speedup_simd = simd['throughput'] / cpu['throughput']
    speedup_quant = quant['throughput'] / cpu['throughput']
    mem_saved_simd = (1 - simd['memory'] / cpu['memory']) * 100
    mem_saved_quant = (1 - quant['memory'] / cpu['memory']) * 100
    
    print(f"\nStrategy 2A (True SIMD Float32):")
    print(f"  Throughput Speedup: {speedup_simd:.2f}x")
    print(f"  Memory Change: {mem_saved_simd:+.1f}%")
    print(f"  Total Time: {simd['total']:.4f} ms")
    
    print(f"\nStrategy 2B (Quantization - Int8 Scalar):")
    print(f"  Throughput Speedup: {speedup_quant:.2f}x")
    print(f"  Memory Reduction: {mem_saved_quant:.1f}%")
    print(f"  Total Time: {quant['total']:.4f} ms")
    
    print("="*80)


if __name__ == '__main__':
    main()
