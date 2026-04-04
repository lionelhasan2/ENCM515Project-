import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from config import WORKLOAD, WORKLOAD_NAME, KERNEL_PROPS
from profiler import measure_matmul
from codesign import MLPCodesignSimulator

# Generate layer names from WORKLOAD
LAYER_NAMES = [f'Layer {i+1} ({K}→{N})' for i, (M, K, N) in enumerate(WORKLOAD)]


def collect_benchmark_data():
    """Collect throughput and GFLOPS data for all kernels"""
    results = {}
    
    for fn, props in KERNEL_PROPS.items():
        kernel_name = props['name']
        latencies = []
        gflops = []
        throughputs = []
        
        for M, K, N in WORKLOAD:
            rng = np.random.default_rng(seed=42)
            A = rng.standard_normal((M, K)).astype(np.float32)
            B = rng.standard_normal((K, N)).astype(np.float32)
            r = measure_matmul(fn, A, B, kernel_name, runs=5)
            latencies.append(r.latency_ms)
            gflops.append(r.gflops)
            # Throughput as output elements per millisecond, converted to Melements/sec
            output_elements = M * N
            tp_melems_sec = (output_elements / r.latency_ms) / 1e3
            throughputs.append(tp_melems_sec)
        
        results[kernel_name] = {'latencies': latencies, 'gflops': gflops, 'throughputs': throughputs}
    
    return results


def collect_codesign_data():
    """Collect codesign comparison data for three scenarios"""
    simulator = MLPCodesignSimulator()
    cpu = simulator.benchmark_cpu_only()
    simd = simulator.benchmark_cpu_accelerator_simd()
    quant = simulator.benchmark_cpu_accelerator_quantization()
    
    return {
        'cpu': {'throughput': cpu['throughput'], 'memory': cpu['memory'], 'total': cpu['total']},
        'simd': {'throughput': simd['throughput'], 'memory': simd['memory'], 'total': simd['total']},
        'quant': {'throughput': quant['throughput'], 'memory': quant['memory'], 'total': quant['total']},
    }


def plot_throughput_by_kernel():
    """Bar chart: Total throughput per kernel"""
    print("Collecting benchmark data...")
    data = collect_benchmark_data()
    
    kernel_names = list(data.keys())
    # Calculate total throughput as total output elements / total time
    total_throughputs = []
    for k in kernel_names:
        total_output_elements = sum(N for M, K, N in WORKLOAD)  # M=1, so just N
        total_time_ms = sum(data[k]['latencies'])
        tp = (total_output_elements / total_time_ms) / 1e3  # M Elements/sec
        total_throughputs.append(tp)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(kernel_names, total_throughputs, color='steelblue', edgecolor='black')
    ax.set_ylabel('Throughput (M Elements/sec)', fontsize=12)
    ax.set_title('Kernel Throughput Performance (Output Elements)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(total_throughputs) * 1.15)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('throughput_by_kernel.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: throughput_by_kernel.png")
    plt.close()


def plot_gflops_by_kernel():
    """Bar chart: Average GFLOPS per kernel"""
    data = collect_benchmark_data()
    
    kernel_names = list(data.keys())
    avg_gflops = [np.mean(data[k]['gflops']) for k in kernel_names]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(kernel_names, avg_gflops, color='coral', edgecolor='black')
    ax.set_ylabel('Average GFLOPS', fontsize=12)
    ax.set_title('Compute Performance Across Kernels', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(avg_gflops) * 1.1)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('gflops_by_kernel.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: gflops_by_kernel.png")
    plt.close()


def plot_accuracy_chart():
    """Bar chart: Accuracy (relative error) across kernel implementations"""
    print("Collecting benchmark data...")
    data = collect_benchmark_data()
    
    kernel_names = list(data.keys())
    # Collect error metrics for each kernel
    errors = []
    
    for fn, props in KERNEL_PROPS.items():
        kernel_name = props['name']
        kernel_errors = []
        
        for M, K, N in WORKLOAD:
            rng = np.random.default_rng(seed=42)
            A = rng.standard_normal((M, K)).astype(np.float32)
            B = rng.standard_normal((K, N)).astype(np.float32)
            r = measure_matmul(fn, A, B, kernel_name, runs=5)
            kernel_errors.append(r.error)
        
        avg_error = np.mean(kernel_errors)
        errors.append(avg_error)
    
    # Color code: green for low error, red for high error
    colors = ['green' if e < 0.01 else 'orange' if e < 0.1 else 'red' for e in errors]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(kernel_names, errors, color=colors, edgecolor='black')
    ax.set_ylabel('Relative Error (% of reference)', fontsize=12)
    ax.set_title('Accuracy Comparison Across Kernels', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(errors) * 1.15 if max(errors) > 0 else 0.1)
    
    # Add value labels on bars
    for bar, error in zip(bars, errors):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{error:.6f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('accuracy_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: accuracy_comparison.png")
    plt.close()


def plot_codesign_comparison():
    """Three-way comparison: CPU-Only vs True SIMD vs Quantization (Throughput & Memory focus)"""
    print("Collecting codesign data...")
    data = collect_codesign_data()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Throughput comparison (in M Elements/sec)
    scenarios = ['CPU Only', 'CPU + SIMD', 'CPU + Quantization']
    throughputs = [data['cpu']['throughput'], data['simd']['throughput'], data['quant']['throughput']]
    colors_tp = ['#FF6B6B', '#4ECDC4', '#95E1D3']
    bars1 = ax1.bar(scenarios, throughputs, color=colors_tp, edgecolor='black', width=0.6)
    ax1.set_ylabel('Throughput (M Elements/sec)', fontsize=12)
    ax1.set_title('Throughput Comparison', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, max(throughputs) * 1.2)
    
    for bar, tp in zip(bars1, throughputs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{tp:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Memory comparison
    memories = [data['cpu']['memory'], data['simd']['memory'], data['quant']['memory']]
    bars2 = ax2.bar(scenarios, memories, color=colors_tp, edgecolor='black', width=0.6)
    ax2.set_ylabel('Memory (KB)', fontsize=12)
    ax2.set_title('Memory Usage', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, max(memories) * 1.2)
    
    for bar, mem in zip(bars2, memories):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{mem:.1f}KB', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add speedup annotations
    cpu_baseline = data['cpu']['throughput']
    simd_speedup = data['simd']['throughput'] / cpu_baseline
    quant_speedup = data['quant']['throughput'] / cpu_baseline
    mem_saved_quant = (1 - data['quant']['memory'] / data['cpu']['memory']) * 100
    
    summary_text = f'SIMD Speedup: {simd_speedup:.2f}x | Quant Speedup: {quant_speedup:.2f}x | Quant Memory Reduction: {mem_saved_quant:.1f}%'
    fig.text(0.5, 0.02, summary_text,
             ha='center', fontsize=11, fontweight='bold', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout(rect=(0, 0.07, 1, 1))
    plt.savefig('codesign_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: codesign_comparison.png")
    plt.close()


def plot_speedup_chart():
    """Bar chart: Speedup relative to Scalar baseline"""
    data = collect_benchmark_data()
    
    scalar_latency = sum(data['Scalar']['latencies'])
    kernel_names = [k for k in data.keys() if k != 'Scalar']
    speedups = [scalar_latency / sum(data[k]['latencies']) for k in kernel_names]
    
    # Highlight int8 kernels
    colors = ['coral' if 'Int8' in k or 'int8' in k.lower() else 'lightblue' for k in kernel_names]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(kernel_names, speedups, color=colors, edgecolor='black')
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Baseline (Scalar)')
    ax.set_ylabel('Speedup (relative to Scalar)', fontsize=12)
    ax.set_title('Kernel Speedup Relative to Scalar MatMul', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(speedups) * 1.15)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}x', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.legend()
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('speedup_chart.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: speedup_chart.png")
    plt.close()


def main():
    print("\n" + "="*60)
    print(" Generating Visualization Plots")
    print("="*60 + "\n")
    
    plot_throughput_by_kernel()
    plot_gflops_by_kernel()
    plot_accuracy_chart()
    plot_codesign_comparison()
    plot_speedup_chart()
    
    print("\n" + "="*60)
    print(" All plots saved! Check for .png files in current directory")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
