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
    """Collect codesign comparison data"""
    simulator = MLPCodesignSimulator()
    cpu = simulator.benchmark_cpu_only()
    accel = simulator.benchmark_cpu_accelerator()
    
    return {
        'cpu_throughput': cpu['throughput'],
        'accel_throughput': accel['throughput'],
        'cpu_memory': cpu['memory'],
        'accel_memory': accel['memory'],
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


def plot_latency_per_layer():
    """Grouped bar chart: Latency breakdown per layer across kernels"""
    print("Collecting benchmark data...")
    data = collect_benchmark_data()
    
    kernel_names = list(data.keys())
    x = np.arange(len(LAYER_NAMES))
    width = 0.11  # Width of each bar
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(kernel_names)))
    
    for i, kernel_name in enumerate(kernel_names):
        latencies = data[kernel_name]['latencies']
        ax.bar(x + i * width, latencies, width, label=kernel_name, color=colors[i], edgecolor='black')
    
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Latency (ms)', fontsize=12)
    ax.set_title('Latency Breakdown by Layer', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (len(kernel_names) - 1) / 2)
    ax.set_xticklabels(LAYER_NAMES)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('latency_per_layer.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: latency_per_layer.png")
    plt.close()


def plot_codesign_comparison():
    """Side-by-side comparison: CPU-Only vs CPU+Accelerator (Throughput focus)"""
    print("Collecting codesign data...")
    data = collect_codesign_data()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Throughput comparison (in M Elements/sec)
    scenarios = ['CPU Only', 'CPU+Accelerator']
    throughputs = [data['cpu_throughput'], data['accel_throughput']]
    colors_tp = ['#FF6B6B', '#4ECDC4']
    bars1 = ax1.bar(scenarios, throughputs, color=colors_tp, edgecolor='black', width=0.6)
    ax1.set_ylabel('Throughput (M Elements/sec)', fontsize=12)
    ax1.set_title('Throughput Comparison', fontsize=13, fontweight='bold')
    
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Memory comparison
    memories = [data['cpu_memory'], data['accel_memory']]
    bars2 = ax2.bar(scenarios, memories, color=colors_tp, edgecolor='black', width=0.6)
    ax2.set_ylabel('Memory (KB)', fontsize=12)
    ax2.set_title('Memory Usage', fontsize=13, fontweight='bold')
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}KB', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add throughput speedup annotation
    tp_speedup = data['accel_throughput'] / data['cpu_throughput']
    mem_saved = (1 - data['accel_memory'] / data['cpu_memory']) * 100
    fig.text(0.5, 0.02, f'Throughput Speedup: {tp_speedup:.2f}x | Memory Reduction: {mem_saved:.1f}%',
             ha='center', fontsize=12, fontweight='bold', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
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
    plot_latency_per_layer()
    plot_codesign_comparison()
    plot_speedup_chart()
    
    print("\n" + "="*60)
    print(" All plots saved! Check for .png files in current directory")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
