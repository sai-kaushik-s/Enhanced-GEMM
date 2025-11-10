#!/usr/bin/env python3
"""
Performance benchmarking script for GEMM implementations.
Tests various matrix sizes and thread counts, generates performance plots.
"""

import subprocess
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import json
import os
from pathlib import Path

matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['savefig.dpi'] = 300

def run_benchmark(mode, N, T):
    """Run a single benchmark and return timing data."""
    try:
        if mode == "baseline":
            cmd = ["python3", "baseline/gemm_baseline.py", str(N), str(T)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                return None, None, f"Error: {result.stderr}"
            
            for line in result.stdout.strip().split('\n'):
                if 'time=' in line:
                    time_str = line.split('time=')[1].split()[0]
                    return float(time_str), None, None
        else:
            cmd = ["./optimized/gemm_opt", str(N), str(T)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                return None, None, f"Error: {result.stderr}"
            
            exec_time = None
            checksum = None
            for line in result.stdout.strip().split('\n'):
                if 'time=' in line:
                    exec_time = float(line.split('time=')[1].split()[0])
                elif 'checksum=' in line:
                    checksum = line.split('checksum=')[1].strip()
            
            return exec_time, checksum, None
            
    except subprocess.TimeoutExpired:
        return None, None, "Timeout"
    except Exception as e:
        return None, None, f"Exception: {str(e)}"

def build_binaries():
    """Build the C++ binaries."""
    print("Building C++ binaries...")
    result = subprocess.run(["make", "clean"], capture_output=True, text=True)
    result = subprocess.run(["make"], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Build failed: {result.stderr}")
        return False
    print("Build successful!")
    return True

def benchmark_matrix_sizes():
    """Benchmark across different matrix sizes with fixed thread count."""
    print("\n=== Benchmarking Matrix Sizes ===")
    
    sizes = [256, 512, 1024, 2048, 4096, 8192]
    threads = 8
    
    baseline_times = []
    optimized_times = []
    speedups = []
    
    for N in sizes:
        print(f"Testing N={N} with T={threads}...")
        
        base_time, _, base_err = run_benchmark("baseline", N, threads)
        if base_err:
            print(f"  Baseline error: {base_err}")
            baseline_times.append(None)
        else:
            baseline_times.append(base_time)
            print(f"  Baseline: {base_time:.4f}s")
        
        opt_time, _, opt_err = run_benchmark("optimized", N, threads)
        if opt_err:
            print(f"  Optimized error: {opt_err}")
            optimized_times.append(None)
        else:
            optimized_times.append(opt_time)
            print(f"  Optimized: {opt_time:.4f}s")
        
        if base_time and opt_time:
            speedup = base_time / opt_time
            speedups.append(speedup)
            print(f"  Speedup: {speedup:.2f}x")
        else:
            speedups.append(None)
    
    return sizes, baseline_times, optimized_times, speedups

def benchmark_thread_scaling():
    """Benchmark thread scaling with fixed matrix size."""
    print("\n=== Benchmarking Thread Scaling ===")
    
    N = 2048
    thread_counts = [1, 2, 4, 8, 16, 32, 48, 64, 72, 88, 96, 128, 144, 160, 176, 192, 208, 224, 240, 256]
    
    baseline_times = []
    optimized_times = []
    speedups = []
    
    for T in thread_counts:
        print(f"Testing N={N} with T={T}...")
        
        base_time, _, base_err = run_benchmark("baseline", N, T)
        if base_err:
            print(f"  Baseline error: {base_err}")
            baseline_times.append(None)
        else:
            baseline_times.append(base_time)
            print(f"  Baseline: {base_time:.4f}s")
        
        opt_time, _, opt_err = run_benchmark("optimized", N, T)
        if opt_err:
            print(f"  Optimized error: {opt_err}")
            optimized_times.append(None)
        else:
            optimized_times.append(opt_time)
            print(f"  Optimized: {opt_time:.4f}s")
        
        if base_time and opt_time:
            speedup = base_time / opt_time
            speedups.append(speedup)
            print(f"  Speedup: {speedup:.2f}x")
        else:
            speedups.append(None)
    
    return thread_counts, baseline_times, optimized_times, speedups

def pack_results(x_key, x_values, base, opt, speedups):
    return {
        x_key: x_values,
        'baseline_times': base,
        'optimized_times': opt,
        'speedups': speedups
    }

def create_plots():
    """Create and save performance plots."""
    os.makedirs("plots", exist_ok=True)
    
    sizes, base_times, opt_times, size_speedups = benchmark_matrix_sizes()
    
    plt.figure(figsize=(10, 6))
    valid_sizes = []
    valid_base = []
    valid_opt = []
    
    for i, (s, b, o) in enumerate(zip(sizes, base_times, opt_times)):
        if b is not None and o is not None:
            valid_sizes.append(s)
            valid_base.append(b)
            valid_opt.append(o)
    
    plt.loglog(valid_sizes, valid_base, 'o-', label='Python Baseline', linewidth=2, markersize=8)
    plt.loglog(valid_sizes, valid_opt, 's-', label='C++ Optimized', linewidth=2, markersize=8)
    plt.xlabel('Matrix Size (N)', fontsize=12)
    plt.ylabel('Execution Time (seconds)', fontsize=12)
    plt.title('GEMM Performance: Execution Time vs Matrix Size', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig('plots/time_vs_size.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    plt.figure(figsize=(10, 6))
    valid_sizes_speedup = []
    valid_speedups = []
    
    for i, (s, sp) in enumerate(zip(sizes, size_speedups)):
        if sp is not None:
            valid_sizes_speedup.append(s)
            valid_speedups.append(sp)
    
    plt.semilogx(valid_sizes_speedup, valid_speedups, 'o-', color='green', 
                 linewidth=2, markersize=8)
    plt.xlabel('Matrix Size (N)', fontsize=12)
    plt.ylabel('Speedup (x)', fontsize=12)
    plt.title('GEMM Speedup vs Matrix Size', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='No speedup')
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig('plots/speedup_vs_size.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    threads, base_times_t, opt_times_t, thread_speedups = benchmark_thread_scaling()
    
    plt.figure(figsize=(10, 6))
    valid_threads = []
    valid_base_t = []
    valid_opt_t = []
    
    for i, (t, b, o) in enumerate(zip(threads, base_times_t, opt_times_t)):
        if b is not None and o is not None:
            valid_threads.append(t)
            valid_base_t.append(b)
            valid_opt_t.append(o)
    
    plt.semilogx(valid_threads, valid_base_t, 'o-', label='Python Baseline', linewidth=2, markersize=8)
    plt.semilogx(valid_threads, valid_opt_t, 's-', label='C++ Optimized', linewidth=2, markersize=8)
    plt.xlabel('Number of Threads', fontsize=12)
    plt.ylabel('Execution Time (seconds)', fontsize=12)
    plt.title('GEMM Performance: Execution Time vs Thread Count (N=2048)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig('plots/time_vs_threads.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    plt.figure(figsize=(10, 6))
    valid_threads_speedup = []
    valid_speedups_t = []
    
    for i, (t, sp) in enumerate(zip(threads, thread_speedups)):
        if sp is not None:
            valid_threads_speedup.append(t)
            valid_speedups_t.append(sp)
    
    plt.semilogx(valid_threads_speedup, valid_speedups_t, 's-', color='purple', 
                 linewidth=2, markersize=8)
    plt.xlabel('Number of Threads', fontsize=12)
    plt.ylabel('Speedup (x)', fontsize=12)
    plt.title('GEMM Speedup vs Thread Count (N=2048)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='No speedup')
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig('plots/speedup_vs_threads.png', bbox_inches='tight', dpi=300)
    plt.close()

    results = {
        'matrix_size_scaling': pack_results('sizes', sizes, base_times, opt_times, size_speedups),
        'thread_scaling': pack_results('threads', threads, base_times_t, opt_times_t, thread_speedups)
    }
    
    with open('plots/benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n=== Results Summary ===")
    print(f"Plots saved to 'plots/' directory:")
    print(f"  - time_vs_size.png")
    print(f"  - speedup_vs_size.png") 
    print(f"  - time_vs_threads.png")
    print(f"  - speedup_vs_threads.png")
    print(f"  - benchmark_results.json")
    
    if valid_speedups:
        max_speedup_idx = np.argmax(valid_speedups)
        print(f"\nBest speedup: {valid_speedups[max_speedup_idx]:.2f}x at N={valid_sizes_speedup[max_speedup_idx]}")
    
    if valid_speedups_t:
        max_thread_speedup_idx = np.argmax(valid_speedups_t)
        print(f"Best thread speedup: {valid_speedups_t[max_thread_speedup_idx]:.2f}x at T={valid_threads_speedup[max_thread_speedup_idx]}")

def main():
    """Main benchmarking function."""
    print("GEMM Performance Benchmarking Script")
    print("====================================")
    
    if not Path("baseline/gemm_baseline.py").exists():
        print("Error: Run this script from the Enhanced-GEMM root directory")
        return 1
    
    if not build_binaries():
        return 1
    
    create_plots()
    
    return 0

if __name__ == "__main__":
    exit(main())