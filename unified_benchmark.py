#!/usr/bin/env python3
"""
Unified Matched Filter Benchmark
Tests NumPy CPU-only, MLX CPU, and MLX GPU performance 
Saves all results to a single HDF5 file for easy comparison.
"""

import numpy as np
import time
import h5py
import argparse
import os
import sys
import multiprocessing
from datetime import datetime

# Force NumPy to use single thread for fair comparison
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

def check_mlx_available():
    """Check if MLX is available on this system."""
    try:
        import mlx.core as mx
        from mlx.core.fft import ifft
        return True, mx, ifft
    except ImportError:
        return False, None, None

def define_match_filter_numpy(a, b):
    """Matched filter using NumPy (CPU-only)."""
    product = np.multiply(a.conj(), b)
    ifft_result = np.fft.ifft(product)
    reshaped_result = ifft_result.reshape(ifft_result.size)
    return np.max(reshaped_result)

def run_numpy_match_filter(args):
    """Wrapper for multiprocessing."""
    a, b = args
    return define_match_filter_numpy(a, b)

def define_match_filter_mlx(a, b, device, mx, ifft):
    """Matched filter using MLX (CPU or GPU)."""
    mx.set_default_device(device)
    mx_a = mx.array(a)
    mx_b = mx.array(b) 
    product = mx.multiply(mx_a.conj(), mx_b)
    ifft_result = ifft(product)
    mx.eval(ifft_result)
    result_np = np.array(ifft_result)
    return np.max(result_np)

def benchmark_numpy_multicore(data_pairs, cores):
    """Benchmark NumPy with specified number of cores."""
    print(f"Running NumPy CPU benchmark with {cores} cores...")
    
    # Create list of (a,b) tuples for multiprocessing
    ab_list = [(a, b) for a, b in data_pairs]
    
    with multiprocessing.Pool(cores) as pool:
        start_time = time.time()
        results = list(pool.map(run_numpy_match_filter, ab_list))
        total_time = time.time() - start_time
    
    avg_time = total_time / len(data_pairs)
    times = [total_time / len(data_pairs)] * len(data_pairs)  # Approximate per-operation time
    
    return {
        'results': np.array(results),
        'times': np.array(times),
        'total_time': total_time,
        'avg_time': avg_time,
        'method': f'NumPy_CPU_{cores}cores'
    }

def benchmark_method(method_name, filter_func, data_pairs, **kwargs):
    """Benchmark a specific method and return results."""
    print(f"Running {method_name} benchmark...")
    start_time = time.time()
    
    results = []
    times = []
    
    for i, (a, b) in enumerate(data_pairs):
        op_start = time.time()
        
        if method_name.startswith("MLX"):
            result = filter_func(a, b, **kwargs)
        else:
            result = filter_func(a, b)
            
        op_time = time.time() - op_start
        
        results.append(result)
        times.append(op_time)
    
    total_time = time.time() - start_time
    avg_time = np.mean(times)
    
    return {
        'results': np.array(results),
        'times': np.array(times),
        'total_time': total_time,
        'avg_time': avg_time,
        'method': method_name
    }

def save_results(results_dict, output_file, args):
    """Save all benchmark results to HDF5 file."""
    print(f"Saving results to {output_file}")
    
    with h5py.File(output_file, 'w') as f:
        # Save metadata
        f.attrs['timestamp'] = datetime.now().isoformat()
        f.attrs['list_size'] = args.list_size
        f.attrs['array_size'] = args.array_size
        f.attrs['cores_tested'] = args.cores
        
        # Save results for each method
        for method_name, data in results_dict.items():
            grp = f.create_group(method_name)
            
            # Save arrays
            grp.create_dataset('results', data=data['results'])
            grp.create_dataset('times', data=data['times'])
            
            # Save scalars
            grp.attrs['total_time'] = data['total_time']
            grp.attrs['avg_time'] = data['avg_time']
            grp.attrs['method'] = data['method']

def main():
    parser = argparse.ArgumentParser(description="Unified Matched Filter Benchmark")
    parser.add_argument("--list_size", type=int, default=1000, 
                       help="Number of match filter operations")
    parser.add_argument("--array_size", type=int, default=int(2**22), 
                       help="Size of each array")
    parser.add_argument("--cores", type=int, nargs="+", default=[1], 
                       help="List of core counts to test for NumPy (default: [1])")
    parser.add_argument("--output", type=str, default="unified_matched_filter_results.h5", 
                       help="Output HDF5 file")
    
    args = parser.parse_args()
    
    # Check MLX availability
    mlx_available, mx, ifft = check_mlx_available()
    
    # Generate test data
    print(f"Generating {args.list_size} test cases with arrays of size {args.array_size}")
    np.random.seed(42)
    
    data_pairs = []
    for i in range(args.list_size):
        a = np.random.rand(args.array_size) + 1j * np.random.rand(args.array_size)
        b = np.random.rand(args.array_size) + 1j * np.random.rand(args.array_size)
        data_pairs.append((a, b))
    
    # Run benchmarks
    results_dict = {}
    
    # NumPy CPU benchmark (with different core counts)
    for core_count in args.cores:
        method_key = f'NumPy_CPU_{core_count}cores'
        results_dict[method_key] = benchmark_numpy_multicore(data_pairs, core_count)
    
    # MLX benchmarks (if available) - these run single-threaded
    if mlx_available:
        print("Starting MLX CPU benchmark...")
        results_dict['MLX_CPU'] = benchmark_method(
            "MLX_CPU", define_match_filter_mlx, data_pairs,
            device=mx.cpu, mx=mx, ifft=ifft
        )
        
        # MLX GPU benchmark (if available)
        try:
            if mx.metal.is_available():
                print("Starting MLX GPU benchmark...")
                results_dict['MLX_GPU'] = benchmark_method(
                    "MLX_GPU", define_match_filter_mlx, data_pairs,
                    device=mx.gpu, mx=mx, ifft=ifft
                )
        except:
            print("MLX GPU not available, skipping GPU benchmark")
    else:
        print("MLX not available, skipping MLX benchmarks")
    
    # Save results
    save_results(results_dict, args.output, args)
    
    # Simple completion message (like original scripts)
    print(f"Benchmark results saved to {args.output}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
