# matchfilter_benchmark.py runs benchmark and saves result to HDF5 file
# matchfilter_plot.py reads from HDF5 file to generate plots

# how to run benchmark:
#   python matchfilter_benchmark.py --list_size_cpu 500 --list_size_gpu 1000 --cores 2 4 8
#   (this command benchmarks 500 CPU operations, 5000 GPU operations (uses 2,4, and 8 cores for testing))

import numpy as np
import mlx.core as mx
from mlx.core.fft import ifft
import time
import multiprocessing
import h5py
import argparse

def define_match_filter(a, b, stream):
    product = mx.multiply(a.conj(), b, stream=stream)
    ifft_result = ifft(product, stream=stream)
    reshaped_result = ifft_result.reshape(ifft_result.size)
    return mx.eval(mx.max(reshaped_result, stream=stream))

def frozen_cpumatchfilter(ab):
    return define_match_filter(*ab, stream=mx.cpu)

def frozen_gpumatchfilter(ab):
    return define_match_filter(*ab, stream=mx.gpu)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Match Filter Performance on CPU and GPU.")
    parser.add_argument("--list_size_cpu", type=int, default=1000, help="Number of match filter operations for CPU")
    parser.add_argument("--list_size_gpu", type=int, default=10000, help="Number of match filter operations for GPU")
    parser.add_argument("--array_size", type=int, default=int(2**20), help="Size of each array")
    parser.add_argument("--cores", type=int, nargs="+", default=[1,2,3,4,5,6,7,8], help="List of core counts to test")
    parser.add_argument("--no_cpu", action="store_true", help="Skip CPU benchmarking")
    parser.add_argument("--no_gpu", action="store_true", help="Skip GPU benchmarking")
    parser.add_argument("--output", type=str, default="match_filter_results.h5", help="Output HDF5 file")

    args = parser.parse_args()

    # initialize arrays
    a_array = mx.array(mx.random.normal(shape=(args.array_size,), dtype=mx.float32) +
                       1j * mx.random.normal(shape=(args.array_size,), dtype=mx.float32)) 
    b_array = mx.array(mx.random.normal(shape=(args.array_size,), dtype=mx.float32) +
                       1j * mx.random.normal(shape=(args.array_size,), dtype=mx.float32)) 

    # create lists for CPU and GPU processing
    ab_list_cpu = list(zip([a_array] * args.list_size_cpu, [b_array] * args.list_size_cpu))
    ab_list_gpu = list(zip([a_array] * args.list_size_gpu, [b_array] * args.list_size_gpu))

    cpu_times_parallel = []
    gpu_times_parallel = []

    for ncores in args.cores:
        with multiprocessing.Pool(ncores) as pool:
            if not args.no_cpu:
                start_time = time.perf_counter()
                list(pool.map(frozen_cpumatchfilter, ab_list_cpu))
                cpu_time_parallel = time.perf_counter() - start_time
                cpu_times_parallel.append(cpu_time_parallel / args.list_size_cpu)
            else:
                cpu_times_parallel.append(None)  # Placeholder

            if not args.no_gpu:
                start_time = time.perf_counter()
                list(pool.map(frozen_gpumatchfilter, ab_list_gpu))
                gpu_time_parallel = time.perf_counter() - start_time
                gpu_times_parallel.append(gpu_time_parallel / args.list_size_gpu)
            else:
                gpu_times_parallel.append(None)  # Placeholder

    # save results
    with h5py.File(args.output, "w") as hdf:
        hdf.create_dataset("test_cores", data=np.array(args.cores))
        if not args.no_cpu:
            hdf.create_dataset("cpu_times_parallel", data=np.array(cpu_times_parallel))
        if not args.no_gpu:
            hdf.create_dataset("gpu_times_parallel", data=np.array(gpu_times_parallel))

    print(f"Benchmark results saved to {args.output}")