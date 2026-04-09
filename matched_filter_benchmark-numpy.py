# matched_filter_benchmark-numpy.py
# Benchmark matched filter using NumPy (single core, CPU only)
# Saves results to HDF5 file

# Force NumPy to use single thread (relevant for Apple Silicon)
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1" 
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


import numpy as np
import time
import h5py
import argparse
import multiprocessing
import json
# Define matched filter using NumPy

def define_match_filter(a, b):
    product = np.multiply(a.conj(), b)
    ifft_result = np.fft.ifft(product)
    reshaped_result = ifft_result.reshape(ifft_result.size)
    return np.max(reshaped_result)

def run_match_filter(args):
    a,b = args
    return define_match_filter(a,b)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Match Filter Performance with NumPy.")
    parser.add_argument("--list_size", type=int, default=1000, help="Number of match filter operations")
    parser.add_argument("--array_size", type=int, default=int(2**22), help="Size of each array")
    parser.add_argument("--cores", type=int, nargs="+", default=[1,2,3,4,5,6,7,8], help="List of core counts to test")
    parser.add_argument("--no_cpu", action="store_true", help="Skip CPU benchmarking")
    parser.add_argument("--output", type=str, default="match_filter_results_22.h5", help="Output HDF5 file")
    args = parser.parse_args()

    # Initialize arrays
    a_array = np.random.normal(size=args.array_size).astype(np.complex64) + 1j * np.random.normal(size=args.array_size).astype(np.complex64)
    b_array = np.random.normal(size=args.array_size).astype(np.complex64) + 1j * np.random.normal(size=args.array_size).astype(np.complex64)

    # Create list for benchmarking
    ab_list = list(zip([a_array] * args.list_size, [b_array] * args.list_size))

    core_counts = []
    avg_times = []
    total_times = []
    fft_timestamps = {"benchmark_start": time.time(), "core_tests": []}

    for ncores in args.cores:
        with multiprocessing.Pool(ncores) as pool:
            if not args.no_cpu:
                start_time = time.perf_counter()
                fft_start = time.time()
                results = list(pool.map(run_match_filter, ab_list))
                fft_end = time.time()
                total_time = time.perf_counter() - start_time
                avg_time = total_time / args.list_size
            else:
                total_time.append(None)
        core_counts.append(ncores)
        avg_times.append(avg_time)
        total_times.append(total_time)
        fft_timestamps["core_tests"].append({"ncores": ncores, "start": fft_start, "end": fft_end})
        print(f"Cores: {ncores} | Avg time/op: {avg_time:.6f} s | Total time: {total_time:.2f} s")

    fft_timestamps["benchmark_end"] = time.time()

    # Save results
    with h5py.File(args.output, "w") as hdf:
        hdf.create_dataset("core_counts", data=np.array(core_counts))
        hdf.create_dataset("avg_times", data=np.array(avg_times))
        hdf.create_dataset("total_times", data=np.array(total_times))
        hdf.create_dataset("results", data=np.array(results))
        hdf.create_dataset("array_size", data=np.array([args.array_size]))
        hdf.create_dataset("list_size", data=np.array([args.list_size]))

    ts_output = args.output.replace(".h5", "_timestamps.json")
    with open(ts_output, "w") as f:
        json.dump(fft_timestamps, f, indent=2)

    print("Benchmark complete. Results saved to {}".format(args.output))
    print(f"FFT timestamps saved to {ts_output}")

    # Plot results
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8,5))
        plt.plot(core_counts, avg_times, marker='o')
        plt.xlabel('Number of Cores')
        plt.ylabel('Average Time per Operation (s)')
        plt.title('NumPy Matched Filter Benchmark vs. Core Count')
        plt.grid(True)
        plt.show()
    except ImportError:
        print("matplotlib not installed; skipping plot.")
