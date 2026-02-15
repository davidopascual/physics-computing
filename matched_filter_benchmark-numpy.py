# matched_filter_benchmark-numpy.py
# Benchmark matched filter using NumPy (single core, CPU only)
# Saves results to HDF5 file

import numpy as np
import time
import h5py
import argparse

# Define matched filter using NumPy

def define_match_filter(a, b):
    product = np.multiply(a.conj(), b)
    ifft_result = np.fft.ifft(product)
    reshaped_result = ifft_result.reshape(ifft_result.size)
    return np.max(reshaped_result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Match Filter Performance with NumPy.")
    parser.add_argument("--list_size", type=int, default=1000, help="Number of match filter operations")
    parser.add_argument("--array_size", type=int, default=int(2**20), help="Size of each array")
    parser.add_argument("--output", type=str, default="match_filter_results_numpy.h5", help="Output HDF5 file")
    args = parser.parse_args()

    # Initialize arrays
    a_array = np.random.normal(size=args.array_size).astype(np.complex64) + 1j * np.random.normal(size=args.array_size).astype(np.complex64)
    b_array = np.random.normal(size=args.array_size).astype(np.complex64) + 1j * np.random.normal(size=args.array_size).astype(np.complex64)

    # Create list for benchmarking
    ab_list = list(zip([a_array] * args.list_size, [b_array] * args.list_size))

    # Benchmark
    start_time = time.perf_counter()
    results = [define_match_filter(a, b) for a, b in ab_list]
    total_time = time.perf_counter() - start_time
    avg_time = total_time / args.list_size

    # Save results
    with h5py.File(args.output, "w") as hdf:
        hdf.create_dataset("avg_time_per_op", data=np.array([avg_time]))
        hdf.create_dataset("total_time", data=np.array([total_time]))
        hdf.create_dataset("results", data=np.array(results))
        hdf.create_dataset("array_size", data=np.array([args.array_size]))
        hdf.create_dataset("list_size", data=np.array([args.list_size]))

    print(f"Benchmark complete. Average time per operation: {avg_time:.6f} seconds.")
    print(f"Results saved to {args.output}")
