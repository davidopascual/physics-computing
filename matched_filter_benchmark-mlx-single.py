# matched_filter_benchmark-mlx-single.py
# Benchmark matched filter using MLX (CPU and GPU)
# Saves results to HDF5 file and FFT timestamps to JSON
#
# Two execution modes:
#   Sequential (default): one mx.eval() per operation — accurate per-op latency
#   Batched (--batched):  one mx.eval() for all operations — accurate throughput,
#                         sustains GPU load so macmon captures real utilization

import numpy as np
import mlx.core as mx
from mlx.core.fft import ifft
import time
import h5py
import argparse
import json


def run_sequential(a_array, b_array, list_size, device, device_name, timestamps):
    """Original mode: eval() after every operation. Measures per-op latency."""
    mx.set_default_device(device)
    mx_a = mx.array(a_array)
    mx_b = mx.array(b_array)
    product = mx.multiply(mx_a.conj(), mx_b)

    operation_results = []
    fft_starts = []
    fft_ends = []

    start_time = time.perf_counter()
    for i in range(list_size):
        fft_starts.append(time.time())
        ifft_result = ifft(product)
        mx.eval(ifft_result)
        fft_ends.append(time.time())
        operation_results.append(float(np.max(np.abs(np.array(ifft_result)))))
        if (i + 1) % 100 == 0:
            print(f"  Completed {i + 1}/{list_size} operations")

    total_time = time.perf_counter() - start_time
    avg_time = total_time / list_size

    timestamps[device_name.lower() + '_fft_starts'] = fft_starts
    timestamps[device_name.lower() + '_fft_ends'] = fft_ends

    return operation_results, avg_time, total_time


def run_batched(a_array, b_array, list_size, device, device_name, timestamps, n_unique=20):
    """
    Batched mode: queue all list_size operations then eval() once.
    Uses n_unique distinct input pairs (cycled) so MLX cannot collapse the graph.
    This sustains full GPU load and gives accurate throughput measurements.
    """
    mx.set_default_device(device)

    # Pre-generate n_unique distinct input pairs on the target device
    k = min(n_unique, list_size)
    np.random.seed(0)
    pairs = []
    for _ in range(k):
        a = mx.array(
            np.random.normal(size=a_array.shape).astype(np.float32)
            + 1j * np.random.normal(size=a_array.shape).astype(np.float32)
        ).astype(mx.complex64)
        b = mx.array(
            np.random.normal(size=b_array.shape).astype(np.float32)
            + 1j * np.random.normal(size=b_array.shape).astype(np.float32)
        ).astype(mx.complex64)
        pairs.append((a, b))

    # Build the full computation graph (lazy — nothing executes yet)
    batch = []
    for i in range(list_size):
        a, b = pairs[i % k]
        result = mx.max(mx.abs(ifft(mx.multiply(a.conj(), b))))
        batch.append(result)

    # Warm-up: run one small eval so Metal shaders are compiled before timing
    warmup = mx.max(mx.abs(ifft(mx.multiply(pairs[0][0].conj(), pairs[0][1]))))
    mx.eval(warmup)

    # Time the full batch — GPU stays loaded for this entire call
    fft_start = time.time()
    mx.eval(batch)
    fft_end = time.time()

    total_time = fft_end - fft_start
    avg_time = total_time / list_size

    timestamps[device_name.lower() + '_fft_starts'] = [fft_start]
    timestamps[device_name.lower() + '_fft_ends']   = [fft_end]

    operation_results = [float(r.item()) for r in batch]
    print(f"  Completed {list_size}/{list_size} operations (batched)")

    return operation_results, avg_time, total_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark Match Filter Performance with MLX (CPU and GPU)."
    )
    parser.add_argument("--list_size",  type=int, default=1000,
                        help="Number of match filter operations")
    parser.add_argument("--array_size", type=int, default=int(2**20),
                        help="Size of each array (GPU max: 2^20 = 1048576)")
    parser.add_argument("--output",     type=str, default="match_filter_results_mlx_single.h5",
                        help="Output HDF5 file")
    parser.add_argument("--batched",    action="store_true",
                        help="Batch all ops into one eval() call — shows real GPU utilization")
    parser.add_argument("--n_unique",   type=int, default=20,
                        help="Number of unique input pairs used in batched mode (default: 20)")
    args = parser.parse_args()

    print(f"Array size: 2^{args.array_size.bit_length()-1} ({args.array_size})")
    print(f"Operations: {args.list_size}")
    print(f"Mode:       {'batched' if args.batched else 'sequential'}")

    np.random.seed(42)
    a_array = (np.random.normal(size=args.array_size).astype(np.float32)
               + 1j * np.random.normal(size=args.array_size).astype(np.float32)).astype(np.complex64)
    b_array = (np.random.normal(size=args.array_size).astype(np.float32)
               + 1j * np.random.normal(size=args.array_size).astype(np.float32)).astype(np.complex64)

    devices      = [mx.cpu, mx.gpu]
    device_names = ["CPU", "GPU"]

    results    = {}
    timestamps = {}

    for device, device_name in zip(devices, device_names):
        print(f"\nTesting MLX {device_name}...")
        if args.batched:
            op_results, avg_time, total_time = run_batched(
                a_array, b_array, args.list_size, device, device_name,
                timestamps, n_unique=args.n_unique
            )
        else:
            op_results, avg_time, total_time = run_sequential(
                a_array, b_array, args.list_size, device, device_name, timestamps
            )

        results[device_name.lower()] = {
            "avg_time":   avg_time,
            "total_time": total_time,
            "results":    op_results,
        }
        print(f"MLX {device_name} | Avg time/op: {avg_time:.6f} s | Total: {total_time:.2f} s")

    # Save HDF5
    with h5py.File(args.output, "w") as hdf:
        hdf.create_dataset("array_size", data=np.array([args.array_size]))
        hdf.create_dataset("list_size",  data=np.array([args.list_size]))
        for dn in ["cpu", "gpu"]:
            grp = hdf.create_group(dn)
            grp.create_dataset("avg_time",   data=np.array([results[dn]["avg_time"]]))
            grp.create_dataset("total_time", data=np.array([results[dn]["total_time"]]))
            grp.create_dataset("results",    data=np.array(results[dn]["results"]))

    # Save timestamps
    ts_file = args.output.replace(".h5", "_timestamps.json")
    with open(ts_file, "w") as f:
        json.dump(timestamps, f, indent=2)

    print(f"\nResults saved to {args.output}")
    print(f"Timestamps saved to {ts_file}")

    cpu_time = results["cpu"]["avg_time"]
    gpu_time = results["gpu"]["avg_time"]
    print(f"MLX GPU speedup over MLX CPU: {cpu_time / gpu_time:.2f}x")
