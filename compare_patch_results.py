import h5py
import numpy as np
import os
import matplotlib.pyplot as plt

def compare_results(mlx_file, numpy_file, plot_title=None):
    # Check if files exist
    if not os.path.exists(mlx_file):
        print(f"File not found: {mlx_file}")
        return
    if not os.path.exists(numpy_file):
        print(f"File not found: {numpy_file}")
        return
    try:
        with h5py.File(mlx_file, 'r') as f_mlx, h5py.File(numpy_file, 'r') as f_np:
            # Check for required datasets
            for ds in ['cpu_times_parallel', 'gpu_times_parallel', 'test_cores']:
                if ds not in f_mlx:
                    print(f"Dataset '{ds}' not found in {mlx_file}")
                    return
            if 'avg_times' not in f_np or 'core_counts' not in f_np:
                print(f"Dataset 'avg_times' or 'core_counts' not found in {numpy_file}")
                return
            mlx_cpu = f_mlx['cpu_times_parallel'][:]
            mlx_gpu = f_mlx['gpu_times_parallel'][:]
            np_cpu = f_np['avg_times'][:]
            numpy_core = f_np['core_counts'][:]
            cores = f_mlx['test_cores'][:]

            # Align NumPy results to MLX core counts
            matching_indices = [np.where(numpy_core == c)[0][0] for c in cores if c in numpy_core]
            np_cpu_matched = np_cpu[matching_indices]

            print(f"Comparing results for cores: {cores}")
            print("\nMLX CPU times per op:", mlx_cpu)
            print("MLX GPU times per op:", mlx_gpu)
            print("NumPy CPU times per op (matched):", np_cpu_matched)

            # Check for NaNs
            print("\nNaN check:")
            print("MLX CPU NaNs:", np.isnan(mlx_cpu).any())
            print("MLX GPU NaNs:", np.isnan(mlx_gpu).any())
            print("NumPy CPU NaNs:", np.isnan(np_cpu_matched).any())

            # Compare timings
            speedup = np_cpu_matched / mlx_gpu
            print("\nSpeedup (NumPy CPU / MLX GPU):", speedup)

            # Plot timings
            plt.figure(figsize=(10,6))
            plt.semilogy(cores, mlx_cpu, marker='s', linestyle='--', label='MLX CPU')
            plt.semilogy(cores, mlx_gpu, marker='^', linestyle='-.', label='MLX GPU')
            plt.semilogy(cores, np_cpu_matched, marker='o', linestyle='-', label='NumPy CPU')
            plt.xlabel('Number of Cores')
            plt.ylabel('Avg Time per Operation (s) [log scale]')
            if plot_title:
                plt.title(plot_title)
            else:
                plt.title('NumPy vs MLX Matched Filter Benchmark')
            plt.legend()
            plt.grid(True, which='both', ls='--', alpha=0.7)
            plt.tight_layout()
            plt.show()

            # Plot speedup
            plt.figure(figsize=(8,5))
            plt.plot(cores, speedup, marker='o', color='tab:green')
            plt.xlabel('Number of Cores')
            plt.ylabel('Speedup (NumPy CPU / MLX GPU)')
            if plot_title:
                plt.title(f'Speedup: {plot_title}')
            else:
                plt.title('MLX GPU Speedup over NumPy CPU')
            plt.grid(True)
            plt.tight_layout()
            plt.show()
    except Exception as e:
        print(f"Error comparing files: {e}")

# Example usage:
compare_results('match_filter_results_21.h5', 'match_filter_results_numpy_21.h5', plot_title='Array Size 2^21')
compare_results('match_filter_results_22.h5', 'match_filter_results_numpy_22.h5', plot_title='Array Size 2^22')