import numpy as np
import h5py
import matplotlib.pyplot as plt

# Load NumPy results
with h5py.File('match_filter_results_numpy.h5', 'r') as hdf:
    numpy_core_counts = hdf['core_counts'][:]
    numpy_avg_times = hdf['avg_times'][:]

# Load MLX results
with h5py.File('match_filter_results.h5', 'r') as hdf:
    mlx_core_counts = hdf['test_cores'][:]
    mlx_cpu_times = hdf['cpu_times_parallel'][:]
    if 'gpu_times_parallel' in hdf:
        mlx_gpu_times = hdf['gpu_times_parallel'][:]
    else:
        mlx_gpu_times = None

# --- Plot 1: Semilogy comparison ---
fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.semilogy(numpy_core_counts, numpy_avg_times, marker='o', linestyle='-', color='tab:blue', label='NumPy CPU')
ax1.semilogy(mlx_core_counts, mlx_cpu_times, marker='s', linestyle='--', color='tab:orange', label='MLX CPU')
if mlx_gpu_times is not None:
    ax1.semilogy(mlx_core_counts, mlx_gpu_times, marker='^', linestyle='-.', color='tab:green', label='MLX GPU')
ax1.set_xlabel('Number of Cores', fontsize=14)
ax1.set_ylabel('Average Time per Operation (s) [log scale]', fontsize=14)
ax1.set_title('NumPy vs MLX Matched Filter Benchmark', fontsize=16)
all_cores = sorted(set(numpy_core_counts) | set(mlx_core_counts))
ax1.set_xticks(all_cores)
ax1.legend(fontsize=12)
ax1.grid(True, which='both', ls='--', alpha=0.7)
fig1.tight_layout()
fig1.savefig('numpy_mlx_comparison.png', dpi=150)

# --- Plot 2: Speedup factor (relative to NumPy 1-core baseline) ---
baseline = numpy_avg_times[numpy_core_counts == 1][0]

numpy_speedup = baseline / numpy_avg_times
mlx_cpu_speedup = baseline / mlx_cpu_times
if mlx_gpu_times is not None:
    mlx_gpu_speedup = baseline / mlx_gpu_times

fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.plot(numpy_core_counts, numpy_speedup, marker='o', linestyle='-', color='tab:blue', label='NumPy CPU')
ax2.plot(mlx_core_counts, mlx_cpu_speedup, marker='s', linestyle='--', color='tab:orange', label='MLX CPU')
if mlx_gpu_times is not None:
    ax2.plot(mlx_core_counts, mlx_gpu_speedup, marker='^', linestyle='-.', color='tab:green', label='MLX GPU')

# ideal linear scaling reference
ideal_cores = np.array(all_cores)
ax2.plot(ideal_cores, ideal_cores.astype(float), linestyle=':', color='gray', label='Ideal linear scaling')

ax2.set_xlabel('Number of Cores', fontsize=14)
ax2.set_ylabel('Speedup Factor (relative to NumPy 1-core)', fontsize=14)
ax2.set_title('Speedup Factor vs Number of Cores', fontsize=16)
ax2.set_xticks(all_cores)
ax2.legend(fontsize=12)
ax2.grid(True, ls='--', alpha=0.7)
fig2.tight_layout()
fig2.savefig('speedup_factor.png', dpi=150)

plt.show()
