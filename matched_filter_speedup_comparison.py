"""
Create speedup plot comparing single-threaded NumPy vs MLX matched filter performance
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt

# Read NumPy results
with h5py.File('match_filter_numpy_single.h5', 'r') as f:
    numpy_avg_time = f['avg_times'][0]  # Single core result
    print(f"NumPy single-thread time: {numpy_avg_time:.6f} s")

# Read MLX results
with h5py.File('match_filter_mlx_single.h5', 'r') as f:
    mlx_cpu_time = f['cpu_times_parallel'][0]  # Single core result
    mlx_gpu_time = f['gpu_times_parallel'][0]  # Single core result
    print(f"MLX CPU single-thread time: {mlx_cpu_time:.6f} s")
    print(f"MLX GPU single-thread time: {mlx_gpu_time:.6f} s")

# Calculate speedups
cpu_speedup = numpy_avg_time / mlx_cpu_time
gpu_speedup = numpy_avg_time / mlx_gpu_time

print(f"\nSpeedup Results:")
print(f"MLX CPU vs NumPy: {cpu_speedup:.2f}x")
print(f"MLX GPU vs NumPy: {gpu_speedup:.2f}x")

# Create speedup plot
backends = ['NumPy CPU', 'MLX CPU', 'MLX GPU']
times = [numpy_avg_time, mlx_cpu_time, mlx_gpu_time]
speedups = [1.0, cpu_speedup, gpu_speedup]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Execution times
bars1 = ax1.bar(backends, times, color=['blue', 'green', 'red'], alpha=0.7)
ax1.set_ylabel('Time per Operation (seconds)')
ax1.set_title('Single-Thread Matched Filter Performance\nArray Size: 2^20 (1M elements)')
ax1.grid(True, alpha=0.3)

# Add value labels on bars
for bar, time in zip(bars1, times):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{time:.4f}s', ha='center', va='bottom')

# Plot 2: Speedup factors
bars2 = ax2.bar(backends, speedups, color=['blue', 'green', 'red'], alpha=0.7)
ax2.set_ylabel('Speedup Factor (vs NumPy)')
ax2.set_title('Speedup vs NumPy CPU (Single Thread)')
ax2.grid(True, alpha=0.3)
ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5)

# Add value labels on bars
for bar, speedup in zip(bars2, speedups):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{speedup:.2f}x', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('single_thread_speedup_comparison.png', dpi=300, bbox_inches='tight')
print("\nSpeedup plot saved as 'single_thread_speedup_comparison.png'")
plt.show()

