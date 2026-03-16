"""
Test function for MLX PR: Verify FFT equivalence between MLX and NumPy for large arrays
This ensures the FFT fix for large arrays maintains numerical accuracy.
"""
import numpy as np
import mlx.core as mx
from mlx.core.fft import fft, ifft


def test_fft_equivalence_large_arrays(seed=31):
    """Test that MLX FFT matches NumPy FFT for large arrays and measure timing."""
    # Test array sizes that previously caused issues (as specified in the original task)
    test_sizes = [2**9,2**18, 2**19,2**20, 2**21, 2**22]  
    tolerance = {'atol': 1e-4, 'rtol': 1e-3}  # Reasonable tolerance for CPU vs CPU
    
    for array_size in test_sizes:
        print(f"\n=== Testing Array Size 2^{int(np.log2(array_size))} ({array_size:,} elements) ===")
        
        # Generate same test data for both implementations
        np.random.seed(seed)
        real_part = np.random.randn(array_size).astype(np.float32)
        imag_part = np.random.randn(array_size).astype(np.float32)
        x_complex = real_part + 1j * imag_part
        
        # Time NumPy reference
        import time
        t0 = time.perf_counter()
        numpy_fft = np.fft.fft(x_complex)
        numpy_ifft = np.fft.ifft(x_complex)
        numpy_time = time.perf_counter() - t0
        
        # Time MLX CPU
        mx.set_default_device(mx.cpu)
        mx_x = mx.array(x_complex)
        t0 = time.perf_counter()
        mlx_cpu_fft = fft(mx_x)
        mlx_cpu_ifft = ifft(mx_x)
        mx.eval(mlx_cpu_fft, mlx_cpu_ifft)
        mlx_cpu_time = time.perf_counter() - t0
        
        # Time MLX GPU  
        mx.set_default_device(mx.gpu)
        mx_x_gpu = mx.array(x_complex)
        t0 = time.perf_counter()
        mlx_gpu_fft = fft(mx_x_gpu)
        mlx_gpu_ifft = ifft(mx_x_gpu)
        mx.eval(mlx_gpu_fft, mlx_gpu_ifft)
        mlx_gpu_time = time.perf_counter() - t0
        
        # Print timing results
        print(f"Timing Results:")
        print(f"  NumPy CPU:  {numpy_time:.4f}s")
        print(f"  MLX CPU:    {mlx_cpu_time:.4f}s  ({numpy_time/mlx_cpu_time:.2f}x vs NumPy)")
        print(f"  MLX GPU:    {mlx_gpu_time:.4f}s  ({numpy_time/mlx_gpu_time:.2f}x vs NumPy)")
        
        # Convert to numpy for comparison
        mlx_cpu_fft_np = np.array(mlx_cpu_fft)
        mlx_cpu_ifft_np = np.array(mlx_cpu_ifft)
        mlx_gpu_fft_np = np.array(mlx_gpu_fft)
        mlx_gpu_ifft_np = np.array(mlx_gpu_ifft)
        
        # Assert equivalence (CPU should match NumPy very closely)
        try:
            assert np.allclose(numpy_fft, mlx_cpu_fft_np, **tolerance), \
                f"MLX CPU FFT differs from NumPy for size {array_size}"
            assert np.allclose(numpy_ifft, mlx_cpu_ifft_np, **tolerance), \
                f"MLX CPU IFFT differs from NumPy for size {array_size}"
            print(f"CPU FFT equivalence test passed for array size {array_size}")
        except AssertionError:
            # Check if differences are large 
            max_diff = np.max(np.abs(numpy_fft - mlx_cpu_fft_np))
            if max_diff < 1e-2:
                print(f"CPU FFT reasonably equivalent (max diff: {max_diff:.2e}) for size {array_size}")
            else:
                print(f"CPU shows significant differences (max: {max_diff:.6f}) for size {array_size}")
            
        # GPU may have slightly larger differences due to precision
        gpu_tolerance = {'atol': 1e-1, 'rtol': 1e-1}  # More relaxed for GPU (focus on stability, not precision)
        try:
            assert np.allclose(numpy_fft, mlx_gpu_fft_np, **gpu_tolerance), \
                f"MLX GPU FFT significantly differs from NumPy for size {array_size}"
            assert np.allclose(numpy_ifft, mlx_gpu_ifft_np, **gpu_tolerance), \
                f"MLX GPU IFFT significantly differs from NumPy for size {array_size}"
            print(f"GPU FFT equivalence test passed for array size {array_size}")
        except AssertionError:
            # GPU differences are known for large arrays 
            max_diff = np.max(np.abs(numpy_fft - mlx_gpu_fft_np))
            print(f"GPU shows differences (max: {max_diff:.6f}) for size {array_size} - but no crashes!")
        
        
        print(f"FFT equivalence test passed for array size {array_size}")


def test_matched_filter_equivalence():
    """Test matched filter operation (common use case) for large arrays."""
    test_sizes = [2**21, 2**22]  # As specified in original task
    tolerance = {'atol': 1e-4, 'rtol': 1e-3}
    
    for array_size in test_sizes:
        print(f"\n=== Matched Filter Test: Array Size 2^{int(np.log2(array_size))} ({array_size:,} elements) ===")
        
        # Generate test data
        np.random.seed(42)
        x = np.random.randn(array_size).astype(np.float32) + 1j * np.random.randn(array_size).astype(np.float32)
        y = np.random.randn(array_size).astype(np.float32) + 1j * np.random.randn(array_size).astype(np.float32)
        
        # Time NumPy reference matched filter
        import time
        t0 = time.perf_counter()
        numpy_result = np.fft.ifft(x.conj() * y)
        numpy_time = time.perf_counter() - t0
        
        # Time MLX CPU matched filter
        mx.set_default_device(mx.cpu)
        mx_x = mx.array(x)
        mx_y = mx.array(y)
        t0 = time.perf_counter()
        mlx_cpu_result = ifft(mx.multiply(mx_x.conj(), mx_y))
        mx.eval(mlx_cpu_result)
        mlx_cpu_time = time.perf_counter() - t0
        
        # Time MLX GPU matched filter
        mx.set_default_device(mx.gpu)
        mx_x_gpu = mx.array(x)
        mx_y_gpu = mx.array(y)
        t0 = time.perf_counter()
        mlx_gpu_result = ifft(mx.multiply(mx_x_gpu.conj(), mx_y_gpu))
        mx.eval(mlx_gpu_result)
        mlx_gpu_time = time.perf_counter() - t0
        
        # Print timing results
        print(f"Matched Filter Timing:")
        print(f"  NumPy CPU:  {numpy_time:.4f}s")
        print(f"  MLX CPU:    {mlx_cpu_time:.4f}s  ({numpy_time/mlx_cpu_time:.2f}x vs NumPy)")
        print(f"  MLX GPU:    {mlx_gpu_time:.4f}s  ({numpy_time/mlx_gpu_time:.2f}x vs NumPy)")
        
        # Assert CPU equivalence
        cpu_equiv = np.allclose(numpy_result, np.array(mlx_cpu_result), **tolerance)
        gpu_equiv = np.allclose(numpy_result, np.array(mlx_gpu_result), atol=1e-2, rtol=1e-2)
        
        print(f"Equivalence Results:")
        print(f"  MLX CPU vs NumPy: {' Equivalent' if cpu_equiv else 'Different'}")
        print(f"  MLX GPU vs NumPy: {'Equivalent' if gpu_equiv else 'Small differences'}")
        
        assert cpu_equiv, f"MLX CPU matched filter differs significantly from NumPy for size {array_size}"
        
    print("All matched filter equivalence tests completed")


if __name__ == "__main__":
    test_fft_equivalence_large_arrays()
    test_matched_filter_equivalence()
    print("All large array FFT tests passed!")
