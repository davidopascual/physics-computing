# Physics Computing

Benchmarking scripts for matched filter operations on Apple Silicon, comparing NumPy (CPU, multi-core) against MLX (CPU and GPU). Matched filter is a core computation in astrophysics signal processing.

---

## Scripts

| Script | What it does |
|--------|-------------|
| `matched_filter_benchmark-numpy.py` | NumPy matched filter across multiple core counts using multiprocessing |
| `matched_filter_benchmark-mlx-single.py` | MLX matched filter on CPU and GPU; sequential or batched mode |
| `plot_macmon_with_fft_timing.py` | Plot system metrics (utilization, power, memory) vs time with FFT start/end markers |
| `monitor_single.sh` | Convenience wrapper: runs any benchmark script with macmon monitoring in one command |

The `scripts/` directory contains earlier exploration scripts (`matchfilter_benchmark.py`, `matchfilter_plot.py`, multiprocessing tests). These are kept for reference but are superseded by the scripts above.

---

## Installation

### Requirements

- macOS with Apple Silicon (M1/M2/M3)
- Python 3.11+
- [macmon](https://github.com/vladkens/macmon) for system monitoring (Apple Silicon only)

### Install Python dependencies

```bash
pip install mlx numpy h5py matplotlib
```

### Install macmon

```bash
brew install vladkens/tap/macmon
```

---

## Running the benchmarks

### NumPy (multi-core scaling)

```bash
python matched_filter_benchmark-numpy.py \
  --list_size 1000 \
  --array_size 4194304 \
  --cores 1 2 4 8 \
  --output results_numpy.h5
```

Outputs `results_numpy.h5` and `results_numpy_timestamps.json`.

| Argument | Default | Description |
|----------|---------|-------------|
| `--list_size` | 1000 | Number of matched filter operations |
| `--array_size` | 4194304 (2²²) | Array size per operation |
| `--cores` | 1 2 3 4 5 6 7 8 | Core counts to test |
| `--output` | `match_filter_results_22.h5` | Output HDF5 file |

### MLX (CPU and GPU)

```bash
python matched_filter_benchmark-mlx-single.py \
  --list_size 1000 \
  --array_size 1048576 \
  --batched \
  --output results_mlx.h5
```

Outputs `results_mlx.h5` and `results_mlx_timestamps.json`.

| Argument | Default | Description |
|----------|---------|-------------|
| `--list_size` | 1000 | Number of operations |
| `--array_size` | 1048576 (2²⁰) | Array size — GPU is limited to ≤ 2²⁰ |
| `--batched` | off | Queue all ops into one `mx.eval()` call for accurate throughput measurement |

**Sequential vs batched:** Without `--batched`, `mx.eval()` is called after every operation, which measures per-op latency including Python overhead. With `--batched`, all operations are queued and evaluated at once, measuring true hardware throughput and sustaining GPU load.

---

## Generating the monitoring plot

The plot shows CPU/GPU utilization, power, and memory over time with vertical markers for FFT start and end.

### Step 1 — Run benchmark with macmon

```bash
mkdir -p monitoring_results

macmon pipe --interval 250 --soc-info > monitoring_results/my_run.json &
MACMON_PID=$!
sleep 2

python matched_filter_benchmark-mlx-single.py \
  --list_size 1000 \
  --array_size 1048576 \
  --batched \
  --output results_mlx.h5

sleep 2
kill $MACMON_PID && wait $MACMON_PID
```

Alternatively, use `monitor_single.sh` for a default run (no custom arguments). Each invocation writes a new timestamped file under `monitoring_results/`:

```bash
chmod +x monitor_single.sh
./monitor_single.sh matched_filter_benchmark-mlx-single.py
```

### Step 2 — Generate the plot

Omit `--macmon` and `--timestamps` to auto-detect the most recently modified files:

```bash
python plot_macmon_with_fft_timing.py --output my_run_plot.png
```

Or pass paths explicitly:

```bash
python plot_macmon_with_fft_timing.py \
  --macmon monitoring_results/monitor_matched_filter_benchmark-mlx-single_<timestamp>.json \
  --timestamps results_mlx_timestamps.json \
  --output my_run_plot.png
```

---

## Troubleshooting

**`RuntimeError: Unable to load function four_step_mem_*`**  
MLX GPU FFT fails on arrays larger than 2²⁰. Use `--array_size 1048576`.

**Low GPU utilization in the plot**  
Run with `--batched`. Without it, each GPU operation completes in ~9 ms but macmon samples every 250 ms, so utilization averages to near zero even when the GPU is active.
