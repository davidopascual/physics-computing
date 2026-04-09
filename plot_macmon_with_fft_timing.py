#!/usr/bin/env python3
"""
Plot macmon system metrics (utilization, power, memory) vs time with
vertical lines marking when FFT operations start and end.

Usage:
    python plot_macmon_with_fft_timing.py --macmon <macmon.json> --timestamps <timestamps.json>

If --macmon or --timestamps are omitted, the most recently modified file of
each type is used automatically.
"""

import json
import argparse
import glob
import os
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D


def parse_macmon_timestamp(ts_str):
    """Convert ISO 8601 timestamp string to Unix epoch seconds."""
    dt = datetime.fromisoformat(ts_str)
    return dt.timestamp()


def load_macmon(path):
    """Load line-delimited JSON from a macmon pipe output file."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def load_timestamps(path):
    with open(path) as f:
        return json.load(f)


def parse_fft_events(timestamps, t0, cmap_name="tab10"):
    """
    Convert a timestamps dict to a list of FFT event dicts with relative times.

    Supports two formats:
      - numpy benchmark: {"core_tests": [{"ncores": N, "start": t, "end": t}, ...]}
      - mlx single:      {"cpu_fft_starts": [...], "cpu_fft_ends": [...],
                          "gpu_fft_starts": [...], "gpu_fft_ends": [...]}
    """
    events = []

    if "core_tests" in timestamps:
        tests = timestamps["core_tests"]
        cmap = plt.get_cmap(cmap_name)
        colors = [cmap(i / max(len(tests) - 1, 1)) for i in range(len(tests))]
        for i, test in enumerate(tests):
            events.append({
                "label": f"{test['ncores']} core{'s' if test['ncores'] > 1 else ''}",
                "start": test["start"] - t0,
                "end": test["end"] - t0,
                "color": colors[i],
            })

    else:
        # MLX-style: many per-operation timestamps — collapse to one region per device
        if "cpu_fft_starts" in timestamps and timestamps["cpu_fft_starts"]:
            events.append({
                "label": "MLX CPU",
                "start": timestamps["cpu_fft_starts"][0] - t0,
                "end":   timestamps["cpu_fft_ends"][-1]  - t0,
                "color": "steelblue",
            })
        if "gpu_fft_starts" in timestamps and timestamps["gpu_fft_starts"]:
            events.append({
                "label": "MLX GPU",
                "start": timestamps["gpu_fft_starts"][0] - t0,
                "end":   timestamps["gpu_fft_ends"][-1]  - t0,
                "color": "mediumseagreen",
            })

    return events


def add_fft_markers(ax, events, t_max):
    """Add shaded FFT regions and dashed start/end vertical lines to an axis."""
    for ev in events:
        s, e = ev["start"], ev["end"]
        # Clip to visible range
        if e < 0 or s > t_max:
            continue
        ax.axvspan(max(s, 0), min(e, t_max), alpha=0.12, color=ev["color"], zorder=1)
        ax.axvline(s, color=ev["color"], linestyle="--", linewidth=1.4, alpha=0.85, zorder=2)
        ax.axvline(e, color=ev["color"], linestyle=":",  linewidth=1.4, alpha=0.85, zorder=2)


def main():
    parser = argparse.ArgumentParser(
        description="Plot macmon metrics with FFT timing markers"
    )
    parser.add_argument(
        "--macmon", type=str, default=None,
        help="Path to macmon JSON file (default: most recent in monitoring_results/)",
    )
    parser.add_argument(
        "--timestamps", type=str, default=None,
        help="Path to FFT timestamps JSON (default: most recent *_timestamps.json)",
    )
    parser.add_argument(
        "--output", type=str, default="macmon_fft_timing.png",
        help="Output plot filename (default: macmon_fft_timing.png)",
    )
    args = parser.parse_args()

    # --- Auto-detect input files ---
    if args.macmon is None:
        candidates = glob.glob("monitoring_results/*.json")
        if not candidates:
            raise FileNotFoundError(
                "No macmon JSON files found in monitoring_results/. "
                "Pass --macmon explicitly."
            )
        args.macmon = max(candidates, key=os.path.getmtime)
        print(f"Using macmon file:      {args.macmon}")

    if args.timestamps is None:
        candidates = glob.glob("*_timestamps.json")
        if not candidates:
            raise FileNotFoundError(
                "No *_timestamps.json files found. Pass --timestamps explicitly."
            )
        args.timestamps = max(candidates, key=os.path.getmtime)
        print(f"Using timestamps file:  {args.timestamps}")

    # --- Load data ---
    records = load_macmon(args.macmon)
    if not records:
        raise ValueError(f"No valid records found in {args.macmon}")
    print(f"Loaded {len(records)} macmon samples")

    timestamps = load_timestamps(args.timestamps)

    # --- Parse macmon time series ---
    macmon_unix = np.array([parse_macmon_timestamp(r["timestamp"]) for r in records])
    t0 = macmon_unix[0]
    t_rel = macmon_unix - t0
    t_max = t_rel[-1]

    pcpu_util  = np.array([r["pcpu_usage"][1]  * 100 for r in records])
    ecpu_util  = np.array([r["ecpu_usage"][1]  * 100 for r in records])
    gpu_util   = np.array([r["gpu_usage"][1]   * 100 for r in records])

    cpu_power  = np.array([r["cpu_power"]  for r in records])
    gpu_power  = np.array([r["gpu_power"]  for r in records])
    sys_power  = np.array([r["sys_power"]  for r in records])

    ram_total_gb  = records[0]["memory"]["ram_total"]  / 1024**3
    ram_usage_gb  = np.array([r["memory"]["ram_usage"]  / 1024**3 for r in records])
    swap_usage_gb = np.array([r["memory"]["swap_usage"] / 1024**3 for r in records])

    # --- Parse FFT events ---
    events = parse_fft_events(timestamps, t0)
    if not events:
        print("Warning: no FFT events found in timestamps file")
    else:
        print(f"Found {len(events)} FFT event(s)")
        for ev in events:
            print(f"  {ev['label']:20s}  start={ev['start']:+.2f}s  end={ev['end']:+.2f}s")

    # --- Build figure ---
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    chip = records[0].get("soc", {}).get("chip_name", "Apple Silicon")
    fig.suptitle(
        f"System Metrics During Matched Filter Benchmark  [{chip}]",
        fontsize=13, fontweight="bold",
    )

    # --- Panel 1: Utilization ---
    ax1 = axes[0]
    ax1.plot(t_rel, pcpu_util, label="P-CPU",  color="steelblue",     linewidth=1.6)
    ax1.plot(t_rel, ecpu_util, label="E-CPU",  color="darkorange",    linewidth=1.6)
    ax1.plot(t_rel, gpu_util,  label="GPU",    color="mediumseagreen", linewidth=1.6)
    ax1.set_ylabel("Utilization (%)", fontsize=10)
    ax1.set_ylim(-2, 108)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("CPU & GPU Utilization", fontsize=10)
    add_fft_markers(ax1, events, t_max)

    # --- Panel 2: Power ---
    ax2 = axes[1]
    ax2.plot(t_rel, cpu_power, label="CPU power",    color="steelblue",  linewidth=1.6)
    ax2.plot(t_rel, gpu_power, label="GPU power",    color="mediumseagreen", linewidth=1.6)
    ax2.plot(t_rel, sys_power, label="System power", color="gray",        linewidth=1.2, linestyle="--")
    ax2.set_ylabel("Power (W)", fontsize=10)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Power Consumption", fontsize=10)
    add_fft_markers(ax2, events, t_max)

    # --- Panel 3: Memory ---
    ax3 = axes[2]
    ax3.plot(t_rel, ram_usage_gb,  label="RAM used",  color="steelblue", linewidth=1.6)
    ax3.plot(t_rel, swap_usage_gb, label="Swap used", color="coral",     linewidth=1.6)
    ax3.axhline(
        ram_total_gb, color="steelblue", linestyle=":", alpha=0.45,
        label=f"RAM total ({ram_total_gb:.0f} GB)",
    )
    ax3.set_ylabel("Memory (GB)", fontsize=10)
    ax3.set_xlabel("Time (s)", fontsize=10)
    ax3.legend(loc="upper right", fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_title("Memory Usage", fontsize=10)
    add_fft_markers(ax3, events, t_max)

    # --- Bottom legend for FFT events ---
    if events:
        unique_labels = {}
        for ev in events:
            if ev["label"] not in unique_labels:
                unique_labels[ev["label"]] = ev["color"]

        legend_handles = [
            mpatches.Patch(facecolor=c, alpha=0.35, edgecolor=c, label=lbl)
            for lbl, c in unique_labels.items()
        ]
        legend_handles += [
            Line2D([0], [0], color="gray", linestyle="--", linewidth=1.4, label="FFT start"),
            Line2D([0], [0], color="gray", linestyle=":",  linewidth=1.4, label="FFT end"),
        ]
        ncols = min(len(legend_handles), 6)
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=ncols,
            fontsize=9,
            title="FFT Operations",
            title_fontsize=9,
            bbox_to_anchor=(0.5, 0.0),
        )

    plt.tight_layout(rect=[0, 0.07, 1, 1])
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {args.output}")


if __name__ == "__main__":
    main()
