#!/bin/bash

# Simple script to monitor a single benchmark with macmon
# Usage: ./monitor_single.sh <script_path> [output_prefix] [conda_env]

if [ $# -lt 1 ]; then
    echo "Usage: $0 <script_path> [output_prefix] [conda_env]"
    echo "Example: $0 matched_filter_benchmark-numpy.py numpy_test"
    echo "Example: $0 scripts/matchfilter_benchmark.py mlx_test pycbc-env"
    exit 1
fi

SCRIPT_PATH="$1"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_PREFIX="${2:-monitoring_results/monitor_$(basename $SCRIPT_PATH .py)}"
OUTPUT_PREFIX="${BASE_PREFIX}_${TIMESTAMP}"
CONDA_ENV="$3"

# Create monitoring directory if it doesn't exist
mkdir -p monitoring_results

echo "Monitoring: $SCRIPT_PATH"
echo "Output prefix: $OUTPUT_PREFIX"
if [ -n "$CONDA_ENV" ]; then
    echo "Using conda environment: $CONDA_ENV"
fi

# Start macmon in background
echo "Starting system monitoring..."
macmon pipe --interval 1000 --soc-info > "${OUTPUT_PREFIX}.json" &
MACMON_PID=$!

# Use python3 if python is not available
PYTHON=$(command -v python || command -v python3)

# Run the script
echo "Running benchmark script..."
if [ -n "$CONDA_ENV" ]; then
    conda run -n "$CONDA_ENV" "$PYTHON" "$SCRIPT_PATH" > "${OUTPUT_PREFIX}.log" 2>&1
else
    "$PYTHON" "$SCRIPT_PATH" > "${OUTPUT_PREFIX}.log" 2>&1
fi
SCRIPT_EXIT_CODE=$?

# Stop monitoring
echo "Stopping monitoring..."
kill $MACMON_PID 2>/dev/null
wait $MACMON_PID 2>/dev/null

echo "Monitoring complete!"
echo "Monitoring data: ${OUTPUT_PREFIX}.json"
echo "Script output: ${OUTPUT_PREFIX}.log"
echo "Script exit code: $SCRIPT_EXIT_CODE"
