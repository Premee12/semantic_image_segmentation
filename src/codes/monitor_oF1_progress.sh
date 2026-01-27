#!/bin/bash
# Monitor oF1 computation progress
# Run this in a separate terminal: watch -n 60 ./monitor_oF1_progress.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "oF1 Computation Progress Monitor"
echo "================================="
echo "Time: $(date)"
echo ""

# Check if predictions are being generated
if [ -d "predictions" ]; then
    PRED_DIRS=$(ls predictions | wc -l)
    echo "Predictions: $PRED_DIRS/13 models completed"

    # Count total prediction files
    TOTAL_PREDS=0
    for dir in predictions/*/; do
        if [ -d "$dir" ]; then
            COUNT=$(ls "$dir"/*.npy 2>/dev/null | wc -l)
            TOTAL_PREDS=$((TOTAL_PREDS + COUNT))
        fi
    done
    echo "Total prediction files: $TOTAL_PREDS"
else
    echo "Predictions: Not started yet"
fi

echo ""

# Check if oF1 computation has started
if [ -d "oF1_results" ]; then
    if [ -f "oF1_results/all_models_oF1_scores.csv" ]; then
        echo "oF1 Results: Complete"
        echo "Results file: oF1_results/all_models_oF1_scores.csv"
    else
        echo "oF1 Results: In progress"
    fi
else
    echo "oF1 Results: Not started yet"
fi

echo ""

# Check for running processes
PYTHON_PROCS=$(pgrep -f "python3.*oF1" | wc -l)
if [ $PYTHON_PROCS -gt 0 ]; then
    echo "Running processes: $PYTHON_PROCS Python oF1 process(es)"
else
    echo "Running processes: None"
fi

echo ""

# Check disk usage
DISK_USAGE=$(df -h . | tail -1 | awk '{print $5}')
echo "Disk usage: $DISK_USAGE"

echo ""

# Check for any error logs
if [ -f "run_output.log" ]; then
    ERRORS=$(grep -c "ERROR\|FAILED\|❌" run_output.log 2>/dev/null || echo "0")
    echo "Errors in log: $ERRORS"
fi

echo "================================="