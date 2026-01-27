#!/bin/bash
# Overnight oF1 computation with logging
# Run with: nohup ./run_oF1_overnight.sh > oF1_overnight.log 2>&1 &

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Set CUDA path for GPU to work with TensorFlow XLA
CUDA_NVCC_PATH="$SCRIPT_DIR/../.venv/lib/python3.12/site-packages/nvidia/cuda_nvcc"
if [ -d "$CUDA_NVCC_PATH" ]; then
    export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_NVCC_PATH"
    echo "✓ CUDA path set: $CUDA_NVCC_PATH"
fi
export TF_CPP_MIN_LOG_LEVEL=2

# Activate venv with GPU-enabled TensorFlow
if [ -d "/home/abhishek/semantic_image_segmentation/.venv" ]; then
    source /home/abhishek/semantic_image_segmentation/.venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "../.venv" ]; then
    source ../.venv/bin/activate
fi

LOGFILE="oF1_overnight_$(date +%Y%m%d_%H%M%S).log"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

echo "=========================================================================="
echo "oF1 COMPUTATION - OVERNIGHT RUN"
echo "Started: $TIMESTAMP"
echo "=========================================================================="
echo ""
echo "Log file: $LOGFILE"
echo "Monitor progress: tail -f $LOGFILE"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Script interrupted. Cleaning up..."
    # Kill any background processes if needed
    exit 1
}

# Set trap for cleanup on interrupt
trap cleanup SIGINT SIGTERM

# Pre-flight check
echo "Running pre-flight check..."
./preflight_check.sh
if [ $? -ne 0 ]; then
    echo "❌ Pre-flight check failed. Aborting."
    exit 1
fi

echo ""
echo "=========================================================================="
echo "STARTING OVERNIGHT PROCESSING"
echo "=========================================================================="
echo ""

START_TIME=$(date +%s)

# Step 1: Generate predictions
echo "=========================================================================="
echo "STEP 1: GENERATING PREDICTIONS (estimated 1-2 hours)"
echo "=========================================================================="
echo ""

python3 generate_predictions_for_oF1.py
STEP1_EXIT=$?

if [ $STEP1_EXIT -ne 0 ]; then
    echo ""
    echo "❌ Prediction generation failed (exit code: $STEP1_EXIT)"
    echo "Check the output above for errors."
    echo "You can try to rerun just this step: python3 generate_predictions_for_oF1.py"
    exit 1
fi

echo ""
echo "✓ Prediction generation complete"
echo ""

# Check if predictions were actually generated
if [ ! -d "predictions" ] || [ -z "$(ls -A predictions)" ]; then
    echo "❌ No predictions found! Prediction generation may have failed silently."
    exit 1
fi

echo "Found $(ls predictions | wc -l) model prediction directories"

# Step 2: Compute oF1 scores
echo "=========================================================================="
echo "STEP 2: COMPUTING oF1 SCORES (estimated 2-4 hours)"
echo "=========================================================================="
echo ""

python3 compute_oF1_all_models.py
STEP2_EXIT=$?

if [ $STEP2_EXIT -ne 0 ]; then
    echo ""
    echo "❌ oF1 computation failed (exit code: $STEP2_EXIT)"
    echo "Check the output above for errors."
    echo "You can try to rerun just this step: python3 compute_oF1_all_models.py"
    exit 1
fi

echo ""
echo "✓ oF1 computation complete"
echo ""

# Check if results were actually generated
if [ ! -f "oF1_results/all_models_oF1_scores.csv" ]; then
    echo "❌ oF1 results file not found! Computation may have failed silently."
    exit 1
fi

echo "Results saved to oF1_results/all_models_oF1_scores.csv"

# Calculate runtime
END_TIME=$(date +%s)
RUNTIME=$((END_TIME - START_TIME))
HOURS=$((RUNTIME / 3600))
MINUTES=$(((RUNTIME % 3600) / 60))

echo "=========================================================================="
echo "✅ OVERNIGHT PROCESSING COMPLETE"
echo "=========================================================================="
echo ""
echo "Total runtime: ${HOURS}h ${MINUTES}m"
echo "Completed: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "Results saved to:"
echo "  - predictions/ (organized by model)"
echo "  - oF1_results/all_models_oF1_scores.csv"
echo "  - oF1_results/oF1_comparison_all_models.png"
echo ""
echo "Next steps:"
echo "  1. Review oF1_results/all_models_oF1_scores.csv"
echo "  2. Add oF1 column to Table 7 in Report/main.tex"
echo "  3. Include comparison plot in report"
echo ""
echo "=========================================================================="
