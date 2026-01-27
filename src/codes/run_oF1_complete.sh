#!/bin/bash
# Quick Start Script for oF1 Computation
# Run this to process all models automatically

echo "=========================================================================="
echo "oF1 COMPUTATION - AUTOMATED WORKFLOW"
echo "=========================================================================="
echo ""
echo "This script will:"
echo "  1. Generate predictions for all 13 ablation models (~60-90 min)"
echo "  2. Compute oF1 scores for all models (~2-4 hours)"
echo "  3. Create comparison tables and plots"
echo ""
echo "Total runtime: ~3-5 hours"
echo "Memory usage: ~6-8GB peak"
echo "Disk space required: ~5GB"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Cancelled."
    exit 1
fi

# Change to project directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "=========================================================================="
echo "STEP 1: GENERATING PREDICTIONS FOR ALL MODELS"
echo "=========================================================================="
echo ""

python3 generate_predictions_for_oF1.py
if [ $? -ne 0 ]; then
    echo "❌ Prediction generation failed!"
    exit 1
fi

echo ""
echo "=========================================================================="
echo "STEP 2: COMPUTING oF1 SCORES FOR ALL MODELS"
echo "=========================================================================="
echo ""

python3 compute_oF1_all_models.py
if [ $? -ne 0 ]; then
    echo "❌ oF1 computation failed!"
    exit 1
fi

echo ""
echo "=========================================================================="
echo "✓ COMPLETE!"
echo "=========================================================================="
echo ""
echo "Results saved to:"
echo "  - oF1_results/all_models_oF1_scores.csv"
echo "  - oF1_results/oF1_comparison_all_models.png"
echo "  - predictions/ (organized by model)"
echo ""
echo "Next steps:"
echo "  1. Review oF1_results/all_models_oF1_scores.csv"
echo "  2. Add oF1 column to Table 7 in Report/main.tex"
echo "  3. Include comparison plot in report if needed"
echo ""
echo "=========================================================================="
