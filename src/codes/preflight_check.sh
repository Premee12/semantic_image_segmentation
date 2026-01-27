#!/bin/bash
# Pre-flight check for oF1 computation
# Verifies all requirements before running overnight

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv with GPU-enabled TensorFlow
if [ -d "/home/abhishek/semantic_image_segmentation/.venv" ]; then
    source /home/abhishek/semantic_image_segmentation/.venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "../.venv" ]; then
    source ../.venv/bin/activate
fi

echo "=========================================================================="
echo "PRE-FLIGHT CHECK - oF1 COMPUTATION"
echo "=========================================================================="
echo ""

ERRORS=0
WARNINGS=0

# Check 1: Python version
echo "✓ Checking Python version..."
python3 --version || { echo "❌ Python not found"; ERRORS=$((ERRORS+1)); }

# Check 2: Virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✓ Virtual environment active: $VIRTUAL_ENV"
else
    echo "⚠️  No virtual environment active (recommended but not required)"
    WARNINGS=$((WARNINGS+1))
fi

# Check 3: Dataset directory
echo "✓ Checking dataset..."
DATASET_DIR="$SCRIPT_DIR/recodai-luc-scientific-image-forgery-detection"
if [ -d "$DATASET_DIR/train_images" ] && [ -d "$DATASET_DIR/train_masks" ]; then
    TRAIN_IMAGES=$(find "$DATASET_DIR/train_images" -type f | wc -l)
    TRAIN_MASKS=$(find "$DATASET_DIR/train_masks" -name "*.npy" | wc -l)
    echo "  - Train images: $TRAIN_IMAGES files"
    echo "  - Train masks: $TRAIN_MASKS files"
    if [ $TRAIN_MASKS -lt 2000 ]; then
        echo "❌ Insufficient mask files (expected ~2700+)"
        ERRORS=$((ERRORS+1))
    fi
else
    echo "❌ Dataset directory not found or incomplete"
    ERRORS=$((ERRORS+1))
fi

# Check 4: Model files
echo "✓ Checking model files..."
cd "$SCRIPT_DIR"
MODELS_FOUND=0
MODELS_MISSING=0

for model in \
    "data/unet_scratch_bce.keras" \
    "data/unet_scratch_dice.keras" \
    "data/unet_scratch_bce_dice.keras" \
    "data/unet_resnet50_bce_dice.keras" \
    "data/unet_scratch_region_full.keras" \
    "data/unet_scratch_region_context.keras" \
    "data/unet_scratch_region_tight.keras" \
    "data/deeplabv3plus_resnet50_full_data/best_model.h5" \
    "data/deeplabv3plus_resnet101/best_model.h5" \
    "data/deeplabv3plus_data25/best_model.h5" \
    "data/deeplabv3plus_data50/best_model.h5" \
    "data/deeplabv3plus_data75/best_model.h5" \
    "data/deeplabv3plus_data100/best_model.h5"
do
    if [ -f "$model" ]; then
        MODELS_FOUND=$((MODELS_FOUND+1))
    else
        echo "  ⚠️  Missing: $model"
        MODELS_MISSING=$((MODELS_MISSING+1))
        WARNINGS=$((WARNINGS+1))
    fi
done

echo "  - Found: $MODELS_FOUND/13 models"
echo "  - Missing: $MODELS_MISSING/13 models"

# Check 5: Python packages
echo "✓ Checking required packages..."
python3 -c "import tensorflow; print('  - TensorFlow:', tensorflow.__version__)" 2>/dev/null || { echo "❌ TensorFlow not installed"; ERRORS=$((ERRORS+1)); }
python3 -c "import numpy; print('  - NumPy:', numpy.__version__)" 2>/dev/null || { echo "❌ NumPy not installed"; ERRORS=$((ERRORS+1)); }
python3 -c "import pandas; print('  - Pandas:', pandas.__version__)" 2>/dev/null || { echo "❌ Pandas not installed"; ERRORS=$((ERRORS+1)); }
python3 -c "import scipy; print('  - SciPy:', scipy.__version__)" 2>/dev/null || { echo "❌ SciPy not installed"; ERRORS=$((ERRORS+1)); }
python3 -c "import PIL; print('  - Pillow:', PIL.__version__)" 2>/dev/null || { echo "❌ Pillow not installed"; ERRORS=$((ERRORS+1)); }
python3 -c "import sklearn; print('  - Scikit-learn:', sklearn.__version__)" 2>/dev/null || { echo "❌ Scikit-learn not installed"; ERRORS=$((ERRORS+1)); }
python3 -c "import tqdm; print('  - tqdm:', tqdm.__version__)" 2>/dev/null || { echo "⚠️  tqdm not installed (will install automatically)"; WARNINGS=$((WARNINGS+1)); }

# Check 6: Disk space
echo "✓ Checking disk space..."
AVAILABLE=$(df -h "$SCRIPT_DIR" | tail -1 | awk '{print $4}')
AVAILABLE_GB=$(df -BG "$SCRIPT_DIR" | tail -1 | awk '{print $4}' | sed 's/G//')
echo "  - Available: $AVAILABLE"
if [ $AVAILABLE_GB -lt 10 ]; then
    echo "❌ Insufficient disk space (need ~10GB, have ${AVAILABLE_GB}GB)"
    ERRORS=$((ERRORS+1))
fi

# Check 7: Memory
echo "✓ Checking system memory..."
TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
echo "  - Total RAM: ${TOTAL_MEM}GB"
if [ $TOTAL_MEM -lt 8 ]; then
    echo "⚠️  Low memory (recommended 16GB, have ${TOTAL_MEM}GB)"
    WARNINGS=$((WARNINGS+1))
fi

# Check 8: Script files
echo "✓ Checking script files..."
for script in "generate_predictions_for_oF1.py" "compute_oF1_all_models.py" "run_oF1_complete.sh"; do
    if [ -f "$script" ]; then
        echo "  ✓ $script"
    else
        echo "  ❌ $script not found"
        ERRORS=$((ERRORS+1))
    fi
done

# Summary
echo ""
echo "=========================================================================="
echo "SUMMARY"
echo "=========================================================================="
echo "Errors: $ERRORS"
echo "Warnings: $WARNINGS"
echo ""

if [ $ERRORS -gt 0 ]; then
    echo "❌ PRE-FLIGHT CHECK FAILED"
    echo "Please fix the errors above before running."
    echo ""
    exit 1
elif [ $WARNINGS -gt 0 ]; then
    echo "⚠️  PRE-FLIGHT CHECK PASSED WITH WARNINGS"
    echo "Some models are missing, but processing can continue."
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        exit 1
    fi
else
    echo "✅ PRE-FLIGHT CHECK PASSED"
    echo "All systems ready for overnight processing!"
    echo ""
fi

# Estimate runtime
echo "=========================================================================="
echo "ESTIMATED RUNTIME"
echo "=========================================================================="
echo "Models to process: $MODELS_FOUND"
echo "Images per model: ~1,026 (validation set)"
echo ""
echo "Estimated time:"
echo "  - Prediction generation: $(($MODELS_FOUND * 5)) - $(($MODELS_FOUND * 10)) minutes"
echo "  - oF1 computation: $(($MODELS_FOUND * 10)) - $(($MODELS_FOUND * 20)) minutes"
echo "  - Total: $(($MODELS_FOUND * 15 / 60)) - $(($MODELS_FOUND * 30 / 60)) hours"
echo ""
echo "=========================================================================="
echo "✓ Ready to run: ./run_oF1_complete.sh"
echo "=========================================================================="
