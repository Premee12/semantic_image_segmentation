"""
Generate predictions from ALL trained models for oF1 computation.

This script processes all models from your ablation studies and generates
predictions for oF1 evaluation as required by the project guidelines.

IMPORTANT: Uses validation set (same as notebooks) for consistent evaluation.
"""

import sys
import os
from pathlib import Path

# Configure GPU before importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF warnings
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU

# Set CUDA path for libdevice - required for XLA compilation on GPU
VENV_PATH = Path(__file__).parent.parent / '.venv'
CUDA_DIR = VENV_PATH / 'lib/python3.12/site-packages/nvidia/cuda_nvcc'
if CUDA_DIR.exists():
    os.environ['XLA_FLAGS'] = f'--xla_gpu_cuda_data_dir={CUDA_DIR}'
else:
    # Fallback: try to find it
    import site
    for sp in site.getsitepackages():
        cuda_path = Path(sp) / 'nvidia/cuda_nvcc'
        if cuda_path.exists():
            os.environ['XLA_FLAGS'] = f'--xla_gpu_cuda_data_dir={cuda_path}'
            break

# Install missing packages if needed
try:
    import numpy as np
    import pandas as pd
    from PIL import Image
    from tqdm import tqdm
    import tensorflow as tf
    from tensorflow import keras
    import json
    import time
    
    # Configure GPU memory growth to avoid OOM errors
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ GPU detected: {len(gpus)} device(s)")
            print(f"  {gpus[0].name}")
        except RuntimeError as e:
            print(f"⚠️  GPU configuration error: {e}")
    else:
        print("⚠️  No GPU detected, using CPU (will be very slow)")
        
except ImportError as e:
    print(f"❌ Missing package: {e}")
    print("Installing required packages...")
    os.system("pip install -q numpy pandas pillow tqdm tensorflow scikit-learn")
    print("Please run the script again after installation.")
    sys.exit(1)

# Configuration
DATA_ROOT = Path(__file__).parent / "recodai-luc-scientific-image-forgery-detection"
BASE_OUTPUT_DIR = Path(__file__).parent / "predictions"
BASE_OUTPUT_DIR.mkdir(exist_ok=True)

# Use validation set (same split as notebooks: 70-20-10 train/val/test)
SEED = 42
USE_VALIDATION_ONLY = True  # Set to False to use all images

# ALL models for ablation studies (as per project guidelines)
MODEL_CONFIGS = {
    # U-Net Ablation Studies
    'unet_scratch_bce': {
        'path': 'data/unet_scratch_bce.keras',
        'input_size': (512, 512),
        'description': 'U-Net (scratch) + BCE',
        'category': 'U-Net Loss Functions'
    },
    'unet_scratch_dice': {
        'path': 'data/unet_scratch_dice.keras',
        'input_size': (512, 512),
        'description': 'U-Net (scratch) + Dice',
        'category': 'U-Net Loss Functions'
    },
    'unet_scratch_bce_dice': {
        'path': 'data/unet_scratch_bce_dice.keras',
        'input_size': (512, 512),
        'description': 'U-Net (scratch) + BCE+Dice',
        'category': 'U-Net Loss Functions'
    },
    'unet_resnet50_bce_dice': {
        'path': 'data/unet_resnet50_bce_dice.keras',
        'input_size': (512, 512),
        'description': 'U-Net (ResNet50) + BCE+Dice',
        'category': 'U-Net Pretraining'
    },
    'unet_scratch_region_full': {
        'path': 'data/unet_scratch_region_full.keras',
        'input_size': (512, 512),
        'description': 'U-Net + Full Image',
        'category': 'U-Net Data Augmentation'
    },
    'unet_scratch_region_context': {
        'path': 'data/unet_scratch_region_context.keras',
        'input_size': (512, 512),
        'description': 'U-Net + Forged+Context',
        'category': 'U-Net Data Augmentation'
    },
    'unet_scratch_region_tight': {
        'path': 'data/unet_scratch_region_tight.keras',
        'input_size': (512, 512),
        'description': 'U-Net + Forged Tight',
        'category': 'U-Net Data Augmentation'
    },
    # DeepLabV3+ Ablation Studies
    'deeplabv3_resnet50': {
        'path': 'data/deeplabv3plus_resnet50_full_data/best_model.h5',
        'input_size': (512, 512),
        'description': 'DeepLabV3+ (ResNet-50)',
        'category': 'DeepLabV3+ Backbone'
    },
    'deeplabv3_resnet101': {
        'path': 'data/deeplabv3plus_resnet101/best_model.h5',
        'input_size': (512, 512),
        'description': 'DeepLabV3+ (ResNet-101)',
        'category': 'DeepLabV3+ Backbone'
    },
    # Additional DeepLabV3+ ablations (data efficiency)
    'deeplabv3_data25': {
        'path': 'data/deeplabv3plus_data25/best_model.h5',
        'input_size': (512, 512),
        'description': 'DeepLabV3+ (25% data)',
        'category': 'DeepLabV3+ Data Efficiency'
    },
    'deeplabv3_data50': {
        'path': 'data/deeplabv3plus_data50/best_model.h5',
        'input_size': (512, 512),
        'description': 'DeepLabV3+ (50% data)',
        'category': 'DeepLabV3+ Data Efficiency'
    },
    'deeplabv3_data75': {
        'path': 'data/deeplabv3plus_data75/best_model.h5',
        'input_size': (512, 512),
        'description': 'DeepLabV3+ (75% data)',
        'category': 'DeepLabV3+ Data Efficiency'
    },
    'deeplabv3_data100': {
        'path': 'data/deeplabv3plus_data100/best_model.h5',
        'input_size': (512, 512),
        'description': 'DeepLabV3+ (100% data)',
        'category': 'DeepLabV3+ Data Efficiency'
    },
}

# Process 'all', 'unet_only', 'deeplabv3_only', or list of specific models
MODELS_TO_PROCESS = 'all'

def get_validation_split():
    """Get validation set using same split as notebooks (70-20-10)."""
    from sklearn.model_selection import train_test_split
    
    # Prepare dataset
    image_authentic = list((DATA_ROOT / "train_images" / "authentic").glob("*"))
    image_forged = list((DATA_ROOT / "train_images" / "forged").glob("*"))
    
    image_paths = []
    labels = []
    
    for img_path in image_authentic:
        mask_path = DATA_ROOT / "train_masks" / f"{img_path.stem}.npy"
        if mask_path.exists():
            image_paths.append(img_path)
            labels.append('authentic')
    
    for img_path in image_forged:
        mask_path = DATA_ROOT / "train_masks" / f"{img_path.stem}.npy"
        if mask_path.exists():
            image_paths.append(img_path)
            labels.append('forged')
    
    # Same split as notebooks: 70-20-10
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels, test_size=0.3, stratify=labels, random_state=SEED
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.33, stratify=temp_labels, random_state=SEED
    )
    
    print(f"✓ Dataset split: Train={len(train_paths)}, Val={len(val_paths)}, Test={len(test_paths)}")
    return val_paths

def load_model(model_key):
    """Load trained model."""
    config = MODEL_CONFIGS[model_key]
    model_path = Path(config['path'])
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return None, None
    
    print(f"Loading model: {config['description']}")
    
    try:
        model = keras.models.load_model(model_path, compile=False)
        print(f"✓ Model loaded successfully")
        return model, config['input_size']
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None

def generate_predictions(model, model_name, input_size, validation_images=None, threshold=0.5):
    """Generate predictions for all images for a specific model."""
    
    # Create model-specific output directory
    output_dir = BASE_OUTPUT_DIR / model_name
    output_dir.mkdir(exist_ok=True)
    
    # Get images to process
    if validation_images is not None and USE_VALIDATION_ONLY:
        all_images = validation_images
        print(f"  Using validation set: {len(all_images)} images")
    else:
        image_authentic = list((DATA_ROOT / "train_images" / "authentic").glob("*"))
        image_forged = list((DATA_ROOT / "train_images" / "forged").glob("*"))
        all_images = image_authentic + image_forged
        print(f"  Using all images: {len(all_images)} images")
    
    print(f"  Input size: {input_size}")
    print(f"  Output: {output_dir}")
    
    generated_count = 0
    skipped_count = 0
    start_time = time.time()
    
    for img_path in tqdm(all_images, desc=f"  {model_name}", leave=False):
        try:
            # Get corresponding mask filename
            mask_filename = img_path.stem + ".npy"
            output_file = output_dir / mask_filename
            
            # Skip if already exists
            if output_file.exists():
                skipped_count += 1
                continue
            
            # Load and preprocess image
            img = Image.open(img_path).convert('RGB')
            img = img.resize(input_size)
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Generate prediction
            pred = model.predict(img_array, verbose=0)
            
            # Handle different output shapes
            if len(pred.shape) == 4:
                pred_mask = pred[0, :, :, 0]
            else:
                pred_mask = pred[0]
            
            # Binarize
            pred_binary = (pred_mask > threshold).astype(np.uint8)
            
            # Save prediction
            np.save(output_file, pred_binary)
            generated_count += 1
            
        except Exception as e:
            print(f"\n⚠️  Error processing {img_path.name}: {e}")
            continue
        
        # Periodic memory cleanup
        if generated_count % 100 == 0:
            import gc
            gc.collect()
    
    elapsed = time.time() - start_time
    print(f"  ✓ Generated {generated_count}/{len(all_images)} predictions in {elapsed:.1f}s (skipped {skipped_count} existing)")
    
    return generated_count + skipped_count, output_dir

def process_all_models():
    """Process all models according to project guidelines."""
    
    # Get validation split
    if USE_VALIDATION_ONLY:
        validation_images = get_validation_split()
    else:
        validation_images = None
    
    # Determine which models to process
    if MODELS_TO_PROCESS == 'all':
        models_to_run = list(MODEL_CONFIGS.keys())
    elif MODELS_TO_PROCESS == 'unet_only':
        models_to_run = [k for k in MODEL_CONFIGS.keys() if 'unet' in k]
    elif MODELS_TO_PROCESS == 'deeplabv3_only':
        models_to_run = [k for k in MODEL_CONFIGS.keys() if 'deeplabv3' in k]
    elif isinstance(MODELS_TO_PROCESS, list):
        models_to_run = MODELS_TO_PROCESS
    else:
        print(f"❌ Invalid MODELS_TO_PROCESS: {MODELS_TO_PROCESS}")
        return [], 0, 0
    
    print(f"Will process {len(models_to_run)} models\n")
    
    # Track results
    results_summary = []
    successful = 0
    failed = 0
    
    for model_key in models_to_run:
        print(f"\n{'='*70}")
        print(f"[{successful + failed + 1}/{len(models_to_run)}] {MODEL_CONFIGS[model_key]['description']}")
        print(f"{'='*70}")
        
        # Load model
        model, input_size = load_model(model_key)
        if model is None:
            failed += 1
            results_summary.append({
                'model': model_key,
                'description': MODEL_CONFIGS[model_key]['description'],
                'category': MODEL_CONFIGS[model_key]['category'],
                'status': 'FAILED - Model not found',
                'predictions': 0
            })
            continue
        
        # Generate predictions
        try:
            count, output_dir = generate_predictions(model, model_key, input_size, validation_images)
            successful += 1
            results_summary.append({
                'model': model_key,
                'description': MODEL_CONFIGS[model_key]['description'],
                'category': MODEL_CONFIGS[model_key]['category'],
                'status': 'SUCCESS',
                'predictions': count,
                'output_dir': str(output_dir)
            })
        except Exception as e:
            print(f"❌ Failed to generate predictions: {e}")
            # Try to clear memory and continue with next model
            try:
                del model
                tf.keras.backend.clear_session()
                import gc
                gc.collect()
            except:
                pass
            failed += 1
            results_summary.append({
                'model': model_key,
                'description': MODEL_CONFIGS[model_key]['description'],
                'category': MODEL_CONFIGS[model_key]['category'],
                'status': f'FAILED - {str(e)}',
                'predictions': 0
            })
            continue  # Continue with next model instead of stopping
        
        # Clear memory
        del model
        tf.keras.backend.clear_session()
    
    return results_summary, successful, failed

def main():
    """Main execution."""
    print("="*70)
    print("PREDICTION GENERATOR FOR oF1 COMPUTATION (ALL ABLATION MODELS)")
    print("="*70)
    print("\nGenerating predictions for ALL models per project guidelines.\n")
    
    total_start = time.time()
    
    # Process all models
    results, successful, failed = process_all_models()
    
    total_elapsed = time.time() - total_start
    
    # Print summary
    print("\n" + "="*70)
    print("GENERATION SUMMARY")
    print("="*70)
    print(f"Total models: {successful + failed}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total time: {total_elapsed/60:.1f} minutes")
    
    if successful > 0:
        print("\nResults by category:")
        categories = {}
        for r in results:
            cat = r['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(r)
        
        for category, models in categories.items():
            print(f"\n{category}:")
            for m in models:
                status_icon = "✓" if m['status'] == 'SUCCESS' else "✗"
                print(f"  {status_icon} {m['description']}: {m['predictions']} predictions")
        
        # Save summary to JSON
        summary_file = BASE_OUTPUT_DIR / "prediction_generation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump({
                'total_models': successful + failed,
                'successful': successful,
                'failed': failed,
                'total_time_minutes': total_elapsed / 60,
                'results': results
            }, f, indent=2)
        
        print(f"\n✓ Summary saved to: {summary_file}")
        print("\n" + "="*70)
        print("NEXT: Run compute_oF1_all_models.py")
        print("="*70)

if __name__ == "__main__":
    main()
