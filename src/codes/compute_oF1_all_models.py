"""
Compute oF1 scores for ALL models from ablation studies.

This script automates oF1 computation for all model predictions
and generates a comprehensive comparison table.
"""

import sys
import os

# Install missing packages if needed
try:
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from scipy.optimize import linear_sum_assignment
    from scipy.ndimage import label as connected_components
    import matplotlib.pyplot as plt
    import seaborn as sns
    from tqdm import tqdm
    import gc
    import json
    import time
except ImportError as e:
    print(f"❌ Missing package: {e}")
    print("Installing required packages...")
    os.system("pip install -q numpy pandas scipy matplotlib seaborn tqdm")
    print("Please run the script again after installation.")
    sys.exit(1)

# Configuration
DATA_ROOT = Path(__file__).parent / "recodai-luc-scientific-image-forgery-detection"
PRED_BASE_DIR = Path(__file__).parent / "predictions"
GT_DIR = DATA_ROOT / "train_masks"
OUTPUT_DIR = Path(__file__).parent / "oF1_results"
OUTPUT_DIR.mkdir(exist_ok=True)

CONFIG = {
    'iou_thresholds': np.arange(0.5, 1.0, 0.05),
    'batch_size': 10,
    'min_instance_size': 10,
}

# Model categories (same as in generate_predictions_for_oF1.py)
MODEL_INFO = {
    'unet_scratch_bce': {'description': 'U-Net (scratch) + BCE', 'category': 'U-Net Loss'},
    'unet_scratch_dice': {'description': 'U-Net (scratch) + Dice', 'category': 'U-Net Loss'},
    'unet_scratch_bce_dice': {'description': 'U-Net (scratch) + BCE+Dice', 'category': 'U-Net Loss'},
    'unet_resnet50_bce_dice': {'description': 'U-Net (ResNet50)', 'category': 'U-Net Pretraining'},
    'unet_scratch_region_full': {'description': 'U-Net + Full', 'category': 'U-Net Augmentation'},
    'unet_scratch_region_context': {'description': 'U-Net + Context', 'category': 'U-Net Augmentation'},
    'unet_scratch_region_tight': {'description': 'U-Net + Tight', 'category': 'U-Net Augmentation'},
    'deeplabv3_resnet50': {'description': 'DeepLabV3+ ResNet-50', 'category': 'DeepLabV3+ Backbone'},
    'deeplabv3_resnet101': {'description': 'DeepLabV3+ ResNet-101', 'category': 'DeepLabV3+ Backbone'},
    'deeplabv3_data25': {'description': 'DeepLabV3+ (25%)', 'category': 'DeepLabV3+ Data'},
    'deeplabv3_data50': {'description': 'DeepLabV3+ (50%)', 'category': 'DeepLabV3+ Data'},
    'deeplabv3_data75': {'description': 'DeepLabV3+ (75%)', 'category': 'DeepLabV3+ Data'},
    'deeplabv3_data100': {'description': 'DeepLabV3+ (100%)', 'category': 'DeepLabV3+ Data'},
}


def compute_iou(mask1, mask2):
    """Compute IoU between two binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0.0


def extract_instances(binary_mask, min_size=10):
    """Extract individual instances using connected components."""
    labeled_mask, num_instances = connected_components(binary_mask)
    instances = []
    for i in range(1, num_instances + 1):
        instance_mask = (labeled_mask == i)
        if instance_mask.sum() >= min_size:
            instances.append(instance_mask)
    return instances, len(instances)


def hungarian_matching(pred_instances, gt_instances, iou_threshold=0.5):
    """Match instances using Hungarian algorithm."""
    n_pred, n_gt = len(pred_instances), len(gt_instances)
    
    if n_pred == 0 and n_gt == 0:
        return 0, 0, 0
    if n_pred == 0:
        return 0, 0, n_gt
    if n_gt == 0:
        return 0, n_pred, 0
    
    # Compute IoU matrix
    iou_matrix = np.zeros((n_pred, n_gt))
    for i, pred_mask in enumerate(pred_instances):
        for j, gt_mask in enumerate(gt_instances):
            iou_matrix[i, j] = compute_iou(pred_mask, gt_mask)
    
    # Hungarian matching
    cost_matrix = 1 - iou_matrix
    pred_indices, gt_indices = linear_sum_assignment(cost_matrix)
    
    # Count matches above threshold
    tp = sum(1 for i, j in zip(pred_indices, gt_indices) if iou_matrix[i, j] >= iou_threshold)
    fp = n_pred - tp
    fn = n_gt - tp
    
    return tp, fp, fn


def compute_f1_score(tp, fp, fn):
    """Compute F1 score."""
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0


def compute_oF1_for_model(model_name, pred_dir, gt_dir, iou_thresholds, batch_size=10):
    """Compute oF1 for a single model."""
    pred_files = sorted(list(pred_dir.glob("*.npy")))
    
    if len(pred_files) == 0:
        print(f"  ⚠️  No predictions found for {model_name}")
        return None
    
    # Initialize accumulators
    results_per_threshold = {
        iou_th: {'tp': 0, 'fp': 0, 'fn': 0} 
        for iou_th in iou_thresholds
    }
    
    # Process in batches
    num_batches = (len(pred_files) + batch_size - 1) // batch_size
    processed_count = 0
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(pred_files))
        batch_files = pred_files[start_idx:end_idx]
        
        for pred_file in batch_files:
            gt_file = gt_dir / pred_file.name
            
            if not gt_file.exists():
                continue
            
            # Load masks
            pred_mask = np.load(pred_file)
            gt_mask = np.load(gt_file)
            
            # Handle ground truth with multiple instances (shape: num_instances, H, W)
            # Collapse to single binary mask
            if len(gt_mask.shape) == 3:
                gt_mask = gt_mask.max(axis=0)  # Collapse instances to single mask
            
            # Resize prediction to match ground truth size if different
            if pred_mask.shape != gt_mask.shape:
                from PIL import Image
                pred_pil = Image.fromarray((pred_mask * 255).astype(np.uint8))
                pred_pil = pred_pil.resize((gt_mask.shape[1], gt_mask.shape[0]), Image.NEAREST)
                pred_mask = np.array(pred_pil) / 255.0
            
            # Binarize
            pred_binary = (pred_mask > 0.5).astype(np.uint8)
            gt_binary = (gt_mask > 0.5).astype(np.uint8)
            
            # Extract instances
            pred_instances, _ = extract_instances(pred_binary, CONFIG['min_instance_size'])
            gt_instances, _ = extract_instances(gt_binary, CONFIG['min_instance_size'])
            
            # Compute metrics for each threshold
            for iou_th in iou_thresholds:
                tp, fp, fn = hungarian_matching(pred_instances, gt_instances, iou_th)
                results_per_threshold[iou_th]['tp'] += tp
                results_per_threshold[iou_th]['fp'] += fp
                results_per_threshold[iou_th]['fn'] += fn
            
            processed_count += 1
        
        gc.collect()
    
    # Compute F1 scores
    f1_scores = []
    for iou_th in iou_thresholds:
        tp = results_per_threshold[iou_th]['tp']
        fp = results_per_threshold[iou_th]['fp']
        fn = results_per_threshold[iou_th]['fn']
        f1 = compute_f1_score(tp, fp, fn)
        f1_scores.append(f1)
    
    # Find optimal F1
    oF1 = max(f1_scores) if f1_scores else 0.0
    optimal_idx = np.argmax(f1_scores) if f1_scores else 0
    optimal_threshold = iou_thresholds[optimal_idx]
    
    return {
        'oF1': oF1,
        'optimal_iou_threshold': optimal_threshold,
        'f1_scores': f1_scores,
        'num_images': processed_count
    }


def main():
    """Main execution."""
    print("="*70)
    print("oF1 COMPUTATION FOR ALL ABLATION MODELS")
    print("="*70)
    
    # Find all model prediction directories
    model_dirs = [d for d in PRED_BASE_DIR.iterdir() if d.is_dir()]
    
    if len(model_dirs) == 0:
        print("❌ No prediction directories found!")
        print(f"   Expected in: {PRED_BASE_DIR}")
        print("   Run generate_predictions_for_oF1.py first")
        return
    
    print(f"\nFound {len(model_dirs)} models with predictions")
    print(f"Computing oF1 with IoU thresholds: {CONFIG['iou_thresholds'][0]:.2f} - {CONFIG['iou_thresholds'][-1]:.2f}")
    
    # Compute oF1 for each model
    all_results = []
    
    for i, pred_dir in enumerate(model_dirs, 1):
        model_name = pred_dir.name
        info = MODEL_INFO.get(model_name, {'description': model_name, 'category': 'Unknown'})
        
        print(f"\n[{i}/{len(model_dirs)}] {info['description']}")
        print(f"  Category: {info['category']}")
        
        start_time = time.time()
        result = compute_oF1_for_model(
            model_name, pred_dir, GT_DIR, 
            CONFIG['iou_thresholds'], CONFIG['batch_size']
        )
        elapsed = time.time() - start_time
        
        if result:
            print(f"  ✓ oF1 = {result['oF1']:.4f} @ IoU={result['optimal_iou_threshold']:.2f}")
            print(f"  ⏱️  {elapsed:.1f}s ({result['num_images']} images)")
            
            all_results.append({
                'model': model_name,
                'description': info['description'],
                'category': info['category'],
                'oF1': result['oF1'],
                'optimal_threshold': result['optimal_iou_threshold'],
                'num_images': result['num_images']
            })
        else:
            print(f"  ❌ Failed to compute oF1 for {model_name}")
            all_results.append({
                'model': model_name,
                'description': info['description'],
                'category': info['category'],
                'oF1': 0.0,
                'optimal_threshold': 0.5,
                'num_images': 0
            })
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('oF1', ascending=False)
    
    # Save results
    csv_file = OUTPUT_DIR / 'all_models_oF1_scores.csv'
    results_df.to_csv(csv_file, index=False)
    
    # Print summary table
    print("\n" + "="*70)
    print("oF1 SCORES - ALL MODELS")
    print("="*70)
    print(results_df.to_string(index=False))
    
    # Create comparison plots
    create_comparison_plots(results_df)
    
    print(f"\n✓ Results saved to: {csv_file}")
    print(f"✓ Plots saved to: {OUTPUT_DIR}")
    
    # Print next steps
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("1. Review oF1_results/all_models_oF1_scores.csv")
    print("2. Add oF1 column to Table 7 in your report")
    print("3. Include oF1 comparison plots if needed")
    print("="*70)


def create_comparison_plots(results_df):
    """Create visualization comparing oF1 across models."""
    
    # Plot 1: oF1 by category
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart
    ax = axes[0]
    colors = plt.cm.Set3(np.linspace(0, 1, len(results_df)))
    bars = ax.barh(range(len(results_df)), results_df['oF1'], color=colors)
    ax.set_yticks(range(len(results_df)))
    ax.set_yticklabels(results_df['description'], fontsize=9)
    ax.set_xlabel('oF1 Score', fontweight='bold')
    ax.set_title('oF1 Scores - All Models', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Annotate values
    for i, (idx, row) in enumerate(results_df.iterrows()):
        ax.text(row['oF1'] + 0.01, i, f"{row['oF1']:.4f}", 
                va='center', fontsize=8)
    
    # Grouped by category
    ax = axes[1]
    categories = results_df['category'].unique()
    cat_data = {cat: results_df[results_df['category'] == cat]['oF1'].values 
                for cat in categories}
    
    positions = []
    labels = []
    colors_cat = []
    for i, (cat, scores) in enumerate(cat_data.items()):
        pos = np.arange(len(scores)) + i * (len(scores) + 0.5)
        positions.extend(pos)
        labels.extend(results_df[results_df['category'] == cat]['description'].values)
        colors_cat.extend([plt.cm.Set2(i)] * len(scores))
        ax.barh(pos, scores, color=plt.cm.Set2(i), label=cat)
    
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('oF1 Score', fontweight='bold')
    ax.set_title('oF1 Scores by Category', fontweight='bold')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'oF1_comparison_all_models.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'oF1_comparison_all_models.pdf', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved comparison plot")
    plt.close()


if __name__ == "__main__":
    main()
