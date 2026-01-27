"""
Compute detailed confusion matrix metrics for all models.
This script calculates TP, FP, TN, FN, Precision, Recall, Specificity, and F1 scores
at the pixel level for all trained models.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import cv2
from tqdm import tqdm
import json

def compute_confusion_matrix(y_true, y_pred, threshold=0.5):
    """
    Compute confusion matrix metrics at pixel level.
    
    Args:
        y_true: Ground truth binary mask (0 or 255)
        y_pred: Predicted probability mask (0-1 or 0-255)
        threshold: Threshold for binarization
    
    Returns:
        Dictionary with TP, FP, TN, FN counts
    """
    # Normalize inputs
    if y_true.max() > 1:
        y_true = (y_true > 127).astype(np.uint8)
    if y_pred.max() > 1:
        y_pred = (y_pred / 255.0)
    
    # Binarize prediction
    y_pred_binary = (y_pred >= threshold).astype(np.uint8)
    
    # Compute confusion matrix
    TP = np.sum((y_true == 1) & (y_pred_binary == 1))
    FP = np.sum((y_true == 0) & (y_pred_binary == 1))
    TN = np.sum((y_true == 0) & (y_pred_binary == 0))
    FN = np.sum((y_true == 1) & (y_pred_binary == 0))
    
    return {'TP': int(TP), 'FP': int(FP), 'TN': int(TN), 'FN': int(FN)}


def compute_metrics(confusion):
    """
    Compute classification metrics from confusion matrix.
    
    Args:
        confusion: Dictionary with TP, FP, TN, FN
    
    Returns:
        Dictionary with all metrics
    """
    TP, FP, TN, FN = confusion['TP'], confusion['FP'], confusion['TN'], confusion['FN']
    
    # Avoid division by zero
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0
    
    # IoU and Dice
    iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0
    dice = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0.0
    
    return {
        'Precision': precision,
        'Recall': recall,
        'Specificity': specificity,
        'F1_Score': f1_score,
        'Accuracy': accuracy,
        'IoU': iou,
        'Dice': dice,
        **confusion
    }


def process_model_predictions(model_name, predictions_dir, ground_truth_dir, threshold=0.5):
    """
    Process all predictions for a model and compute aggregate metrics.
    
    Args:
        model_name: Name of the model
        predictions_dir: Directory containing prediction masks
        ground_truth_dir: Directory containing ground truth masks
        threshold: Threshold for binarization
    
    Returns:
        Dictionary with aggregate metrics
    """
    pred_path = Path(predictions_dir) / model_name
    gt_path = Path(ground_truth_dir)
    
    if not pred_path.exists():
        print(f"Warning: Predictions not found for {model_name}")
        return None
    
    # Aggregate confusion matrix
    total_confusion = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}
    num_images = 0
    
    # Process each prediction - check for both .npy and .png files
    pred_files = sorted(list(pred_path.glob('*.npy')) + list(pred_path.glob('*.png')))
    
    for pred_file in tqdm(pred_files, desc=f"Processing {model_name}"):
        # Load prediction
        if pred_file.suffix == '.npy':
            pred = np.load(str(pred_file))
            if pred.ndim == 3:
                pred = pred[:, :, 0]  # Take first channel if multi-channel
            pred = (pred * 255).astype(np.uint8)  # Convert to 0-255 range
        else:
            pred = cv2.imread(str(pred_file), cv2.IMREAD_GRAYSCALE)
        
        if pred is None:
            continue
        
        # Find corresponding ground truth (try both .npy and .png)
        gt_file = gt_path / pred_file.with_suffix('.png').name
        if not gt_file.exists():
            gt_file = gt_path / pred_file.with_suffix('.npy').name
        if not gt_file.exists():
            continue
        
        if gt_file.suffix == '.npy':
            gt = np.load(str(gt_file))
            if gt.ndim == 3:
                gt = gt[:, :, 0]
            gt = (gt * 255).astype(np.uint8)
        else:
            gt = cv2.imread(str(gt_file), cv2.IMREAD_GRAYSCALE)
        
        if gt is None:
            continue
        
        # Ensure same size
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))
        
        # Compute confusion matrix for this image
        confusion = compute_confusion_matrix(gt, pred, threshold)
        
        # Aggregate
        for key in total_confusion:
            total_confusion[key] += confusion[key]
        
        num_images += 1
    
    if num_images == 0:
        return None
    
    # Compute metrics from aggregate confusion matrix
    metrics = compute_metrics(total_confusion)
    metrics['Model'] = model_name
    metrics['Num_Images'] = num_images
    metrics['Threshold'] = threshold
    
    return metrics


def main():
    """Main function to compute confusion matrix metrics for all models."""
    
    # Setup paths
    base_dir = Path(__file__).parent.parent.parent
    
    # Check multiple possible locations for predictions
    if (base_dir / 'predictions').exists():
        predictions_base = base_dir / 'predictions'
    elif (base_dir.parent / 'predictions').exists():
        predictions_base = base_dir.parent / 'predictions'
    else:
        predictions_base = base_dir / 'predictions'
    
    # Check multiple possible locations for ground truth
    possible_gt_paths = [
        base_dir / 'ground_truth',
        base_dir.parent / 'ground_truth',
        base_dir.parent / 'recodai-luc-scientific-image-forgery-detection' / 'train_masks',
        base_dir / 'recodai-luc-scientific-image-forgery-detection' / 'train_masks'
    ]
    
    ground_truth_dir = None
    for gt_path in possible_gt_paths:
        if gt_path.exists() and list(gt_path.iterdir()):
            ground_truth_dir = gt_path
            break
    
    if ground_truth_dir is None:
        ground_truth_dir = base_dir / 'ground_truth'
    
    results_dir = base_dir / 'results'
    results_dir.mkdir(exist_ok=True)
    
    # Check if directories exist
    if not predictions_base.exists():
        print(f"Error: Predictions directory not found: {predictions_base}")
        print("Please run generate_predictions_for_oF1.py first")
        return
    
    if not ground_truth_dir.exists():
        print(f"Error: Ground truth directory not found: {ground_truth_dir}")
        print("Please ensure ground truth masks are in the correct location")
        return
    
    # List of models to evaluate
    model_dirs = [d.name for d in predictions_base.iterdir() if d.is_dir()]
    
    if not model_dirs:
        print("No model predictions found")
        return
    
    print(f"Found {len(model_dirs)} models to evaluate")
    print("Computing confusion matrix metrics...\n")
    
    # Process each model
    all_metrics = []
    thresholds = [0.3, 0.5, 0.7]  # Test multiple thresholds
    
    for model_name in sorted(model_dirs):
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print('='*60)
        
        for threshold in thresholds:
            metrics = process_model_predictions(
                model_name, 
                predictions_base, 
                ground_truth_dir, 
                threshold=threshold
            )
            
            if metrics:
                all_metrics.append(metrics)
                
                # Print summary
                print(f"\nThreshold: {threshold}")
                print(f"  TP: {metrics['TP']:,} | FP: {metrics['FP']:,}")
                print(f"  TN: {metrics['TN']:,} | FN: {metrics['FN']:,}")
                print(f"  Precision: {metrics['Precision']:.4f}")
                print(f"  Recall: {metrics['Recall']:.4f}")
                print(f"  Specificity: {metrics['Specificity']:.4f}")
                print(f"  F1 Score: {metrics['F1_Score']:.4f}")
                print(f"  IoU: {metrics['IoU']:.4f}")
                print(f"  Dice: {metrics['Dice']:.4f}")
    
    # Save results
    if all_metrics:
        # Convert to DataFrame
        df = pd.DataFrame(all_metrics)
        
        # Reorder columns
        column_order = [
            'Model', 'Threshold', 'Num_Images',
            'TP', 'FP', 'TN', 'FN',
            'Precision', 'Recall', 'Specificity', 'F1_Score',
            'Accuracy', 'IoU', 'Dice'
        ]
        df = df[column_order]
        
        # Save to CSV
        output_csv = results_dir / 'confusion_matrix_metrics.csv'
        df.to_csv(output_csv, index=False)
        print(f"\n{'='*60}")
        print(f"Results saved to: {output_csv}")
        
        # Save summary JSON
        summary = {
            'total_models': len(model_dirs),
            'thresholds_tested': thresholds,
            'metrics_computed': list(column_order),
            'output_file': str(output_csv)
        }
        
        summary_json = results_dir / 'confusion_matrix_summary.json'
        with open(summary_json, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Summary saved to: {summary_json}")
        
        # Print best models
        print(f"\n{'='*60}")
        print("Top 5 Models by F1 Score (threshold=0.5):")
        print('='*60)
        df_05 = df[df['Threshold'] == 0.5].sort_values('F1_Score', ascending=False)
        for idx, row in df_05.head(5).iterrows():
            print(f"{row['Model']:40s} | F1: {row['F1_Score']:.4f} | Prec: {row['Precision']:.4f} | Rec: {row['Recall']:.4f}")
        
        print(f"\nTop 5 Models by IoU (threshold=0.5):")
        print('='*60)
        df_05_iou = df[df['Threshold'] == 0.5].sort_values('IoU', ascending=False)
        for idx, row in df_05_iou.head(5).iterrows():
            print(f"{row['Model']:40s} | IoU: {row['IoU']:.4f} | Dice: {row['Dice']:.4f}")
    else:
        print("\nNo metrics computed. Please check if predictions and ground truth are available.")


if __name__ == '__main__':
    main()
