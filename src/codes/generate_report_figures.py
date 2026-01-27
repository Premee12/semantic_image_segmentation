"""
Generate all figures for the scientific image forgery detection report
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9

# Create output directory
output_dir = Path("Report/figures")
output_dir.mkdir(exist_ok=True, parents=True)

# ============================================================================
# 1. U-Net Ablation Study - Bar Chart Comparison
# ============================================================================
print("Generating U-Net ablation comparison...")
unet_data = pd.read_csv("data/unet_ablation_results.csv")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Dice Score
axes[0].barh(unet_data['Experiment'], unet_data['Val Dice'], color='steelblue')
axes[0].set_xlabel('Validation Dice Score')
axes[0].set_title('U-Net Ablation: Dice Score')
axes[0].grid(axis='x', alpha=0.3)

# IoU Score
axes[1].barh(unet_data['Experiment'], unet_data['Val IoU'], color='coral')
axes[1].set_xlabel('Validation IoU Score')
axes[1].set_title('U-Net Ablation: IoU Score')
axes[1].grid(axis='x', alpha=0.3)

# Loss
axes[2].barh(unet_data['Experiment'], unet_data['Val Loss'], color='lightgreen')
axes[2].set_xlabel('Validation Loss')
axes[2].set_title('U-Net Ablation: Validation Loss')
axes[2].invert_xaxis()  # Lower is better for loss
axes[2].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "unet_ablation_comparison.png", bbox_inches='tight')
plt.savefig(output_dir / "unet_ablation_comparison.pdf", bbox_inches='tight')
print(f"Saved: {output_dir / 'unet_ablation_comparison.png'}")
plt.close()

# ============================================================================
# 2. Training Curves - U-Net ResNet50 BCE+Dice (Best Performer)
# ============================================================================
print("Generating U-Net ResNet50 training curves...")
resnet_log = pd.read_csv("data/unet_resnet50_bce_dice_log.csv")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Dice Score
axes[0, 0].plot(resnet_log['epoch'], resnet_log['dice_coef'], label='Train', linewidth=2)
axes[0, 0].plot(resnet_log['epoch'], resnet_log['val_dice_coef'], label='Validation', linewidth=2)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Dice Coefficient')
axes[0, 0].set_title('U-Net ResNet50: Dice Score')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# IoU Score
axes[0, 1].plot(resnet_log['epoch'], resnet_log['iou_coef'], label='Train', linewidth=2)
axes[0, 1].plot(resnet_log['epoch'], resnet_log['val_iou_coef'], label='Validation', linewidth=2)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('IoU Coefficient')
axes[0, 1].set_title('U-Net ResNet50: IoU Score')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Loss
axes[1, 0].plot(resnet_log['epoch'], resnet_log['loss'], label='Train', linewidth=2)
axes[1, 0].plot(resnet_log['epoch'], resnet_log['val_loss'], label='Validation', linewidth=2)
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Loss')
axes[1, 0].set_title('U-Net ResNet50: Training Loss')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Learning Rate
axes[1, 1].plot(resnet_log['epoch'], resnet_log['learning_rate'], linewidth=2, color='purple')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Learning Rate')
axes[1, 1].set_title('U-Net ResNet50: Learning Rate Schedule')
axes[1, 1].set_yscale('log')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "unet_resnet50_training_curves.png", bbox_inches='tight')
plt.savefig(output_dir / "unet_resnet50_training_curves.pdf", bbox_inches='tight')
print(f"Saved: {output_dir / 'unet_resnet50_training_curves.png'}")
plt.close()

# ============================================================================
# 3. DeepLabV3+ Backbone Comparison
# ============================================================================
print("Generating DeepLabV3+ backbone comparison...")
with open("data/ablation_backbone_results.json", "r") as f:
    backbone_data = json.load(f)

backbones = list(backbone_data.keys())
metrics = ['val_dice', 'val_iou', 'val_loss']
metric_names = ['Dice Score', 'IoU Score', 'Validation Loss']

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
    values = [backbone_data[b][metric] for b in backbones]
    axes[idx].bar(backbones, values, color=['steelblue', 'coral'])
    axes[idx].set_ylabel(name)
    axes[idx].set_title(f'DeepLabV3+ Backbone: {name}')
    axes[idx].grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(values):
        axes[idx].text(i, v + (0.01 if idx < 2 else -0.02), f'{v:.4f}', 
                      ha='center', va='bottom' if idx < 2 else 'top', fontsize=9)

    if metric == 'val_loss':
        axes[idx].invert_yaxis()

plt.tight_layout()
plt.savefig(output_dir / "deeplabv3_backbone_comparison.png", bbox_inches='tight')
plt.savefig(output_dir / "deeplabv3_backbone_comparison.pdf", bbox_inches='tight')
print(f"Saved: {output_dir / 'deeplabv3_backbone_comparison.png'}")
plt.close()

# ============================================================================
# 4. Training Data Size Ablation
# ============================================================================
print("Generating training size ablation...")
with open("data/ablation_training_size_results.json", "r") as f:
    size_data = json.load(f)

sizes = ['25%', '50%', '75%', '100%']
fractions = [size_data[s]['fraction'] for s in sizes]
dice_scores = [size_data[s]['val_dice'] for s in sizes]
iou_scores = [size_data[s]['val_iou'] for s in sizes]
losses = [size_data[s]['val_loss'] for s in sizes]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Dice vs Training Size
axes[0].plot(fractions, dice_scores, marker='o', linewidth=2, markersize=8, color='steelblue')
axes[0].set_xlabel('Training Data Fraction')
axes[0].set_ylabel('Validation Dice Score')
axes[0].set_title('DeepLabV3+: Dice vs Training Size')
axes[0].grid(alpha=0.3)
axes[0].set_xticks(fractions)
axes[0].set_xticklabels(sizes)

# IoU vs Training Size
axes[1].plot(fractions, iou_scores, marker='s', linewidth=2, markersize=8, color='coral')
axes[1].set_xlabel('Training Data Fraction')
axes[1].set_ylabel('Validation IoU Score')
axes[1].set_title('DeepLabV3+: IoU vs Training Size')
axes[1].grid(alpha=0.3)
axes[1].set_xticks(fractions)
axes[1].set_xticklabels(sizes)

# Loss vs Training Size
axes[2].plot(fractions, losses, marker='^', linewidth=2, markersize=8, color='lightgreen')
axes[2].set_xlabel('Training Data Fraction')
axes[2].set_ylabel('Validation Loss')
axes[2].set_title('DeepLabV3+: Loss vs Training Size')
axes[2].grid(alpha=0.3)
axes[2].set_xticks(fractions)
axes[2].set_xticklabels(sizes)

plt.tight_layout()
plt.savefig(output_dir / "deeplabv3_training_size_ablation.png", bbox_inches='tight')
plt.savefig(output_dir / "deeplabv3_training_size_ablation.pdf", bbox_inches='tight')
print(f"Saved: {output_dir / 'deeplabv3_training_size_ablation.png'}")
plt.close()

# ============================================================================
# 5. Threshold Ablation
# ============================================================================
print("Generating threshold ablation...")
with open("data/ablation_threshold_results.json", "r") as f:
    threshold_data = json.load(f)

thresholds = sorted([float(k) for k in threshold_data.keys()])
dice_thresh = [threshold_data[str(t)]['dice'] for t in thresholds]
iou_thresh = [threshold_data[str(t)]['iou'] for t in thresholds]

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

ax.plot(thresholds, dice_thresh, marker='o', linewidth=2, markersize=8, label='Dice Score', color='steelblue')
ax.plot(thresholds, iou_thresh, marker='s', linewidth=2, markersize=8, label='IoU Score', color='coral')
ax.set_xlabel('Binarization Threshold')
ax.set_ylabel('Score')
ax.set_title('DeepLabV3+: Performance vs Binarization Threshold')
ax.legend()
ax.grid(alpha=0.3)

# Mark optimal threshold
optimal_idx = np.argmax(dice_thresh)
optimal_thresh = thresholds[optimal_idx]
ax.axvline(optimal_thresh, color='red', linestyle='--', alpha=0.5, label=f'Optimal: {optimal_thresh}')
ax.legend()

plt.tight_layout()
plt.savefig(output_dir / "deeplabv3_threshold_ablation.png", bbox_inches='tight')
plt.savefig(output_dir / "deeplabv3_threshold_ablation.pdf", bbox_inches='tight')
print(f"Saved: {output_dir / 'deeplabv3_threshold_ablation.png'}")
plt.close()

# ============================================================================
# 6. Overall Model Comparison
# ============================================================================
print("Generating overall model comparison...")

# Prepare data for comparison
comparison_data = {
    'Model': [
        'U-Net (Scratch) BCE',
        'U-Net (Scratch) Dice',
        'U-Net (Scratch) BCE+Dice',
        'U-Net ResNet50 BCE+Dice',
        'DeepLabV3+ ResNet50',
        'DeepLabV3+ ResNet101'
    ],
    'Dice': [
        0.0538,  # U-Net BCE
        0.4638,  # U-Net Dice
        0.1049,  # U-Net BCE+Dice
        0.2493,  # U-Net ResNet50
        0.4458,  # DeepLabV3+ ResNet50
        0.4647   # DeepLabV3+ ResNet101
    ],
    'IoU': [
        0.4638,  # U-Net BCE
        0.4638,  # U-Net Dice
        0.2004,  # U-Net BCE+Dice
        0.3053,  # U-Net ResNet50
        0.3209,  # DeepLabV3+ ResNet50
        0.3313   # DeepLabV3+ ResNet101
    ]
}

df_comparison = pd.DataFrame(comparison_data)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Dice comparison
axes[0].barh(df_comparison['Model'], df_comparison['Dice'], color='steelblue')
axes[0].set_xlabel('Validation Dice Score')
axes[0].set_title('Model Comparison: Dice Score')
axes[0].grid(axis='x', alpha=0.3)

# IoU comparison
axes[1].barh(df_comparison['Model'], df_comparison['IoU'], color='coral')
axes[1].set_xlabel('Validation IoU Score')
axes[1].set_title('Model Comparison: IoU Score')
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "overall_model_comparison.png", bbox_inches='tight')
plt.savefig(output_dir / "overall_model_comparison.pdf", bbox_inches='tight')
print(f"Saved: {output_dir / 'overall_model_comparison.png'}")
plt.close()

# ============================================================================
# 7. Performance Summary Table (as image for easy insertion)
# ============================================================================
print("Generating performance summary table...")

fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('tight')
ax.axis('off')

# Create table data
table_data = [
    ['Model', 'Backbone', 'Loss Function', 'Val Dice ↑', 'Val IoU ↑', 'Val Loss ↓'],
    ['U-Net (Scratch)', '-', 'BCE', '0.0538', '0.4638', '0.1112'],
    ['U-Net (Scratch)', '-', 'Dice', '0.4638', '0.4638', '0.5362'],
    ['U-Net (Scratch)', '-', 'BCE+Dice', '0.1049', '0.2004', '1.0279'],
    ['U-Net', 'ResNet50', 'BCE+Dice', '0.2493', '0.3053', '0.9324'],
    ['DeepLabV3+', 'ResNet50', 'BCE+Dice', '0.4458', '0.3209', '0.6796'],
    ['DeepLabV3+', 'ResNet101', 'BCE+Dice', '0.4647', '0.3313', '0.6649'],
]

table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.18, 0.15, 0.15, 0.15, 0.15, 0.15])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Style header row
for i in range(6):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Highlight best scores
for i in range(1, 7):
    for j in range(6):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#F2F2F2')

plt.savefig(output_dir / "performance_summary_table.png", bbox_inches='tight', dpi=300)
print(f"Saved: {output_dir / 'performance_summary_table.png'}")
plt.close()

print("\n" + "="*60)
print("All figures generated successfully!")
print("="*60)
print(f"\nFigures saved in: {output_dir.absolute()}")
print("\nGenerated files:")
for f in sorted(output_dir.glob("*.png")):
    print(f"  - {f.name}")
