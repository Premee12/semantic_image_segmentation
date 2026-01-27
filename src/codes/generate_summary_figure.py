"""
Create a comprehensive single-page summary figure for presentations
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 9

# Create output directory
output_dir = Path("Report/figures")
output_dir.mkdir(exist_ok=True, parents=True)

# Create a comprehensive summary figure
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# ============================================================================
# 1. Overall Model Comparison (Top Left)
# ============================================================================
ax1 = fig.add_subplot(gs[0, :2])
models = ['U-Net\n(BCE)', 'U-Net\n(Dice)', 'U-Net\n(BCE+Dice)', 
          'U-Net\nResNet50', 'DeepLabV3+\nResNet50', 'DeepLabV3+\nResNet101']
dice_scores = [0.0538, 0.4638, 0.1049, 0.2493, 0.4458, 0.4647]
colors = ['#ff6b6b', '#51cf66', '#ff8787', '#4dabf7', '#ffd43b', '#74c0fc']

bars = ax1.barh(models, dice_scores, color=colors, edgecolor='black', linewidth=1.5)
ax1.set_xlabel('Validation Dice Score', fontsize=11, fontweight='bold')
ax1.set_title('Overall Model Performance Comparison', fontsize=13, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, dice_scores)):
    ax1.text(val + 0.01, i, f'{val:.4f}', va='center', fontweight='bold')

# Highlight best
ax1.axvline(max(dice_scores), color='green', linestyle='--', alpha=0.5, linewidth=2)

# ============================================================================
# 2. Key Finding Highlight (Top Right)
# ============================================================================
ax2 = fig.add_subplot(gs[0, 2])
ax2.axis('off')

# Create text box with key findings
findings_text = """
KEY FINDINGS

✓ Best Model:
  DeepLabV3+ ResNet-101
  Dice: 0.4647
  IoU: 0.3313

✓ Loss Function Impact:
  Dice vs BCE: 8.6× better

✓ Optimal Threshold: 0.6

✓ Data Efficiency:
  50% data → 87.5% performance

✓ Architecture Depth:
  ResNet-101 vs 50: +4.2%
"""

ax2.text(0.1, 0.5, findings_text, transform=ax2.transAxes, 
         fontsize=10, verticalalignment='center',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
         family='monospace', fontweight='bold')

# ============================================================================
# 3. Loss Function Comparison (Middle Left)
# ============================================================================
ax3 = fig.add_subplot(gs[1, 0])
loss_funcs = ['BCE', 'Dice', 'BCE+Dice']
loss_dice = [0.0538, 0.4638, 0.1049]
loss_colors = ['#ff6b6b', '#51cf66', '#ff8787']

ax3.bar(loss_funcs, loss_dice, color=loss_colors, edgecolor='black', linewidth=2)
ax3.set_ylabel('Validation Dice Score', fontweight='bold')
ax3.set_title('U-Net: Loss Function Impact', fontsize=11, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)

for i, v in enumerate(loss_dice):
    ax3.text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')

# ============================================================================
# 4. Backbone Comparison (Middle Center)
# ============================================================================
ax4 = fig.add_subplot(gs[1, 1])
backbones = ['ResNet-50', 'ResNet-101']
backbone_dice = [0.4458, 0.4647]
backbone_colors = ['#ffd43b', '#74c0fc']

ax4.bar(backbones, backbone_dice, color=backbone_colors, edgecolor='black', linewidth=2)
ax4.set_ylabel('Validation Dice Score', fontweight='bold')
ax4.set_title('DeepLabV3+: Backbone Depth', fontsize=11, fontweight='bold')
ax4.set_ylim([0.42, 0.48])
ax4.grid(axis='y', alpha=0.3)

for i, v in enumerate(backbone_dice):
    ax4.text(i, v + 0.002, f'{v:.4f}', ha='center', fontweight='bold')

# Add improvement annotation
improvement = ((backbone_dice[1] - backbone_dice[0]) / backbone_dice[0]) * 100
ax4.annotate(f'+{improvement:.1f}%', xy=(0.5, 0.465), fontsize=10, 
             ha='center', fontweight='bold', color='green')

# ============================================================================
# 5. Training Data Size Efficiency (Middle Right)
# ============================================================================
ax5 = fig.add_subplot(gs[1, 2])
with open("data/ablation_training_size_results.json", "r") as f:
    size_data = json.load(f)

sizes = ['25%', '50%', '75%', '100%']
fractions = [0.25, 0.5, 0.75, 1.0]
dice_scores_size = [size_data[s]['val_dice'] for s in sizes]

ax5.plot(fractions, dice_scores_size, marker='o', linewidth=3, 
         markersize=10, color='#4dabf7')
ax5.fill_between(fractions, dice_scores_size, alpha=0.3, color='#4dabf7')
ax5.set_xlabel('Training Data Fraction', fontweight='bold')
ax5.set_ylabel('Validation Dice', fontweight='bold')
ax5.set_title('Data Efficiency Curve', fontsize=11, fontweight='bold')
ax5.set_xticks(fractions)
ax5.set_xticklabels(sizes)
ax5.grid(alpha=0.3)

# Annotate 50% performance
ax5.annotate('87.5% of\nfull performance', 
             xy=(0.5, dice_scores_size[1]), 
             xytext=(0.3, 0.39),
             arrowprops=dict(arrowstyle='->', color='red', lw=2),
             fontsize=9, fontweight='bold', color='red')

# ============================================================================
# 6. Threshold Optimization (Bottom Left)
# ============================================================================
ax6 = fig.add_subplot(gs[2, 0])
with open("data/ablation_threshold_results.json", "r") as f:
    threshold_data = json.load(f)

thresholds = sorted([float(k) for k in threshold_data.keys()])
dice_thresh = [threshold_data[str(t)]['dice'] for t in thresholds]

ax6.plot(thresholds, dice_thresh, marker='s', linewidth=3, 
         markersize=8, color='#ff6b6b', label='Dice Score')
ax6.set_xlabel('Binarization Threshold', fontweight='bold')
ax6.set_ylabel('Dice Score', fontweight='bold')
ax6.set_title('Optimal Threshold = 0.6', fontsize=11, fontweight='bold')
ax6.grid(alpha=0.3)

# Mark optimal
optimal_idx = np.argmax(dice_thresh)
optimal_thresh = thresholds[optimal_idx]
ax6.axvline(optimal_thresh, color='green', linestyle='--', 
            alpha=0.7, linewidth=2, label=f'Optimal: {optimal_thresh}')
ax6.scatter([optimal_thresh], [dice_thresh[optimal_idx]], 
           s=200, color='green', zorder=5, marker='*')
ax6.legend()

# ============================================================================
# 7. Performance Metrics Table (Bottom Center & Right)
# ============================================================================
ax7 = fig.add_subplot(gs[2, 1:])
ax7.axis('tight')
ax7.axis('off')

# Create performance table
table_data = [
    ['Model', 'Dice ↑', 'IoU ↑', 'Loss ↓', 'Rank'],
    ['DeepLabV3+ ResNet-101', '0.4647', '0.3313', '0.6649', '🥇'],
    ['U-Net Dice Loss', '0.4638', '0.4638', '0.5362', '🥈'],
    ['DeepLabV3+ ResNet-50', '0.4458', '0.3209', '0.6796', '🥉'],
    ['U-Net ResNet-50', '0.2493', '0.3053', '0.9324', '4'],
    ['U-Net BCE+Dice', '0.1049', '0.2004', '1.0279', '5'],
    ['U-Net BCE', '0.0538', '0.4638', '0.1112', '6'],
]

table = ax7.table(cellText=table_data, cellLoc='center', loc='center',
                 colWidths=[0.35, 0.15, 0.15, 0.15, 0.10])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Style header row
for i in range(5):
    cell = table[(0, i)]
    cell.set_facecolor('#2c3e50')
    cell.set_text_props(weight='bold', color='white', fontsize=11)

# Color code rows
colors_rows = ['#74c0fc', '#51cf66', '#ffd43b', '#ffa94d', '#ff8787', '#ff6b6b']
for i in range(1, 7):
    for j in range(5):
        table[(i, j)].set_facecolor(colors_rows[i-1])
        table[(i, j)].set_alpha(0.3)

ax7.set_title('Final Performance Rankings', fontsize=12, fontweight='bold', pad=20)

# ============================================================================
# Main Title
# ============================================================================
fig.suptitle('Scientific Image Forgery Detection: Comprehensive Results Summary', 
             fontsize=16, fontweight='bold', y=0.98)

plt.savefig(output_dir / "comprehensive_summary.png", bbox_inches='tight', dpi=300)
plt.savefig(output_dir / "comprehensive_summary.pdf", bbox_inches='tight')
print(f"✓ Saved comprehensive summary figure: {output_dir / 'comprehensive_summary.png'}")
plt.close()

print("\n" + "="*60)
print("Comprehensive summary figure created successfully!")
print("="*60)
print(f"\nThis single figure summarizes all key findings and can be")
print(f"used for presentations, posters, or quick reference.")
