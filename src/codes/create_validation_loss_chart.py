"""
Create validation loss chart for U-Net Forged Tight Crop
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 9

# Create output directory
output_dir = Path("Report/figures")
output_dir.mkdir(exist_ok=True, parents=True)

# Read data
tight_log = pd.read_csv("data/unet_scratch_region_tight_log.csv")

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

# Validation Loss
axes[0].plot(tight_log['epoch'], tight_log['loss'], label='Train', linewidth=2, alpha=0.8)
axes[0].plot(tight_log['epoch'], tight_log['val_loss'], label='Validation', linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('U-Net Forged Tight Crop: Training Loss')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Validation Dice
axes[1].plot(tight_log['epoch'], tight_log['dice_coef'], label='Train', linewidth=2, alpha=0.8)
axes[1].plot(tight_log['epoch'], tight_log['val_dice_coef'], label='Validation', linewidth=2)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Dice Coefficient')
axes[1].set_title('U-Net Forged Tight Crop: Dice Score')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "unet_tight_crop_training.png", bbox_inches='tight', dpi=300)
plt.savefig(output_dir / "unet_tight_crop_training.pdf", bbox_inches='tight')
print(f"✓ Saved: {output_dir / 'unet_tight_crop_training.png'}")
plt.close()

print("\nValidation loss chart created successfully!")
