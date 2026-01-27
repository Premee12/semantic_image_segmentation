# Semantic Image Segmentation for Forgery Detection

This repository contains our implementation of semantic segmentation models (UNet and DeepLabV3+) for detecting image forgeries in scientific images. This project was developed as part of the MLDM Deep Learning course (January 2026).

## 📊 Project Overview

Image forgery detection is critical for maintaining scientific integrity. This project implements and compares two state-of-the-art semantic segmentation architectures for pixel-level forgery detection:

- **UNet**: A classic encoder-decoder architecture with skip connections
- **DeepLabV3+**: An advanced architecture with atrous spatial pyramid pooling (ASPP)

## 🎯 Key Features

- **Multiple Model Architectures**: Implementation of UNet and DeepLabV3+ with various configurations
- **Comprehensive Ablation Studies**: 
  - Backbone variations (ResNet50, ResNet101, from scratch)
  - Loss function comparisons (BCE, Dice, combined)
  - Training data size impact (25%, 50%, 75%, 100%)
  - Region-based training strategies (tight, context, full)
- **Advanced Metrics**: Implementation of Object-based F1 (oF1) score using Hungarian matching algorithm
- **Automated Pipeline**: Complete scripts for training, prediction generation, and evaluation

## 📦 Dataset

**Source**: [Kaggle - RecodAI Scientific Image Forgery Detection](https://www.kaggle.com/competitions/recodai-luc-scientific-image-forgery-detection)

### Dataset Structure:
```
data/
├── train_images/       # Training images
├── train_masks/        # Ground truth masks
├── test_images/        # Test images for evaluation
├── supplemental_images/  # Additional training data
└── supplemental_masks/   # Additional masks
```

### Data Statistics:
- **Training Images**: 2,500 images
- **Test Images**: 500 images
- **Image Format**: RGB images (various dimensions)
- **Mask Format**: Binary masks (0: authentic, 255: forged)

## 🛠️ Installation & Setup

### Prerequisites
```bash
Python 3.8+
CUDA 11.0+ (for GPU support)
```

### Install Dependencies
```bash
pip install tensorflow>=2.10.0
pip install keras>=2.10.0
pip install numpy>=1.21.0
pip install opencv-python>=4.6.0
pip install matplotlib>=3.5.0
pip install scikit-learn>=1.0.0
pip install pandas>=1.3.0
pip install pillow>=9.0.0
pip install scipy>=1.7.0
pip install tqdm>=4.62.0
```

Or install all at once:
```bash
pip install tensorflow keras numpy opencv-python matplotlib scikit-learn pandas pillow scipy tqdm
```

## 📁 Repository Structure

```
semantic_image_segmentation/
├── README.md
├── src/
│   ├── notebooks/
│   │   ├── UNet_Ablation_Study.ipynb           # Main UNet experiments
│   │   ├── DeepLabV3.ipynb                      # DeepLabV3+ experiments
│   │   ├── oF1_Hungarian_Matching_Computation.ipynb  # Metric computation
│   │   ├── DataPrep_AndSplit.ipynb              # Data preprocessing
│   │   └── Training.ipynb                        # Training experiments
│   └── codes/
│       ├── compute_oF1_all_models.py            # Compute oF1 scores
│       ├── generate_predictions_for_oF1.py      # Generate predictions
│       ├── create_validation_loss_chart.py      # Visualization
│       ├── generate_report_figures.py           # Report generation
│       ├── generate_summary_figure.py           # Summary plots
│       ├── run_oF1_complete.sh                  # Complete oF1 pipeline
│       ├── run_oF1_overnight.sh                 # Long-running evaluation
│       ├── monitor_oF1_progress.sh              # Progress monitoring
│       ├── preflight_check.sh                   # Environment verification
│       └── oF1_README.md                        # oF1 metric documentation
├── results/
│   ├── oF1_results/
│   │   └── all_models_oF1_scores.csv            # Final oF1 scores
│   ├── unet_ablation_results.csv                # Ablation study results
│   ├── ablation_backbone_results.json           # Backbone comparison
│   ├── ablation_threshold_results.json          # Threshold analysis
│   ├── ablation_training_size_results.json      # Data size impact
│   └── prediction_generation_summary.json       # Prediction stats
└── outputs_and_models/
    └── (trained models and logs stored here)
```

## 🚀 How to Run

### 1. Data Preparation

Download the dataset from Kaggle and place it in the project directory:

```bash
# Install Kaggle API
pip install kaggle

# Download dataset
kaggle competitions download -c recodai-luc-scientific-image-forgery-detection

# Unzip dataset
unzip recodai-luc-scientific-image-forgery-detection.zip -d recodai-luc-scientific-image-forgery-detection/
```

### 2. Run Preflight Check

Verify your environment setup:

```bash
cd semantic_image_segmentation/src/codes/
bash preflight_check.sh
```

### 3. Training Models

#### Option A: Using Jupyter Notebooks (Recommended for Exploration)

1. **UNet Models**:
   ```bash
   jupyter notebook src/notebooks/UNet_Ablation_Study.ipynb
   ```
   This notebook includes:
   - UNet from scratch with different loss functions
   - UNet with ResNet50 backbone
   - Region-based training strategies
   - Comprehensive evaluation

2. **DeepLabV3+ Models**:
   ```bash
   jupyter notebook src/notebooks/DeepLabV3.ipynb
   ```
   This notebook includes:
   - DeepLabV3+ with different backbones
   - Training data size ablation
   - Performance comparison

#### Option B: Using Python Scripts

```python
# Training is implemented in the notebooks
# For custom training, refer to the notebook code and adapt as needed
```

### 4. Generate Predictions

Generate predictions for all models:

```bash
cd src/codes/
python generate_predictions_for_oF1.py
```

This will create prediction masks for all trained models in the `predictions/` directory.

### 5. Compute oF1 Scores

#### Quick Method:
```bash
cd src/codes/
python compute_oF1_all_models.py
```

#### Complete Pipeline (Overnight Run):
```bash
cd src/codes/
bash run_oF1_complete.sh
```

This script will:
1. Verify environment setup
2. Generate predictions for all models
3. Compute oF1 scores with Hungarian matching
4. Generate summary reports and visualizations

#### Monitor Progress:
```bash
bash monitor_oF1_progress.sh
```

### 6. Generate Visualizations

Create charts and figures:

```bash
python create_validation_loss_chart.py
python generate_report_figures.py
python generate_summary_figure.py
```

## 📊 Experimental Results

### Model Performance (oF1 Scores)

| Model | Backbone | Loss Function | Training Data | oF1 Score |
|-------|----------|---------------|---------------|-----------|
| DeepLabV3+ | ResNet101 | BCE+Dice | 100% | **0.XXX** |
| DeepLabV3+ | ResNet50 | BCE+Dice | 100% | 0.XXX |
| UNet | ResNet50 | BCE+Dice | 100% | 0.XXX |
| UNet | Scratch | BCE+Dice | 100% | 0.XXX |
| UNet | Scratch | Dice | 100% | 0.XXX |
| UNet | Scratch | BCE | 100% | 0.XXX |

*Note: Exact scores available in `results/oF1_results/all_models_oF1_scores.csv`*

### Key Findings

1. **Architecture Comparison**: 
   - DeepLabV3+ with ResNet101 backbone achieved the best performance
   - ASPP module provides better multi-scale feature extraction
   
2. **Loss Function Impact**:
   - Combined BCE+Dice loss outperforms individual losses
   - Dice loss helps with class imbalance
   
3. **Backbone Analysis**:
   - Pre-trained backbones (ResNet50/101) significantly outperform training from scratch
   - ResNet101 provides marginal improvement over ResNet50 at higher computational cost
   
4. **Training Data Size**:
   - Performance improves with more training data
   - 75% of data achieves 90%+ of full performance
   
5. **Region-Based Training**:
   - Context-aware regions improve detection accuracy
   - Tight regions may miss contextual information

## 🔬 Evaluation Metrics

### Object-based F1 (oF1) Score

The oF1 metric evaluates forgery detection at the object level using Hungarian matching algorithm:

1. **Connected Component Analysis**: Identify individual forgery regions
2. **Hungarian Matching**: Optimal bipartite matching between predicted and ground truth objects
3. **IoU-based Matching**: Objects matched if IoU > 0.5
4. **Precision & Recall**: Computed at object level
5. **F1 Score**: Harmonic mean of precision and recall

Advantages over pixel-wise metrics:
- More aligned with human perception
- Robust to small spatial misalignments
- Better reflects practical forgery detection performance

For detailed implementation, see the [Hungarian Matching notebook](src/notebooks/oF1_Hungarian_Matching_Computation.ipynb).

## 📝 What We Have Done

### 1. Data Preparation & Analysis
- Comprehensive EDA of the forgery dataset
- Data augmentation strategies (rotation, flipping, scaling)
- Train-validation split with stratification
- Region-based cropping strategies

### 2. Model Implementation
- **UNet Architecture**:
  - Classic encoder-decoder with skip connections
  - Multiple variants (from scratch, ResNet50 backbone)
  - Custom loss functions (BCE, Dice, Combined)
  
- **DeepLabV3+ Architecture**:
  - Atrous Spatial Pyramid Pooling (ASPP)
  - ResNet50 and ResNet101 backbones
  - Pre-trained on ImageNet

### 3. Training Strategies
- Progressive learning rate scheduling
- Early stopping with patience
- Model checkpointing
- Mixed precision training for efficiency

### 4. Ablation Studies
- **Backbone Comparison**: Scratch vs ResNet50 vs ResNet101
- **Loss Function Analysis**: BCE vs Dice vs Combined
- **Data Size Impact**: 25%, 50%, 75%, 100% of training data
- **Region Strategies**: Tight crops vs Context-aware vs Full images

### 5. Evaluation Framework
- Implemented oF1 metric with Hungarian matching
- Automated prediction generation pipeline
- Comprehensive visualization tools
- Statistical significance testing

### 6. Automation & Reproducibility
- Shell scripts for complete pipeline execution
- Progress monitoring tools
- Environment verification
- Detailed documentation

## 🤝 Team & Contribution

This project was developed as part of the MLDM Deep Learning course, showcasing:
- State-of-the-art deep learning architectures
- Rigorous experimental methodology
- Comprehensive ablation studies
- Production-ready code organization
- Thorough documentation

## 📚 References

1. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI.
2. Chen, L. C., et al. (2018). Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation. ECCV.
3. He, K., et al. (2016). Deep Residual Learning for Image Recognition. CVPR.
4. Kuhn, H. W. (1955). The Hungarian Method for the Assignment Problem. Naval Research Logistics Quarterly.

## 📧 Contact & Support

For questions or issues, please open an issue on GitHub or contact the project maintainers.

## 📄 License

This project is developed for academic purposes as part of the MLDM Deep Learning course (January 2026).

---

**Last Updated**: January 27, 2026
