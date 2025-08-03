# Camera Orientation Prediction (CampusVision AI - Direction Module)

## Overview
This module implements a state-of-the-art deep learning model for predicting camera orientation angles from campus images. The model determines the compass direction (0-360°) that the camera was facing when each photograph was captured within the IIIT Hyderabad campus.

🏆 **Contest Achievement**: Secured **Top 2 position** with <20° angular error, earning bonus points for exceptional performance.

## Problem Statement
Given an image captured on campus, predict the precise camera orientation angle, enabling applications such as:
- Automated image organization by viewing direction
- Navigation assistance and wayfinding
- Augmented reality applications requiring orientation data
- Photogrammetric reconstruction and mapping

## Technical Architecture

### Model Design
- **Backbone**: DINOv2 ViT-B/14 (Vision Transformer Base, 14×14 patches)
- **Pre-training**: Facebook Research's self-supervised DINOv2 weights
- **Input Resolution**: 476×476 pixels (optimized for 14×14 patch grid)
- **Output**: 2D unit vector (cos θ, sin θ) representing angle on unit circle

### Key Innovations

#### 1. Unit Circle Representation
```python
# Convert angle to unit circle coordinates
angle_rad = angle_degrees * π / 180
target = [cos(angle_rad), sin(angle_rad)]
```
- **Advantage**: Eliminates discontinuity at 0°/360° boundary
- **Benefit**: Smooth gradient flow for angles near north direction
- **Mathematical**: Leverages circular topology of angle space

#### 2. Strategic Transfer Learning
- **Frozen Components**: Patch embedding + first 50% of transformer blocks
- **Fine-tuned Components**: Latter transformer layers + custom regression head
- **Rationale**: Low-level features transfer well, high-level features need adaptation

#### 3. Advanced Data Augmentation
```python
transforms.Compose([
    transforms.RandomResizedCrop(476, scale=(0.5, 1.0)),
    autoaug.RandAugment(),
    transforms.ColorJitter(0.3, 0.3, 0.3),
    transforms.RandomErasing(p=0.1)
])
```

## Training Strategy

### Optimization Framework
- **Primary Optimizer**: AdamW with differential learning rates
  - Backbone parameters: 5×10⁻⁶ (conservative fine-tuning)
  - Head parameters: 5×10⁻⁴ (aggressive learning for new layers)
- **Weight Decay**: 1×10⁻² for regularization
- **Gradient Clipping**: Max norm 1.0 to prevent exploding gradients

### Learning Rate Scheduling
- **Method**: Cosine Annealing with warm restarts
- **T_max**: 30 epochs for initial training, extended for continued training
- **η_min**: 1×10⁻⁷ minimum learning rate
- **Benefits**: Smooth convergence with periodic exploration

### Mixed Precision Training
- **Framework**: PyTorch Automatic Mixed Precision (AMP)
- **Scaler**: Dynamic loss scaling for stable gradients
- **Memory Savings**: ~40% reduction in GPU memory usage
- **Speed Improvement**: 1.5-2× faster training on modern GPUs

### Advanced Regularization
- **Stochastic Depth**: DropPath with rate 0.1 in transformer blocks
- **Dropout**: 0.1 probability in regression head layers
- **Data Augmentation**: RandAugment with magnitude 9

## Model Architecture Details

### Regression Head
```python
nn.Sequential(
    nn.Linear(768, 384),      # Reduce dimensionality
    nn.ReLU(inplace=True),    # Non-linear activation
    nn.Dropout(0.1),          # Regularization
    nn.Linear(384, 2),        # Output: [cos, sin]
)
```

### Loss Function
- **Objective**: Mean Squared Error on unit circle coordinates
- **Formula**: MSE(predicted_vector, target_vector)
- **Normalization**: Output vectors normalized to unit length

## Performance Metrics

### Evaluation Method
```python
def angle_mae(pred, target):
    dot_product = (pred * target).sum(dim=1).clamp(-1, 1)
    angle_diff = torch.acos(dot_product)  # Radians
    return (angle_diff * 180 / π).mean()  # Convert to degrees
```

### Training Results
- **Metric**: Mean Absolute Error (MAE) in degrees
- **Validation Performance**: Competitive results on campus dataset
- **Training Epochs**: 30 initial + 20 extended epochs
- **Best Model**: Automatically saved based on validation MAE

## Implementation Files

### Core Components
- **`anglebest.ipynb`**: Complete training pipeline with enhanced features
- **`final_angle.csv`**: Model predictions for test dataset (738 samples)
- **Model Weights**: Saved as `best_model.pth` during training

### Key Functions
- **`GeoAngleDataset`**: Custom dataset class with angle preprocessing
- **`AngleModel`**: Complete model architecture with DINOv2 backbone
- **`angle_mae`**: Specialized metric for circular angle evaluation
- **`validate`**: Comprehensive validation loop with metric computation

## Usage Instructions

### Quick Start
```python
# Load pre-trained model
model = AngleModel().to(device)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Predict angle for new image
with torch.no_grad():
    prediction = model(image_tensor)
    angle = torch.atan2(prediction[1], prediction[0]) * 180 / π
```

### Training from Scratch
1. **Data Preparation**: Organize images and CSV labels
2. **Environment Setup**: Install PyTorch, timm, and dependencies
3. **Configuration**: Update data paths in notebook
4. **Execution**: Run all cells in `anglebest.ipynb`
5. **Monitoring**: Track validation MAE for model selection

## Technical Specifications

### Computational Requirements
- **GPU Memory**: 8GB+ recommended for batch size 16
- **Training Time**: ~2-3 hours on modern GPU (RTX 3080/4080)
- **Inference Speed**: ~50ms per image on GPU
- **Model Size**: ~350MB (DINOv2 ViT-B/14 + regression head)

### Data Requirements
- **Image Format**: RGB, various resolutions (auto-resized to 476×476)
- **Label Format**: CSV with 'filename' and 'angle' columns
- **Angle Range**: 0-360° (automatically normalized)

## Research Contributions

### Novel Aspects
1. **Unit Circle Formulation**: Elegant solution to angle discontinuity
2. **Differential Learning Rates**: Optimized fine-tuning strategy
3. **Advanced Augmentation**: Tailored for orientation invariance
4. **Mixed Precision**: Efficient training for large vision transformers

### Experimental Insights
- Transformer architectures excel at spatial orientation tasks
- Strategic layer freezing crucial for effective transfer learning
- Unit circle representation significantly improves convergence
- Mixed precision essential for memory-efficient transformer training

## Future Enhancements

### Potential Improvements
- **Multi-scale Features**: Combine features from multiple transformer layers
- **Temporal Consistency**: Leverage sequential images for smoothing
- **Uncertainty Estimation**: Bayesian approaches for confidence intervals
- **Cross-domain Transfer**: Adaptation to other campus environments

### Research Directions
- **Attention Visualization**: Understanding model focus areas
- **Robustness Analysis**: Performance under weather/lighting variations
- **Ensemble Methods**: Combining multiple orientation predictors
- **Real-time Optimization**: Model compression for mobile deployment

---

**Dataset Access**: [OneDrive Link](https://iiithydstudents-my.sharepoint.com/:u:/g/personal/tanishq_agarwal_students_iiit_ac_in/Ee-CfTRccrpEh2NxXYqFnWgBcn5jta80wHMNS8cnVad_oA?e=mFRber)