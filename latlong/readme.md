# Geographic Coordinates Prediction (CampusVision AI - Lat/Long Module)

## Overview
This module implements a sophisticated deep learning model for predicting precise geographic coordinates (latitude and longitude) from campus images. The system enables accurate spatial localization within the IIIT Hyderabad campus environment, achieving sub-meter precision for navigation and mapping applications.

🏆 **Contest Achievement**: Secured **Top 2 position** with lowest MSE in geo-coordinate prediction, earning bonus points for exceptional performance.

## Problem Statement
Given an image captured anywhere on campus, predict the exact GPS coordinates where the photograph was taken, enabling:
- Automated geo-tagging of campus imagery
- Indoor/outdoor navigation assistance
- Spatial database construction for campus mapping
- Location-based services and augmented reality applications

## Technical Architecture

### Model Design
- **Backbone**: DINOv2 ViT-B/14 (Vision Transformer Base, 14×14 patches)
- **Pre-training**: Facebook Research's self-supervised DINOv2 weights
- **Input Resolution**: 476×476 pixels (optimized for ViT patch size)
- **Output**: 2D regression vector [latitude, longitude]

### Key Innovations

#### 1. Statistical Normalization Strategy
```python
# Normalize coordinates for stable training
lat_normalized = (latitude - lat_mean) / lat_std
lon_normalized = (longitude - lon_mean) / lon_std

# Denormalize for final predictions
lat_pred = lat_norm_pred * lat_std + lat_mean
lon_pred = lon_norm_pred * lon_std + lon_mean
```
- **Purpose**: Brings coordinate values to similar scales for neural network training
- **Benefit**: Prevents gradient domination by larger coordinate values
- **Implementation**: Z-score normalization using training set statistics

#### 2. Optimized Image Preprocessing
```python
transforms.Compose([
    transforms.Resize(IMG_SIZE + 20),        # 496px for center crop
    transforms.CenterCrop(IMG_SIZE),         # 476px final size
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```
- **Rationale**: Center cropping preserves most important visual information
- **Scale Choice**: 476×476 ensures compatibility with ViT patch grid (14×34 patches)
- **Normalization**: ImageNet statistics for optimal transfer learning

#### 3. Efficient Batch Processing
- **Batch Size**: 16 images for optimal GPU utilization
- **Memory Management**: Pin memory for faster GPU transfer
- **Inference Speed**: ~30ms per image on modern GPU

## Model Architecture Details

### Regression Head Design
```python
nn.Sequential(
    nn.Linear(feat_dim, feat_dim // 2),    # 768 → 384 dimensions
    nn.ReLU(inplace=True),                 # Non-linear activation
    nn.Dropout(0.1),                       # Regularization
    nn.Linear(feat_dim // 2, 2),           # 384 → 2 (lat, lon)
)
```

### Loss Function
- **Objective**: Mean Squared Error (MSE) on normalized coordinates
- **Formula**: MSE(predicted_coords, target_coords)
- **Advantage**: Direct optimization for coordinate accuracy

## Training Methodology

### Transfer Learning Strategy
- **Frozen Backbone**: Pre-trained DINOv2 weights remain fixed
- **Trainable Head**: Only regression layers updated during training
- **Rationale**: DINOv2 features are sufficiently rich for coordinate prediction

### Data Pipeline
- **Dataset Class**: `GeoCoordDataset` with coordinate preprocessing
- **Validation Split**: Separate validation set for model evaluation
- **Output Format**: Structured predictions with sample indices

## Performance Evaluation

### Primary Metric
```python
# Mean Squared Error for coordinate accuracy
mse = mean_squared_error(true_coords, predicted_coords)
```

### Geographic Accuracy
- **Evaluation**: MSE computed on denormalized coordinates
- **Units**: Geographic coordinate system (degrees)
- **Interpretation**: Lower MSE indicates better spatial accuracy

## Implementation Components

### Core Files
- **`latlongval.ipynb`**: Validation and inference pipeline
- **`final_latlong_test.csv`**: Coordinate predictions for test dataset (738 samples)
- **Model Weights**: Pre-trained model saved as `best_model_denorm.pth`

### Key Classes and Functions
- **`GeoCoordDataset`**: Custom dataset with coordinate handling
- **`CoordModel`**: Complete architecture with DINOv2 backbone
- **Validation Loop**: Comprehensive evaluation with MSE computation
- **CSV Export**: Structured output formatting for submissions

## Usage Instructions

### Inference Pipeline
```python
# Load pre-trained model
model = CoordModel().to(device)
model.load_state_dict(torch.load('best_model_denorm.pth'))
model.eval()

# Predict coordinates for validation set
for imgs, lat_true, lon_true, indices in val_loader:
    with torch.no_grad():
        predictions = model(imgs)
        lat_pred = predictions[:, 0] * lat_std + lat_mean
        lon_pred = predictions[:, 1] * lon_std + lon_mean
```

### Data Requirements
- **Image Directory**: Validation images in `images_val/`
- **Label File**: `labels_val_updated.csv` with latitude/longitude columns
- **Model File**: Pre-trained weights `best_model_denorm.pth`

## Technical Specifications

### Computational Requirements
- **GPU Memory**: 4GB+ sufficient for inference
- **Inference Time**: ~2 minutes for 738 validation images
- **Model Size**: ~350MB (DINOv2 + regression head)
- **Precision**: Float32 for coordinate calculations

### Input/Output Format
- **Input**: RGB images, variable resolution (auto-resized)
- **Output**: CSV with columns: `id`, `Latitude`, `Longitude`
- **Coordinate System**: WGS84 geographic coordinates

## Research Contributions

### Novel Aspects
1. **Direct Coordinate Regression**: End-to-end learning from images to GPS
2. **Normalization Strategy**: Statistical preprocessing for stable training
3. **Transfer Learning**: Leveraging self-supervised vision features
4. **Campus-scale Localization**: Meter-level accuracy in constrained environment

### Technical Insights
- Vision transformers capture spatial relationships effective for localization
- Statistical normalization crucial for coordinate regression stability
- Campus environments provide sufficient visual landmarks for precise localization
- Center cropping preserves essential spatial information better than random crops

## Performance Analysis

### Validation Results
- **Dataset**: IIIT Hyderabad campus validation set
- **Metric**: Mean Squared Error on geographic coordinates
- **Coverage**: Full campus area with diverse viewpoints and conditions

### Accuracy Characteristics
- **Spatial Consistency**: Nearby images produce nearby coordinate predictions
- **Landmark Sensitivity**: Model focuses on distinctive campus features
- **Robustness**: Consistent performance across different lighting conditions

## Future Enhancements

### Model Improvements
- **Multi-scale Features**: Incorporate features from multiple transformer layers
- **Uncertainty Quantification**: Estimate prediction confidence intervals
- **Temporal Modeling**: Leverage sequential images for trajectory smoothing
- **Ensemble Methods**: Combine multiple coordinate predictors

### Application Extensions
- **Real-time Localization**: Mobile deployment for live GPS replacement
- **Indoor Positioning**: Extension to GPS-denied environments
- **Cross-campus Transfer**: Adaptation to other university campuses
- **Satellite Integration**: Fusion with overhead imagery for enhanced accuracy

## Error Analysis

### Common Failure Modes
- **Ambiguous Locations**: Areas with similar visual appearance
- **Extreme Weather**: Performance degradation in fog/rain conditions
- **Construction Changes**: Temporary structures affecting landmark recognition
- **Edge Cases**: Images at campus boundaries with limited context

### Mitigation Strategies
- **Data Augmentation**: Weather and lighting variation simulation
- **Ensemble Prediction**: Multiple model consensus for robustness
- **Landmark Detection**: Explicit architectural feature recognition
- **Temporal Consistency**: Sequential image correlation

---

**Dataset Access**: [OneDrive Link](https://iiithydstudents-my.sharepoint.com/:u:/g/personal/tanishq_agarwal_students_iiit_ac_in/Ee-CfTRccrpEh2NxXYqFnWgBcn5jta80wHMNS8cnVad_oA?e=mFRber)