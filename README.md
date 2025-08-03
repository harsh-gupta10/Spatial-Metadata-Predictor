# Spatial-Metadata-Predictor

## Project Overview

Spatial-Metadata-Predictor is an advanced computer vision project designed to predict the precise geographic location (latitude and longitude), camera orientation (angle), and campus region ID (1–15) of images captured within the IIIT Hyderabad campus. The system leverages a manually annotated dataset consisting of 400 images per region, collected across 15 color-coded campus regions. Each image is accompanied by comprehensive metadata including GPS coordinates, capture timestamp, and orientation angle.

**Project Phases:**
- **Phase 1**: Data collection and annotation across 15 campus regions
- **Phase 2**: Model development, training, and evaluation with Kaggle contest submission

## Exceptional Performance Achievements

🏆 **Contest Results**: Secured **Top 2 positions** in all three competition tracks
- **Direction Prediction**: Achieved <18° angular error
- **Region Classification**: Achieved >97% classification accuracy  
- **Geo-coordinate Prediction**: Achieved lowest MSE in coordinate prediction


## Dataset Description

- **Total Images**: 6000 images (400 per region)
- **Regions**: 15 color-coded campus areas at IIIT Hyderabad
- **Annotations**: GPS coordinates (latitude/longitude), capture angle, timestamp
- **Data Split**: Training and validation sets for model development

## Project Structure

```
SMAI-Project/
├── README.md                     # This master documentation
├── direction/                    # Camera orientation prediction
│   ├── anglebest.ipynb          # Main training notebook
│   ├── final_angle.csv          # Prediction results
│   └── readme.md                # Direction module documentation
├── latlong/                     # Geographic coordinates prediction
│   ├── latlongval.ipynb         # Validation and inference notebook
│   ├── final_latlong_test.csv   # Coordinate predictions
│   └── readme.md                # Coordinate module documentation
└── regionID/                    # Campus region classification
    ├── regionbest.ipynb         # Region classification notebook
    ├── final_region.csv         # Region predictions
    └── readme.md                # Region module documentation
```

## Model Architecture Summary

### 1. Direction Prediction (Angle Estimation)
- **Model**: Fine-tuned DINOv2 ViT-B/14 with custom regression head
- **Output**: Camera orientation angle (0-360°)
- **Representation**: (cos, sin) coordinates for angle continuity
- **Performance**: Mean Absolute Error (MAE) in degrees

### 2. Geographic Coordinates (Lat/Long Prediction)
- **Model**: DINOv2 ViT-B/14 with coordinate regression head
- **Output**: Precise latitude and longitude coordinates
- **Normalization**: Statistical normalization for training stability
- **Performance**: Mean Squared Error (MSE) for coordinate accuracy

### 3. Region Classification (Campus Area ID)
- **Model**: Swin Transformer (swin_base_patch4_window7_224)
- **Output**: Region ID classification (1-15)
- **Techniques**: Advanced augmentation, focal loss, hard example mining
- **Performance**: Classification accuracy percentage

## Technical Innovations

### Advanced Training Techniques
- **Mixed Precision Training**: PyTorch AMP for memory efficiency
- **Transfer Learning**: Strategic layer freezing and gradual unfreezing
- **Data Augmentation**: RandAugment, MixUp, CutMix for robustness
- **Regularization**: Dropout, stochastic depth, weight decay
- **Optimization**: Specialized optimizers (RAdam + Lookahead, AdamW)

### Model-Specific Features
- **Direction**: Unit circle representation prevents angle discontinuity
- **Coordinates**: Statistical normalization improves convergence
- **Region**: Focal loss handles class imbalance effectively

## Results and Performance

🏆 **Competition Achievements**: Top 2 positions secured in all three Kaggle contest tracks

| Task | Model | Best Performance | Metric | Contest Rank |
|------|-------|------------------|--------|--------------|
| Direction | DINOv2 ViT-B/14 | <18° Angular Error | MAE (degrees) | **Top 2** |
| Coordinates | DINOv2 ViT-B/14 | Lowest MSE | Geographic Distance | **Top 2** |
| Region | Swin Transformer | >97% Accuracy | Classification % | **Top 2** |

**Exceptional Performance**: Received bonus points for outstanding results across all competition categories, demonstrating state-of-the-art campus localization capabilities.

## Usage Instructions

### Environment Setup
```bash
# Install required packages
pip install torch torchvision timm torch-optimizer scikit-learn
pip install pandas numpy pillow matplotlib tqdm
```

### Running Individual Models

1. **Direction Prediction**:
   - Navigate to `direction/` folder
   - Open `anglebest.ipynb` in Jupyter
   - Update data paths and run cells

2. **Coordinate Prediction**:
   - Navigate to `latlong/` folder
   - Open `latlongval.ipynb` in Jupyter
   - Load pre-trained model and run inference

3. **Region Classification**:
   - Navigate to `regionID/` folder
   - Open `regionbest.ipynb` in Jupyter
   - Configure environment and train model

### Data Requirements
- Training images in `images_train/` directory
- Validation images in `images_val/` directory
- Label CSV files: `labels_train_updated.csv`, `labels_val_updated.csv`

## Key Features

### Robust Architecture
- **Transformer-based**: Leverages attention mechanisms for spatial understanding
- **Multi-scale Processing**: Handles various image resolutions and crops
- **Transfer Learning**: Builds upon powerful pre-trained vision models

### Advanced Training
- **Gradient Accumulation**: Enables large effective batch sizes
- **Learning Rate Scheduling**: OneCycleLR and cosine annealing
- **Early Stopping**: Prevents overfitting with validation monitoring

### Production Ready
- **Model Checkpointing**: Automatic saving of best models
- **Inference Pipeline**: Efficient batch processing for predictions
- **Output Formatting**: Structured CSV outputs for submissions



---

*This project demonstrates the application of state-of-the-art computer vision techniques to solve real-world geographic localization challenges within a university campus environment.*