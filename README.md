# Leukemia Classification using Deep Learning

A comprehensive deep learning solution for classifying Acute Lymphoblastic Leukemia (ALL) and Healthy (HEM) blood cell images using state-of-the-art CNN architectures.

## 🔬 Overview

This project implements an ensemble of CNN models to classify microscopic blood cell images for leukemia detection. The system achieves high accuracy in distinguishing between ALL (Acute Lymphoblastic Leukemia) cells and HEM (healthy cells) using advanced computer vision techniques.

## 📊 Dataset

The project uses the CNMC2019 dataset with the following structure:
```
dataset/cnmc2019_splits/images/
├── train/
│   ├── all/  # ALL (Acute Lymphoblastic Leukemia) images
│   └── hem/  # HEM (Healthy) images
├── val/
│   ├── all/
│   └── hem/
└── test/
    ├── ALL/  # Note: uppercase in test set
    └── HEM/
```

## 🏗️ Architecture

### Models Implemented
- **EfficientNetV2-B3**: Primary model with optimized performance
- **ResNet variants**: Robust feature extraction
- **DenseNet architectures**: Dense connectivity patterns
- **Ensemble approach**: Combines predictions from multiple models

### Key Features
- **Advanced Data Augmentation**: Mixup, rotation, color jittering
- **Mixed Precision Training**: Faster training with automatic mixed precision
- **Class Balancing**: Weighted loss functions for imbalanced datasets
- **Early Stopping**: Prevents overfitting with patience-based stopping
- **Test Time Augmentation (TTA)**: Improves inference accuracy

## 🚀 Getting Started

### Prerequisites
```bash
pip install torch torchvision
pip install timm
pip install scikit-learn
pip install pandas numpy matplotlib seaborn
pip install Pillow tqdm
```

### Configuration
Create a `config.py` file with your settings:
```python
import torch

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data paths
TRAIN_DIR = 'dataset/cnmc2019_splits/images/train'
VAL_DIR = 'dataset/cnmc2019_splits/images/val'
TEST_DIR = 'dataset/cnmc2019_splits/images/test'

# Model configuration
IMAGE_SIZE = 224
NUM_CLASSES = 2
BATCH_SIZE = 32
NUM_WORKERS = 4

# Training parameters
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.01
SEED = 42

# Models to train
CNN_MODELS = [
    'tf_efficientnetv2_b3',
    # Add other models as needed
]

# Save directory
SAVE_DIR = 'models/modular'
```

## 📚 Usage

### Training Models
```python
# Run the training pipeline
python train.py

# Or use the Jupyter notebook
jupyter notebook "train.ipynb"
```

### Evaluating Models
```python
# Evaluate all trained models
python eval_img.py

# Or use the evaluation notebook
jupyter notebook eval_img.ipynb
```

### Single Image Prediction
```python
from eval_img import predict_single_image

# Predict with a specific model
result = predict_single_image('path/to/image.jpg', model_name='tf_efficientnetv2_b3')

# Predict with ensemble
result = predict_single_image('path/to/image.jpg')

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.4f}")
```

## 📈 Results

### Model Performance
The system achieves the following metrics on the test set:

| Model | F1-Score | Accuracy | Precision | Recall |
|-------|----------|----------|-----------|---------|
| EfficientNetV2-B3 | 0.9529 | 0.9533 | 0.95+ | 0.94+ |
| Ensemble (Voting) | 0.9529 | 0.9533 | 0.95+ | 0.94+ |
| Ensemble (Average) | 0.9529 | 0.9533 | 0.95+ | 0.94+ |

### Error Analysis Insights
Patient-level error analysis reveals that most misclassifications are concentrated in specific patients:
- **Patient h3**: 16 misclassified images (highest error concentration)
- **Patient 15**: 2 misclassified images
- **Patient H4**: 2 misclassified images

This pattern suggests that certain patients may have atypical cell characteristics or image quality issues that make classification more challenging, which is valuable information for clinical validation.

### Key Metrics
- **High Sensitivity**: Minimizes false negatives for medical safety
- **Robust Precision**: Reduces false positives
- **Balanced Performance**: Works well on both classes
- **Clinical Reliability**: Suitable for medical screening applications

## 🔧 Advanced Features

### Ensemble Methods
- **Voting Ensemble**: Majority vote from multiple models
- **Probability Averaging**: Weighted average of prediction probabilities
- **Model Selection**: Automatic best model selection based on validation performance

### Data Leakage Prevention
- **Patient-Level Splitting**: Ensures no patient appears in both train and test sets
- **Comprehensive Validation**: Built-in data leakage detection
- **Robust Evaluation**: Patient-independent performance assessment

### Error Analysis
```python
from eval_img import analyze_errors

# Analyze model errors and confidence
errors_df, low_confidence_df = analyze_errors()
```

## 📊 Visualization

The system provides comprehensive visualizations:
- **Confusion Matrices**: Per-model and ensemble performance
- **ROC Curves**: Model discrimination ability
- **Confidence Distribution**: Prediction reliability analysis
- **Error Analysis**: Detailed failure case examination

## 🔍 Model Architecture Details

### CNN Model Structure
```python
class CNNModel(nn.Module):
    def __init__(self, model_name, num_classes=2):
        super().__init__()
        # Pretrained backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            global_pool='avg'
        )
        
        # Custom classifier with regularization
        self.classifier = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
```

### Training Strategy
- **Progressive Learning Rates**: Different rates for backbone and classifier
- **Cosine Annealing**: Dynamic learning rate scheduling  
- **Gradient Clipping**: Prevents exploding gradients
- **Mixed Precision**: Accelerated training with AMP

## 📁 Project Structure
```
├── config.py                 # Configuration settings
├── train copy 2.ipynb       # Training notebook
├── eval_img.ipynb           # Evaluation notebook
├── models/modular/          # Saved model checkpoints
├── test_results/            # Evaluation outputs
├── dataset/                 # Dataset directory
└── README.md               # This file
```

## 🩺 Medical Application Notes

This system is designed for research and educational purposes. For clinical applications:
- Always validate with medical professionals
- Consider regulatory requirements (FDA, CE marking)
- Implement proper quality assurance protocols
- Ensure patient data privacy compliance

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- CNMC2019 dataset contributors
- PyTorch and timm library developers
- Medical imaging research community

## 📞 Contact

For questions or collaborations, please open an issue or contact the maintainers.

---

**Disclaimer**: This tool is for research purposes only and should not be used for clinical diagnosis without proper medical supervision and validation.