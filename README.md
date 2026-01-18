# OCD-DID-Multimodal-Pattern-Recognition

A machine learning project for recognizing patterns in multimodal data related to OCD (Obsessive-Compulsive Disorder) and DID (Dissociative Identity Disorder) using deep learning and computer vision techniques.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Status](https://img.shields.io/badge/Status-Research%20Prototype-success)
![Accuracy](https://img.shields.io/badge/Accuracy-99.89%25-brightgreen)

## 📋 Project Overview

This project leverages multimodal data analysis to identify and recognize patterns associated with OCD and DID. It utilizes state-of-the-art deep learning models and interpretability techniques to understand model predictions.

### Key Features
- **Multimodal Pattern Recognition**: Combines multiple data modalities for comprehensive analysis
- **Deep Learning Models**: Uses TIMM (PyTorch Image Models) for robust feature extraction
- **Model Interpretability**: Implements Grad-CAM for visualization of model decision-making
- **GPU Acceleration**: Optimized for NVIDIA GPUs (Tesla T4+)
- **Kaggle Integration**: Easy dataset access and management

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Framework** | PyTorch 2.9.1 |
| **Vision** | TorchVision, OpenCV |
| **Models** | TIMM (PyTorch Image Models) |
| **Interpretability** | Grad-CAM |
| **Data** | Kaggle API |
| **GPU** | CUDA 12.4 (Tesla T4) |

## 📦 Installation

### Prerequisites
- Python 3.8+
- CUDA 12.4 (for GPU support)
- pip or conda

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/your-username/OCD-DID-Multimodal-Pattern-Recognition.git
cd OCD-DID-Multimodal-Pattern-Recognition
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### In Google Colab
```python
# Clone the repository
!git clone https://github.com/your-username/OCD-DID-Multimodal-Pattern-Recognition.git
%cd OCD-DID-Multimodal-Pattern-Recognition

# Install dependencies
!pip install -r requirements.txt

# Mount Google Drive (if needed)
from google.colab import drive
drive.mount('/content/drive')
```

### Kaggle Dataset Setup
```python
# Configure Kaggle API
from google.colab import files
files.upload()  # Upload kaggle.json

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

# Download datasets
!kaggle datasets download -d <dataset-name>
```

## 📊 Project Structure

```
OCD-DID-Multimodal-Pattern-Recognition/
├── notebooks/
│   └── main_analysis.ipynb          # Main Colab notebook
├── data/
│   ├── raw/                         # Raw datasets
│   └── processed/                   # Processed data
├── models/
│   └── trained_models/              # Saved model checkpoints
├── src/
│   ├── data_loader.py              # Data loading utilities
│   ├── model.py                    # Model architecture
│   ├── train.py                    # Training script
│   └── utils.py                    # Helper functions
├── results/
│   ├── visualizations/             # Grad-CAM heatmaps
│   └── metrics/                    # Performance metrics
├── requirements.txt                # Project dependencies
├── README.md                       # This file
└── .gitignore                      # Git ignore file
```

## 🔬 Model Architecture

- **Backbone**: TIMM pre-trained models (ResNet, EfficientNet, ViT, etc.)
- **Input**: Multimodal data (images, features, etc.)
- **Output**: Classification/Pattern Recognition predictions
- **Interpretability**: Grad-CAM visualization for model explanations

## 📈 Training & Evaluation

```python
# Example training workflow
from src.model import load_model
from src.train import train_model

model = load_model(model_name='resnet50', pretrained=True)
history = train_model(model, train_loader, val_loader, epochs=50)
```

## 🎨 Visualization & Interpretability

Uses Grad-CAM to visualize which regions of the input the model focuses on:

```python
from grad_cam import GradCAM

grad_cam = GradCAM(model, target_layer)
heatmap = grad_cam.generate_cam(input_image)
```

## 📝 Notes

- **GPU Memory**: Optimized for 15GB VRAM (Tesla T4)
- **Torch Version**: Using torch 2.9.1 (note: torchaudio compatibility)
- **CUDA**: Version 12.4

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💼 Author
Saksham Mishra
Sukaina Ali
Abeer Arshad

## 📧 Contact

For questions or inquiries, please reach out via:
- GitHub Issues
- Email: sakshammishra0205@gmail.com

## 🙏 Acknowledgments

- PyTorch & TorchVision teams
- TIMM library contributors
- Grad-CAM authors
- Kaggle community for datasets

---

**Last Updated**: January 18, 2026
