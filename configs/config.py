"""
Configuration file for OCD-DID Pattern Recognition
"""

import os
from pathlib import Path

# Project root
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"

# Dataset paths
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model configuration
MODEL_CONFIG = {
    'num_classes': 3,  # OCD, DID, Control
    'facial_model': 'swin_base_patch4_window7_224',
    'facial_features_dim': 1024,
    'pose_features_dim': 36,  # 18 keypoints * 2 (x, y)
    'lstm_hidden_size': 256,
    'gru_hidden_size': 128,
    'dropout': 0.4,
}

# Training configuration
TRAINING_CONFIG = {
    'batch_size': 32,
    'num_epochs': 50,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'early_stopping_patience': 10,
    'device': 'cuda',
}

# Data configuration
DATA_CONFIG = {
    'train_split': 0.7,
    'val_split': 0.15,
    'test_split': 0.15,
    'random_seed': 42,
    'image_size': (224, 224),
    'sequence_length': 30,  # 30 frames = 1 second at 30 FPS
}

# Paths
PATHS = {
    'ocd_clinical': RAW_DATA_DIR / "ocd_clinical" / "ocd_patient_data.csv",
    'fer2013': RAW_DATA_DIR / "fer2013" / "fer2013.csv",
    'pose_data': RAW_DATA_DIR / "pose_data",
    'models': RESULTS_DIR / "models",
    'figures': RESULTS_DIR / "figures",
    'logs': RESULTS_DIR / "logs",
}

# Create directories
for path in PATHS.values():
    if isinstance(path, Path):
        path.parent.mkdir(parents=True, exist_ok=True)
