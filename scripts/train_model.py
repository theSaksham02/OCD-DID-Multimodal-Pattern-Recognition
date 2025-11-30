"""
Main training script for OCD-DID multimodal classification
Usage: python scripts/train_model.py --config configs/training_config.yaml
"""

import argparse
import sys
sys.path.append('.')

from src.config import MODEL_CONFIG, TRAINING_CONFIG
from src.models.multimodal_fusion import MultimodalOCDDIDClassifier
from src.training.train import train_model
from src.data.dataset import get_dataloaders

def main(args):
    print("=== OCD-DID Pattern Recognition Training ===")
    
    # Load data
    print("Loading datasets...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=TRAINING_CONFIG['batch_size']
    )
    
    # Initialize model
    print("Initializing model...")
    model = MultimodalOCDDIDClassifier(
        num_classes=MODEL_CONFIG['num_classes']
    )
    
    # Train
    print("Starting training...")
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=TRAINING_CONFIG['num_epochs'],
        learning_rate=TRAINING_CONFIG['learning_rate']
    )
    
    print("=== Training Complete! ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/training_config.yaml')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    args = parser.parse_args()
    
    main(args)
