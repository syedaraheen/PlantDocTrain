#!/usr/bin/env python3
"""
YOLOv8 Training Script for PlantDoc Dataset
This script trains a YOLOv8 model for plant disease classification.
"""

import os
import yaml
import torch
import argparse
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def setup_training_environment():
    """Setup training environment and check dependencies."""
    print("Setting up training environment...")
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    return device

def load_dataset_config(config_path):
    """Load dataset configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_model(model_size='n', num_classes=26):
    """Create YOLOv8 model for classification."""
    print(f"Creating YOLOv8-{model_size} model for {num_classes} classes...")
    
    # Load pre-trained YOLOv8 model
    model = YOLO(f'yolov8{model_size}-cls.pt')
    
    return model

def train_model(model, dataset_config, args):
    """Train the YOLOv8 model."""
    print("Starting training...")
    
    # Training arguments
    train_args = {
        'data': dataset_config['path'],
        'epochs': args.epochs,
        'imgsz': args.img_size,
        'batch': args.batch_size,
        'device': args.device,
        'workers': args.workers,
        'project': args.project,
        'name': args.name,
        'exist_ok': True,
        'patience': args.patience,
        'save': True,
        'save_period': 10,
        'cache': args.cache,
        'optimizer': args.optimizer,
        'lr0': args.lr0,
        'lrf': args.lrf,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        'warmup_epochs': args.warmup_epochs,
        'warmup_momentum': args.warmup_momentum,
        'warmup_bias_lr': args.warmup_bias_lr,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'pose': 12.0,
        'kobj': 1.0,
        'label_smoothing': 0.0,
        'nbs': 64,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.0,
        'copy_paste': 0.0,
        'auto_augment': 'randaugment',
        'erasing': 0.4,
        'crop_fraction': 1.0,
    }
    
    # Start training
    results = model.train(**train_args)
    
    return results

def evaluate_model(model, dataset_config, args):
    """Evaluate the trained model."""
    print("Evaluating model...")
    
    # Run validation
    val_results = model.val(
        data=dataset_config['path'],
        imgsz=args.img_size,
        batch=args.batch_size,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=f"{args.name}_val",
        exist_ok=True
    )
    
    return val_results

def plot_training_results(results, save_path):
    """Plot and save training results."""
    print("Plotting training results...")
    
    # Create results directory
    os.makedirs(save_path, exist_ok=True)
    
    # Plot training curves
    try:
        results.plot(save=True, save_dir=save_path)
        print(f"Training plots saved to {save_path}")
    except Exception as e:
        print(f"Error plotting results: {e}")

def create_classification_report(model, dataset_config, args):
    """Create detailed classification report."""
    print("Creating classification report...")
    
    # This would require running inference on the validation set
    # For now, we'll create a placeholder
    print("Classification report would be generated here with detailed metrics per class")

def save_model_info(model, args, save_path):
    """Save model information and configuration."""
    info_path = os.path.join(save_path, 'model_info.txt')
    
    with open(info_path, 'w') as f:
        f.write("PlantDoc YOLOv8 Training Information\n")
        f.write("="*50 + "\n")
        f.write(f"Model: YOLOv8-{args.model_size}\n")
        f.write(f"Dataset: PlantDoc\n")
        f.write(f"Classes: 26\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Image Size: {args.img_size}\n")
        f.write(f"Batch Size: {args.batch_size}\n")
        f.write(f"Device: {args.device}\n")
        f.write(f"Optimizer: {args.optimizer}\n")
        f.write(f"Learning Rate: {args.lr0}\n")
        f.write(f"Weight Decay: {args.weight_decay}\n")
        f.write(f"Project: {args.project}\n")
        f.write(f"Name: {args.name}\n")
    
    print(f"Model info saved to {info_path}")

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train YOLOv8 on PlantDoc dataset')
    
    # Dataset arguments
    parser.add_argument('--data', type=str, default='PlantDoc-YOLO/dataset.yaml',
                       help='path to dataset yaml file')
    parser.add_argument('--config', type=str, default='plantdoc_config.yaml',
                       help='path to dataset config file')
    
    # Model arguments
    parser.add_argument('--model-size', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'],
                       help='YOLOv8 model size (n, s, m, l, x)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='batch size')
    parser.add_argument('--img-size', type=int, default=224,
                       help='image size for training')
    parser.add_argument('--device', type=str, default='',
                       help='device to use (cpu, cuda, 0, 1, etc.)')
    parser.add_argument('--workers', type=int, default=8,
                       help='number of dataloader workers')
    
    # Optimization arguments
    parser.add_argument('--optimizer', type=str, default='AdamW', choices=['SGD', 'Adam', 'AdamW', 'RMSProp'],
                       help='optimizer to use')
    parser.add_argument('--lr0', type=float, default=0.01,
                       help='initial learning rate')
    parser.add_argument('--lrf', type=float, default=0.01,
                       help='final learning rate factor')
    parser.add_argument('--momentum', type=float, default=0.937,
                       help='momentum for SGD optimizer')
    parser.add_argument('--weight-decay', type=float, default=0.0005,
                       help='weight decay')
    parser.add_argument('--warmup-epochs', type=int, default=3,
                       help='warmup epochs')
    parser.add_argument('--warmup-momentum', type=float, default=0.8,
                       help='warmup momentum')
    parser.add_argument('--warmup-bias-lr', type=float, default=0.1,
                       help='warmup bias learning rate')
    
    # Training control arguments
    parser.add_argument('--patience', type=int, default=50,
                       help='early stopping patience')
    parser.add_argument('--cache', action='store_true',
                       help='cache images for faster training')
    
    # Output arguments
    parser.add_argument('--project', type=str, default='runs/train',
                       help='project name')
    parser.add_argument('--name', type=str, default='plantdoc_yolov8',
                       help='experiment name')
    parser.add_argument('--resume', type=str, default='',
                       help='resume training from checkpoint')
    
    args = parser.parse_args()
    
    print("PlantDoc YOLOv8 Training")
    print("="*50)
    
    # Setup environment
    device = setup_training_environment()
    if not args.device:
        args.device = device
    
    # Load dataset configuration
    if os.path.exists(args.config):
        dataset_config = load_dataset_config(args.config)
        print(f"Loaded dataset config from {args.config}")
    else:
        print(f"Config file {args.config} not found. Using default paths.")
        dataset_config = {
            'path': 'PlantDoc-YOLO',
            'train': 'train',
            'val': 'val',
            'nc': 26
        }
    
    # Check if dataset exists
    if not os.path.exists(dataset_config['path']):
        print(f"Dataset not found at {dataset_config['path']}")
        print("Please run prepare_dataset.py first to prepare the dataset.")
        return
    
    # Create model
    model = create_model(args.model_size, dataset_config.get('nc', 26))
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming training from {args.resume}")
        model = YOLO(args.resume)
    
    # Train model
    try:
        results = train_model(model, dataset_config, args)
        print("Training completed successfully!")
        
        # Evaluate model
        val_results = evaluate_model(model, dataset_config, args)
        
        # Save results
        save_path = os.path.join(args.project, args.name)
        plot_training_results(results, save_path)
        save_model_info(model, args, save_path)
        
        print(f"\nTraining completed!")
        print(f"Results saved to: {save_path}")
        print(f"Best model saved to: {results.save_dir}/weights/best.pt")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise

if __name__ == "__main__":
    main()