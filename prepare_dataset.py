#!/usr/bin/env python3
"""
Data preparation script for PlantDoc dataset to work with YOLOv8 classification.
This script organizes the dataset and creates necessary directory structure.
"""

import os
import shutil
import yaml
from pathlib import Path
import random
from collections import defaultdict

def count_images_in_directory(directory):
    """Count images in a directory and its subdirectories."""
    count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                count += 1
    return count

def get_class_names(dataset_path):
    """Extract class names from directory structure."""
    train_path = os.path.join(dataset_path, 'train')
    class_names = []
    
    if os.path.exists(train_path):
        for item in sorted(os.listdir(train_path)):
            if os.path.isdir(os.path.join(train_path, item)):
                class_names.append(item)
    
    return class_names

def create_yolo_structure(dataset_path, output_path):
    """Create YOLO-compatible directory structure."""
    print("Creating YOLO-compatible directory structure...")
    
    # Create output directories
    os.makedirs(os.path.join(output_path, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'val'), exist_ok=True)
    
    # Get class names
    class_names = get_class_names(dataset_path)
    print(f"Found {len(class_names)} classes: {class_names}")
    
    # Create class directories in train and val
    for class_name in class_names:
        os.makedirs(os.path.join(output_path, 'train', class_name), exist_ok=True)
        os.makedirs(os.path.join(output_path, 'val', class_name), exist_ok=True)
    
    return class_names

def copy_images_with_split(dataset_path, output_path, train_split=0.8):
    """Copy images from original dataset to YOLO structure with train/val split."""
    print("Copying images with train/validation split...")
    
    train_path = os.path.join(dataset_path, 'train')
    val_path = os.path.join(dataset_path, 'val')
    
    # Process training images
    if os.path.exists(train_path):
        for class_name in os.listdir(train_path):
            class_dir = os.path.join(train_path, class_name)
            if os.path.isdir(class_dir):
                images = [f for f in os.listdir(class_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                
                # Split into train and val
                random.shuffle(images)
                split_idx = int(len(images) * train_split)
                train_images = images[:split_idx]
                val_images = images[split_idx:]
                
                # Copy training images
                for img in train_images:
                    src = os.path.join(class_dir, img)
                    dst = os.path.join(output_path, 'train', class_name, img)
                    shutil.copy2(src, dst)
                
                # Copy validation images
                for img in val_images:
                    src = os.path.join(class_dir, img)
                    dst = os.path.join(output_path, 'val', class_name, img)
                    shutil.copy2(src, dst)
                
                print(f"Class '{class_name}': {len(train_images)} train, {len(val_images)} val")
    
    # Process validation images (if they exist separately)
    if os.path.exists(val_path):
        for class_name in os.listdir(val_path):
            class_dir = os.path.join(val_path, class_name)
            if os.path.isdir(class_dir):
                images = [f for f in os.listdir(class_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                
                # Add to validation set
                for img in images:
                    src = os.path.join(class_dir, img)
                    dst = os.path.join(output_path, 'val', class_name, img)
                    shutil.copy2(src, dst)
                
                print(f"Additional val images for '{class_name}': {len(images)}")

def create_dataset_yaml(output_path, class_names):
    """Create dataset.yaml file for YOLOv8."""
    yaml_content = {
        'path': str(output_path),
        'train': 'train',
        'val': 'val',
        'nc': len(class_names),
        'names': {i: name for i, name in enumerate(class_names)}
    }
    
    yaml_path = os.path.join(output_path, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print(f"Created dataset.yaml at {yaml_path}")
    return yaml_path

def print_dataset_statistics(output_path):
    """Print dataset statistics."""
    print("\n" + "="*50)
    print("DATASET STATISTICS")
    print("="*50)
    
    train_path = os.path.join(output_path, 'train')
    val_path = os.path.join(output_path, 'val')
    
    total_train = 0
    total_val = 0
    
    print("\nTraining set:")
    for class_name in sorted(os.listdir(train_path)):
        if os.path.isdir(os.path.join(train_path, class_name)):
            count = len([f for f in os.listdir(os.path.join(train_path, class_name)) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
            total_train += count
            print(f"  {class_name}: {count} images")
    
    print(f"\nTotal training images: {total_train}")
    
    print("\nValidation set:")
    for class_name in sorted(os.listdir(val_path)):
        if os.path.isdir(os.path.join(val_path, class_name)):
            count = len([f for f in os.listdir(os.path.join(val_path, class_name)) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
            total_val += count
            print(f"  {class_name}: {count} images")
    
    print(f"\nTotal validation images: {total_val}")
    print(f"Total images: {total_train + total_val}")
    print("="*50)

def main():
    """Main function to prepare the dataset."""
    # Set random seed for reproducibility
    random.seed(42)
    
    # Paths
    dataset_path = "/home/raheen/PlantDocTrain/PlantDoc-Dataset"
    output_path = "/home/raheen/PlantDocTrain/PlantDoc-YOLO"
    
    print("PlantDoc Dataset Preparation for YOLOv8")
    print("="*50)
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return
    
    # Create output directory
    if os.path.exists(output_path):
        print(f"Output directory {output_path} already exists. Removing...")
        shutil.rmtree(output_path)
    
    os.makedirs(output_path, exist_ok=True)
    
    # Create YOLO structure
    class_names = create_yolo_structure(dataset_path, output_path)
    
    # Copy images with train/val split
    copy_images_with_split(dataset_path, output_path, train_split=0.8)
    
    # Create dataset.yaml
    yaml_path = create_dataset_yaml(output_path, class_names)
    
    # Print statistics
    print_dataset_statistics(output_path)
    
    print(f"\nDataset preparation complete!")
    print(f"YOLO dataset saved to: {output_path}")
    print(f"Dataset config saved to: {yaml_path}")

if __name__ == "__main__":
    main()