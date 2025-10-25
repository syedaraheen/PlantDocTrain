# PlantDoc YOLOv8 Training

This repository contains scripts for training a YOLOv8 model on the PlantDoc dataset for plant disease classification.

## Dataset

The PlantDoc dataset contains 26 different plant disease classes:
- Apple leaf, Apple rust leaf, Apple Scab Leaf
- Bell_pepper leaf, Bell_pepper leaf spot
- Blueberry leaf, Cherry leaf
- Corn Gray leaf spot, Corn leaf blight, Corn rust leaf
- grape leaf, grape leaf black rot
- Peach leaf
- Potato leaf early blight, Potato leaf late blight
- Raspberry leaf, Soyabean leaf
- Squash Powdery mildew leaf, Strawberry leaf
- Tomato Early blight leaf, Tomato leaf, Tomato leaf bacterial spot
- Tomato leaf late blight, Tomato leaf mosaic virus, Tomato leaf yellow virus
- Tomato mold leaf, Tomato Septoria leaf spot

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Install YOLOv8:
```bash
pip install ultralytics
```

## Usage

### 1. Prepare Dataset

First, prepare the dataset for YOLOv8 training:

```bash
python prepare_dataset.py
```

This script will:
- Create a YOLO-compatible directory structure
- Split the dataset into training (80%) and validation (20%) sets
- Create a `dataset.yaml` configuration file
- Print dataset statistics

### 2. Train the Model

Train a YOLOv8 model on the PlantDoc dataset:

```bash
# Basic training with default parameters
python train_plantdoc.py

# Custom training with specific parameters
python train_plantdoc.py \
    --epochs 200 \
    --batch-size 32 \
    --img-size 224 \
    --model-size s \
    --device cuda \
    --project runs/train \
    --name plantdoc_yolov8s
```

### 3. Training Parameters

Key training parameters:

- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Batch size (default: 16)
- `--img-size`: Image size for training (default: 224)
- `--model-size`: YOLOv8 model size - n, s, m, l, x (default: n)
- `--device`: Device to use - cpu, cuda, 0, 1, etc. (default: auto)
- `--optimizer`: Optimizer - SGD, Adam, AdamW, RMSProp (default: AdamW)
- `--lr0`: Initial learning rate (default: 0.01)
- `--patience`: Early stopping patience (default: 50)

### 4. Model Sizes

Available YOLOv8 model sizes:
- `n`: Nano (fastest, least accurate)
- `s`: Small
- `m`: Medium
- `l`: Large
- `x`: Extra Large (slowest, most accurate)

### 5. Resume Training

To resume training from a checkpoint:

```bash
python train_plantdoc.py --resume runs/train/plantdoc_yolov8/weights/last.pt
```

## Output

The training script will create:
- `runs/train/plantdoc_yolov8/`: Training results directory
- `weights/best.pt`: Best model weights
- `weights/last.pt`: Last epoch weights
- Training plots and metrics
- Model information file

## Monitoring Training

Training progress can be monitored through:
- Console output showing loss and metrics
- TensorBoard logs (if available)
- Training plots saved in the results directory

## Example Training Commands

```bash
# Quick training with nano model
python train_plantdoc.py --epochs 50 --model-size n --batch-size 32

# High-accuracy training with large model
python train_plantdoc.py --epochs 300 --model-size l --batch-size 16 --img-size 416

# GPU training with custom settings
python train_plantdoc.py --device 0 --epochs 200 --batch-size 64 --lr0 0.001
```

## Troubleshooting

1. **CUDA out of memory**: Reduce batch size or image size
2. **Dataset not found**: Run `prepare_dataset.py` first
3. **Slow training**: Use smaller model size or reduce image size
4. **Poor accuracy**: Increase epochs, use larger model, or adjust learning rate

## File Structure

```
PlantDocTrain/
├── PlantDoc-Dataset/          # Original dataset
├── PlantDoc-YOLO/            # Prepared YOLO dataset
├── prepare_dataset.py        # Dataset preparation script
├── train_plantdoc.py         # Training script
├── plantdoc_config.yaml      # Dataset configuration
├── requirements.txt          # Dependencies
└── README.md                 # This file
```# PlantDocTrain
# PlantDocTrain
# PlantDocTrain
