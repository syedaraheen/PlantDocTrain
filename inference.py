#!/usr/bin/env python3
"""
Inference script for trained PlantDoc YOLOv8 model.
This script loads a trained model and performs inference on new images.
"""

import os
import argparse
from pathlib import Path
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy as np

def load_model(model_path):
    """Load the trained YOLOv8 model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    model = YOLO(model_path)
    print(f"Loaded model from {model_path}")
    return model

def predict_single_image(model, image_path, save_results=True, output_dir="inference_results"):
    """Predict on a single image."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    # Run inference
    results = model(image_path)
    
    # Get prediction
    result = results[0]
    class_id = result.probs.top1
    confidence = result.probs.top1conf.item()
    class_name = model.names[class_id]
    
    print(f"Prediction: {class_name} (confidence: {confidence:.3f})")
    
    # Save results if requested
    if save_results:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save annotated image
        annotated_img = result.plot()
        output_path = os.path.join(output_dir, f"predicted_{os.path.basename(image_path)}")
        cv2.imwrite(output_path, annotated_img)
        print(f"Annotated image saved to {output_path}")
        
        # Save prediction text
        txt_path = os.path.join(output_dir, f"prediction_{os.path.basename(image_path)}.txt")
        with open(txt_path, 'w') as f:
            f.write(f"Image: {image_path}\n")
            f.write(f"Predicted Class: {class_name}\n")
            f.write(f"Confidence: {confidence:.3f}\n")
            f.write(f"Class ID: {class_id}\n")
    
    return {
        'class_name': class_name,
        'confidence': confidence,
        'class_id': class_id
    }

def predict_batch(model, image_dir, save_results=True, output_dir="inference_results"):
    """Predict on a batch of images."""
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Directory not found at {image_dir}")
    
    # Get all image files
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = []
    
    for file in os.listdir(image_dir):
        if file.lower().endswith(image_extensions):
            image_files.append(os.path.join(image_dir, file))
    
    if not image_files:
        print(f"No image files found in {image_dir}")
        return []
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    results = []
    for i, image_path in enumerate(image_files):
        print(f"\nProcessing {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
        try:
            result = predict_single_image(model, image_path, save_results, output_dir)
            results.append({
                'image_path': image_path,
                **result
            })
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            results.append({
                'image_path': image_path,
                'error': str(e)
            })
    
    return results

def create_prediction_summary(results, output_dir="inference_results"):
    """Create a summary of predictions."""
    if not results:
        return
    
    # Count predictions by class
    class_counts = {}
    total_confidence = 0
    valid_predictions = 0
    
    for result in results:
        if 'error' not in result:
            class_name = result['class_name']
            confidence = result['confidence']
            
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            total_confidence += confidence
            valid_predictions += 1
    
    # Create summary
    summary_path = os.path.join(output_dir, "prediction_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("PlantDoc Model Prediction Summary\n")
        f.write("="*50 + "\n")
        f.write(f"Total images processed: {len(results)}\n")
        f.write(f"Valid predictions: {valid_predictions}\n")
        f.write(f"Average confidence: {total_confidence/valid_predictions:.3f}\n\n")
        
        f.write("Predictions by class:\n")
        f.write("-" * 30 + "\n")
        for class_name, count in sorted(class_counts.items()):
            f.write(f"{class_name}: {count}\n")
        
        f.write("\nDetailed results:\n")
        f.write("-" * 30 + "\n")
        for result in results:
            if 'error' not in result:
                f.write(f"{os.path.basename(result['image_path'])}: {result['class_name']} ({result['confidence']:.3f})\n")
            else:
                f.write(f"{os.path.basename(result['image_path'])}: ERROR - {result['error']}\n")
    
    print(f"Prediction summary saved to {summary_path}")

def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='Run inference with trained PlantDoc YOLOv8 model')
    
    parser.add_argument('--model', type=str, required=True,
                       help='path to trained model weights (.pt file)')
    parser.add_argument('--image', type=str, default='',
                       help='path to single image for prediction')
    parser.add_argument('--dir', type=str, default='',
                       help='path to directory of images for batch prediction')
    parser.add_argument('--output', type=str, default='inference_results',
                       help='output directory for results')
    parser.add_argument('--no-save', action='store_true',
                       help='do not save results to disk')
    
    args = parser.parse_args()
    
    print("PlantDoc YOLOv8 Inference")
    print("="*50)
    
    # Load model
    try:
        model = load_model(args.model)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Run inference
    if args.image:
        # Single image prediction
        print(f"Predicting on single image: {args.image}")
        try:
            result = predict_single_image(
                model, 
                args.image, 
                save_results=not args.no_save,
                output_dir=args.output
            )
            print(f"Result: {result['class_name']} (confidence: {result['confidence']:.3f})")
        except Exception as e:
            print(f"Error: {e}")
    
    elif args.dir:
        # Batch prediction
        print(f"Predicting on directory: {args.dir}")
        try:
            results = predict_batch(
                model,
                args.dir,
                save_results=not args.no_save,
                output_dir=args.output
            )
            
            if not args.no_save:
                create_prediction_summary(results, args.output)
            
            print(f"\nProcessed {len(results)} images")
            print(f"Results saved to: {args.output}")
        except Exception as e:
            print(f"Error: {e}")
    
    else:
        print("Please specify either --image for single image or --dir for batch prediction")
        print("Example usage:")
        print("  python inference.py --model runs/train/plantdoc_yolov8/weights/best.pt --image test_image.jpg")
        print("  python inference.py --model runs/train/plantdoc_yolov8/weights/best.pt --dir test_images/")

if __name__ == "__main__":
    main()