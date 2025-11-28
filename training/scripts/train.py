#!/usr/bin/env python3
"""
AF-Training: Main Training Script
Train YOLO models for DeepStream deployment across multiple domains.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from ultralytics import YOLO


def train_model(
    data_yaml: str,
    project_name: str,
    model_size: str = 's',
    epochs: int = 200,
    imgsz: int = 640,
    device: int = 0,
    batch: Optional[int] = None,
    patience: int = 50,
    resume: bool = False,
    output_dir: str = 'outputs/trained',
):
    """
    Train YOLO model optimized for multi-platform deployment.
    
    Args:
        data_yaml: Path to dataset YAML configuration
        project_name: Name for this training run
        model_size: Model size (n, s, m, l, x)
        epochs: Number of training epochs
        imgsz: Input image size
        device: GPU device ID
        batch: Batch size (auto-calculated if None)
        patience: Early stopping patience
        resume: Resume from last checkpoint
        output_dir: Output directory for trained models
    """
    
    # Model size to base architecture mapping
    model_path = f'yolov8{model_size}.pt'
    
    # Recommended batch sizes for RTX 5090 (32GB VRAM)
    # Adjust based on your GPU memory
    default_batch_sizes = {
        'n': 128,  # Nano - very light
        's': 64,   # Small - recommended for most use cases
        'm': 32,   # Medium - good accuracy/speed balance
        'l': 16,   # Large - high accuracy
        'x': 8,    # Extra large - maximum accuracy
    }
    
    if batch is None:
        batch = default_batch_sizes.get(model_size, 16)
    
    print(f"\n{'='*60}")
    print(f"Training Configuration")
    print(f"{'='*60}")
    print(f"Project: {project_name}")
    print(f"Model: YOLOv8{model_size}")
    print(f"Dataset: {data_yaml}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch}")
    print(f"Image size: {imgsz}")
    print(f"Device: cuda:{device}")
    print(f"{'='*60}\n")
    
    # Load model
    model = YOLO(model_path)
    
    # Training
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=8,
        patience=patience,
        
        # Save settings
        project=output_dir,
        name=project_name,
        exist_ok=True,
        save=True,
        save_period=10,  # Save checkpoint every 10 epochs
        
        # Optimization
        amp=True,        # Mixed precision training (faster on modern GPUs)
        cos_lr=True,     # Cosine learning rate scheduler
        
        # Data augmentation
        hsv_h=0.015,     # HSV-Hue augmentation
        hsv_s=0.7,       # HSV-Saturation augmentation
        hsv_v=0.4,       # HSV-Value augmentation
        degrees=10,      # Rotation
        translate=0.1,   # Translation
        scale=0.5,       # Scaling
        shear=0.0,       # Shear
        perspective=0.0, # Perspective
        flipud=0.0,      # Flip up-down
        fliplr=0.5,      # Flip left-right
        mosaic=1.0,      # Mosaic augmentation (YOLOv5 technique)
        mixup=0.1,       # MixUp augmentation
        copy_paste=0.0,  # Copy-paste augmentation
        
        # Resume from checkpoint
        resume=resume,
    )
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"{'='*60}")
    print(f"Best model: {output_dir}/{project_name}/weights/best.pt")
    print(f"Last model: {output_dir}/{project_name}/weights/last.pt")
    print(f"{'='*60}\n")
    
    return results


def train_multi_size(
    data_yaml: str,
    project_base: str,
    sizes: list = ['n', 's', 'm'],
    epochs: int = 200,
    device: int = 0,
):
    """
    Train multiple model sizes for different deployment targets.
    
    Useful for:
    - Edge devices (nano, small)
    - Mid-range GPUs (small, medium)
    - Server/cloud (medium, large)
    """
    
    print(f"\n{'='*60}")
    print(f"Multi-Size Training")
    print(f"{'='*60}")
    print(f"Training {len(sizes)} model sizes: {', '.join(sizes)}")
    print(f"Dataset: {data_yaml}")
    print(f"{'='*60}\n")
    
    results = {}
    
    for size in sizes:
        project_name = f"{project_base}_{size}"
        print(f"\n\n>>> Training YOLOv8{size} variant...")
        
        try:
            result = train_model(
                data_yaml=data_yaml,
                project_name=project_name,
                model_size=size,
                epochs=epochs,
                device=device,
            )
            results[size] = result
        except Exception as e:
            print(f"ERROR training {size}: {e}")
            results[size] = None
    
    # Summary
    print(f"\n\n{'='*60}")
    print(f"Multi-Size Training Summary")
    print(f"{'='*60}")
    for size, result in results.items():
        status = "✓ Success" if result else "✗ Failed"
        print(f"YOLOv8{size}: {status}")
    print(f"{'='*60}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Train YOLO models for DeepStream deployment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train single model
  python train.py --data configs/datasets/ppe.yaml --name ppe_v1 --size s --epochs 200
  
  # Train multiple sizes for different deployment targets
  python train.py --data configs/datasets/traffic.yaml --name traffic_v1 --multi --sizes n s m
  
  # Resume training
  python train.py --data configs/datasets/ppe.yaml --name ppe_v1 --size s --resume
  
  # Quick test run
  python train.py --data configs/datasets/ppe.yaml --name ppe_test --size n --epochs 10
        """
    )
    
    parser.add_argument('--data', required=True, help='Path to dataset YAML file')
    parser.add_argument('--name', required=True, help='Project name')
    parser.add_argument('--size', default='s', choices=['n', 's', 'm', 'l', 'x'],
                        help='Model size (default: s)')
    parser.add_argument('--epochs', type=int, default=200, help='Training epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Input image size')
    parser.add_argument('--batch', type=int, help='Batch size (auto if not specified)')
    parser.add_argument('--device', type=int, default=0, help='GPU device ID')
    parser.add_argument('--patience', type=int, default=50, help='Early stopping patience')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--output', default='outputs/trained', help='Output directory')
    
    # Multi-size training
    parser.add_argument('--multi', action='store_true', 
                        help='Train multiple model sizes')
    parser.add_argument('--sizes', nargs='+', default=['n', 's', 'm'],
                        choices=['n', 's', 'm', 'l', 'x'],
                        help='Model sizes for multi-size training')
    
    args = parser.parse_args()
    
    # Validate data file exists
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: Dataset YAML not found: {args.data}")
        sys.exit(1)
    
    # Create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    # Train
    if args.multi:
        train_multi_size(
            data_yaml=args.data,
            project_base=args.name,
            sizes=args.sizes,
            epochs=args.epochs,
            device=args.device,
        )
    else:
        train_model(
            data_yaml=args.data,
            project_name=args.name,
            model_size=args.size,
            epochs=args.epochs,
            imgsz=args.imgsz,
            device=args.device,
            batch=args.batch,
            patience=args.patience,
            resume=args.resume,
            output_dir=args.output,
        )


if __name__ == '__main__':
    main()
