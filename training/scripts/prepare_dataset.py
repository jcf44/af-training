#!/usr/bin/env python3
"""
Dataset Preparation Utilities
Tools for preparing datasets for YOLO training.
"""

import argparse
import random
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import yaml


def extract_frames_from_video(
    video_path: str,
    output_dir: str,
    fps: float = 1.0,
    prefix: str = 'frame',
) -> List[Path]:
    """
    Extract frames from video at specified FPS.
    
    Args:
        video_path: Path to video file
        output_dir: Output directory for frames
        fps: Frames per second to extract
        prefix: Prefix for frame filenames
        
    Returns:
        List of extracted frame paths
    """
    
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame interval
    interval = int(video_fps / fps) if fps < video_fps else 1
    
    print(f"\n{'='*60}")
    print(f"Extracting Frames")
    print(f"{'='*60}")
    print(f"Video: {video_path.name}")
    print(f"Video FPS: {video_fps:.2f}")
    print(f"Total frames: {total_frames}")
    print(f"Extraction FPS: {fps}")
    print(f"Frame interval: {interval}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")
    
    extracted = []
    frame_idx = 0
    extracted_idx = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        if frame_idx % interval == 0:
            frame_file = output_dir / f"{prefix}_{extracted_idx:06d}.jpg"
            cv2.imwrite(str(frame_file), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            extracted.append(frame_file)
            extracted_idx += 1
            
            if extracted_idx % 100 == 0:
                print(f"Extracted {extracted_idx} frames...")
        
        frame_idx += 1
    
    cap.release()
    
    print(f"\n✓ Extracted {len(extracted)} frames")
    print(f"  Output: {output_dir}\n")
    
    return extracted


def split_dataset(
    images_dir: str,
    labels_dir: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42,
):
    """
    Split dataset into train/val/test sets.
    
    Args:
        images_dir: Directory containing images
        labels_dir: Directory containing YOLO format labels
        output_dir: Output directory for split dataset
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed for reproducibility
    """
    
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)
    
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")
    
    # Validate ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.001:
        raise ValueError("Train/val/test ratios must sum to 1.0")
    
    # Find all images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    images = [
        f for f in images_dir.iterdir()
        if f.suffix.lower() in image_extensions
    ]
    
    print(f"\n{'='*60}")
    print(f"Dataset Splitting")
    print(f"{'='*60}")
    print(f"Total images: {len(images)}")
    print(f"Train ratio: {train_ratio:.1%}")
    print(f"Val ratio: {val_ratio:.1%}")
    print(f"Test ratio: {test_ratio:.1%}")
    print(f"{'='*60}\n")
    
    # Shuffle with seed
    random.seed(seed)
    random.shuffle(images)
    
    # Calculate split indices
    n_train = int(len(images) * train_ratio)
    n_val = int(len(images) * val_ratio)
    
    train_images = images[:n_train]
    val_images = images[n_train:n_train + n_val]
    test_images = images[n_train + n_val:]
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Copy files
    def copy_split(image_list: List[Path], split_name: str):
        copied_images = 0
        copied_labels = 0
        
        for img_path in image_list:
            # Copy image
            dest_img = output_dir / split_name / 'images' / img_path.name
            shutil.copy(img_path, dest_img)
            copied_images += 1
            
            # Copy label if exists
            label_path = labels_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                dest_label = output_dir / split_name / 'labels' / label_path.name
                shutil.copy(label_path, dest_label)
                copied_labels += 1
        
        print(f"{split_name.capitalize():5s}: {copied_images:4d} images, {copied_labels:4d} labels")
        
        return copied_images, copied_labels
    
    copy_split(train_images, 'train')
    copy_split(val_images, 'val')
    copy_split(test_images, 'test')
    
    print(f"\n✓ Dataset split complete")
    print(f"  Output: {output_dir}\n")


def create_dataset_yaml(
    dataset_dir: str,
    class_names: List[str],
    output_file: str,
):
    """
    Create YOLO dataset YAML configuration.
    
    Args:
        dataset_dir: Root directory of split dataset
        class_names: List of class names
        output_file: Output YAML file path
    """
    
    dataset_dir = Path(dataset_dir).absolute()
    output_file = Path(output_file)
    
    yaml_content = {
        'path': str(dataset_dir),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': len(class_names),
        'names': class_names,
    }
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
    
    print(f"✓ Created dataset YAML: {output_file}")
    print(f"  Classes: {', '.join(class_names)}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Dataset preparation utilities',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Extract frames command
    extract_parser = subparsers.add_parser('extract', help='Extract frames from video')
    extract_parser.add_argument('--video', required=True, help='Input video file')
    extract_parser.add_argument('--output', required=True, help='Output directory')
    extract_parser.add_argument('--fps', type=float, default=1.0, help='Frames per second')
    extract_parser.add_argument('--prefix', default='frame', help='Frame filename prefix')
    
    # Split dataset command
    split_parser = subparsers.add_parser('split', help='Split dataset into train/val/test')
    split_parser.add_argument('--images', required=True, help='Images directory')
    split_parser.add_argument('--labels', required=True, help='Labels directory')
    split_parser.add_argument('--output', required=True, help='Output directory')
    split_parser.add_argument('--train', type=float, default=0.7, help='Train ratio')
    split_parser.add_argument('--val', type=float, default=0.2, help='Val ratio')
    split_parser.add_argument('--test', type=float, default=0.1, help='Test ratio')
    split_parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Create YAML command
    yaml_parser = subparsers.add_parser('yaml', help='Create dataset YAML')
    yaml_parser.add_argument('--dataset-dir', required=True, help='Dataset directory')
    yaml_parser.add_argument('--classes', required=True, nargs='+', help='Class names')
    yaml_parser.add_argument('--output', required=True, help='Output YAML file')
    
    args = parser.parse_args()
    
    if args.command == 'extract':
        extract_frames_from_video(
            video_path=args.video,
            output_dir=args.output,
            fps=args.fps,
            prefix=args.prefix,
        )
    elif args.command == 'split':
        split_dataset(
            images_dir=args.images,
            labels_dir=args.labels,
            output_dir=args.output,
            train_ratio=args.train,
            val_ratio=args.val,
            test_ratio=args.test,
            seed=args.seed,
        )
    elif args.command == 'yaml':
        create_dataset_yaml(
            dataset_dir=args.dataset_dir,
            class_names=args.classes,
            output_file=args.output,
        )
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
