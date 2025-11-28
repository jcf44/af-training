#!/usr/bin/env python3
"""
INT8 Calibration Cache Generator
Generate calibration cache for INT8 quantization on training machine.
The cache can be transferred to edge devices for faster engine building.
"""

import argparse
import shutil
import sys
from pathlib import Path

from ultralytics import YOLO


def generate_calibration_cache(
    model_path: str,
    calibration_data: str,
    output_dir: str,
    imgsz: int = 640,
    workspace: int = 8,
) -> Path:
    """
    Generate INT8 calibration cache.
    
    This builds a TensorRT INT8 engine on the training machine,
    which creates a calibration cache file. The cache can be
    transferred to edge devices to speed up INT8 engine building.
    
    Args:
        model_path: Path to .pt model file
        calibration_data: Path to calibration dataset YAML
        output_dir: Output directory for calibration cache
        imgsz: Input image size
        workspace: TensorRT workspace size in GB
        
    Returns:
        Path to calibration cache file
    """
    
    model_path = Path(model_path)
    calibration_data = Path(calibration_data)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    if not calibration_data.exists():
        raise FileNotFoundError(f"Calibration data not found: {calibration_data}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"INT8 Calibration Cache Generation")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Calibration data: {calibration_data}")
    print(f"Image size: {imgsz}")
    print(f"Workspace: {workspace} GB")
    print(f"{'='*60}\n")
    
    print("NOTE: This will build a TensorRT engine for calibration.")
    print("      The engine itself is device-specific and won't be used.")
    print("      Only the calibration.cache file is portable.\n")
    
    # Load model
    model = YOLO(str(model_path))
    
    # Build INT8 engine (we only want the calibration cache)
    print("Building INT8 engine for calibration...")
    print("This may take 5-15 minutes...\n")
    
    try:
        model.export(
            format='engine',
            device=0,
            int8=True,
            data=str(calibration_data),
            imgsz=imgsz,
            workspace=workspace,
            verbose=True,
        )
    except Exception as e:
        print(f"ERROR during calibration: {e}")
        print("\nTip: Ensure you have representative calibration images")
        print("     and that your GPU has enough memory.")
        raise
    
    # Find and copy calibration cache
    # Ultralytics creates it in the same directory as the model
    model_dir = model_path.parent
    cache_files = list(model_dir.glob('*.cache'))
    
    if not cache_files:
        raise FileNotFoundError("Calibration cache not generated")
    
    # Copy cache files to output directory
    copied_files = []
    for cache_file in cache_files:
        dest = output_dir / cache_file.name
        shutil.copy(cache_file, dest)
        copied_files.append(dest)
        print(f"✓ Saved calibration cache: {dest}")
    
    print(f"\n{'='*60}")
    print(f"Calibration Complete!")
    print(f"{'='*60}")
    print(f"Cache files: {len(copied_files)}")
    print(f"Output directory: {output_dir}")
    print(f"\nThese cache files can be copied to edge devices to speed up")
    print(f"INT8 engine building on Jetson devices.")
    print(f"{'='*60}\n")
    
    return copied_files[0] if copied_files else None


def main():
    parser = argparse.ArgumentParser(
        description='Generate INT8 calibration cache for edge deployment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate calibration cache
  python generate_calibration.py \\
      --model outputs/trained/ppe_v1/weights/best.pt \\
      --data configs/calibration/ppe_calib.yaml \\
      --output outputs/calibration

  # With custom image size
  python generate_calibration.py \\
      --model outputs/trained/traffic_v1/weights/best.pt \\
      --data configs/calibration/traffic_calib.yaml \\
      --output outputs/calibration \\
      --imgsz 1280

Note: This requires a CUDA-capable GPU and TensorRT installed.
      The process builds a TensorRT INT8 engine to generate the
      calibration cache. The engine itself is discarded; only the
      portable .cache file is kept.
        """
    )
    
    parser.add_argument('--model', required=True, 
                        help='Path to trained .pt model')
    parser.add_argument('--data', required=True, 
                        help='Path to calibration dataset YAML')
    parser.add_argument('--output', default='outputs/calibration', 
                        help='Output directory for cache files')
    parser.add_argument('--imgsz', type=int, default=640, 
                        help='Input image size')
    parser.add_argument('--workspace', type=int, default=8, 
                        help='TensorRT workspace size (GB)')
    
    args = parser.parse_args()
    
    try:
        generate_calibration_cache(
            model_path=args.model,
            calibration_data=args.data,
            output_dir=args.output,
            imgsz=args.imgsz,
            workspace=args.workspace,
        )
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
