#!/usr/bin/env python3
"""
ONNX Export Script
Export trained YOLO models to ONNX format for cross-platform deployment.
"""

import argparse
import shutil
import sys
from pathlib import Path
from typing import Optional

from ultralytics import YOLO


def export_onnx(
    model_path: str,
    output_dir: str,
    imgsz: int = 640,
    opset: int = 12,
    simplify: bool = True,
    dynamic: bool = False,
    half: bool = False,
) -> Path:
    """
    Export trained YOLO model to ONNX format.
    
    Args:
        model_path: Path to .pt model file
        output_dir: Output directory for ONNX file
        imgsz: Input image size
        opset: ONNX opset version (12 recommended for TensorRT compatibility)
        simplify: Simplify ONNX graph
        dynamic: Dynamic input shapes (not recommended for edge)
        half: Export as FP16
        
    Returns:
        Path to exported ONNX file
    """
    
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"ONNX Export")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Image size: {imgsz}")
    print(f"ONNX opset: {opset}")
    print(f"Simplify: {simplify}")
    print(f"Dynamic shapes: {dynamic}")
    print(f"Half precision: {half}")
    print(f"{'='*60}\n")
    
    # Load model
    model = YOLO(str(model_path))
    
    # Export to ONNX
    exported_path = model.export(
        format='onnx',
        imgsz=imgsz,
        simplify=simplify,
        dynamic=dynamic,
        opset=opset,
        half=half,
    )
    
    # Determine output filename
    # Extract project name from path (e.g., outputs/trained/ppe_v1/weights/best.pt -> ppe_v1)
    if 'weights' in model_path.parts:
        project_name = model_path.parts[-3]
        model_type = model_path.stem  # best or last
        final_name = f"{project_name}_{model_type}.onnx"
    else:
        final_name = model_path.stem + ".onnx"
    
    final_path = output_dir / final_name
    
    # Copy to output directory with clean name
    shutil.copy(exported_path, final_path)
    
    # Get file size
    size_mb = final_path.stat().st_size / (1024 * 1024)
    
    print(f"\n{'='*60}")
    print(f"Export Complete!")
    print(f"{'='*60}")
    print(f"Output: {final_path}")
    print(f"Size: {size_mb:.2f} MB")
    print(f"{'='*60}\n")
    
    return final_path


def export_all_models(
    trained_dir: str,
    output_dir: str,
    model_type: str = 'best',
    **export_kwargs,
):
    """
    Export all trained models in a directory to ONNX.
    
    Args:
        trained_dir: Directory containing trained model folders
        output_dir: Output directory for ONNX files
        model_type: Which model to export ('best' or 'last')
        **export_kwargs: Additional arguments for export_onnx
    """
    
    trained_dir = Path(trained_dir)
    if not trained_dir.exists():
        print(f"ERROR: Training directory not found: {trained_dir}")
        return
    
    exported_count = 0
    failed_count = 0
    
    print(f"\n{'='*60}")
    print(f"Batch ONNX Export")
    print(f"{'='*60}")
    print(f"Searching: {trained_dir}")
    print(f"Model type: {model_type}")
    print(f"{'='*60}\n")
    
    # Find all model directories
    for model_dir in sorted(trained_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        
        model_file = model_dir / 'weights' / f'{model_type}.pt'
        
        if not model_file.exists():
            print(f"⊘ Skipping {model_dir.name} (no {model_type}.pt found)")
            continue
        
        print(f"\n>>> Exporting {model_dir.name}...")
        
        try:
            export_onnx(
                model_path=str(model_file),
                output_dir=output_dir,
                **export_kwargs
            )
            exported_count += 1
        except Exception as e:
            print(f"✗ ERROR exporting {model_dir.name}: {e}")
            failed_count += 1
    
    # Summary
    print(f"\n\n{'='*60}")
    print(f"Batch Export Summary")
    print(f"{'='*60}")
    print(f"✓ Exported: {exported_count}")
    print(f"✗ Failed: {failed_count}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Export trained YOLO models to ONNX',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export single model
  python export_onnx.py --model outputs/trained/ppe_v1/weights/best.pt --output outputs/onnx
  
  # Export all trained models
  python export_onnx.py --all --input outputs/trained --output outputs/onnx
  
  # Export with dynamic shapes (not recommended for edge)
  python export_onnx.py --model outputs/trained/ppe_v1/weights/best.pt --output outputs/onnx --dynamic
  
  # Export with FP16
  python export_onnx.py --model outputs/trained/ppe_v1/weights/best.pt --output outputs/onnx --half
        """
    )
    
    parser.add_argument('--model', help='Path to specific .pt file')
    parser.add_argument('--all', action='store_true', 
                        help='Export all models in input directory')
    parser.add_argument('--input', default='outputs/trained', 
                        help='Input directory for batch export')
    parser.add_argument('--output', default='outputs/onnx', 
                        help='Output directory')
    parser.add_argument('--imgsz', type=int, default=640, 
                        help='Input image size')
    parser.add_argument('--opset', type=int, default=12, 
                        help='ONNX opset version')
    parser.add_argument('--no-simplify', action='store_true', 
                        help='Disable ONNX graph simplification')
    parser.add_argument('--dynamic', action='store_true', 
                        help='Enable dynamic input shapes')
    parser.add_argument('--half', action='store_true', 
                        help='Export as FP16')
    parser.add_argument('--model-type', default='best', choices=['best', 'last'],
                        help='Which model to export (for batch mode)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.all and not args.model:
        parser.error("Either --model or --all must be specified")
    
    if args.all and args.model:
        parser.error("Cannot specify both --model and --all")
    
    # Export kwargs
    export_kwargs = {
        'imgsz': args.imgsz,
        'opset': args.opset,
        'simplify': not args.no_simplify,
        'dynamic': args.dynamic,
        'half': args.half,
    }
    
    # Export
    if args.all:
        export_all_models(
            trained_dir=args.input,
            output_dir=args.output,
            model_type=args.model_type,
            **export_kwargs
        )
    else:
        export_onnx(
            model_path=args.model,
            output_dir=args.output,
            **export_kwargs
        )


if __name__ == '__main__':
    main()
