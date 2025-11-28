#!/usr/bin/env python3
"""
Model Benchmarking Tool
Compare performance across different model sizes and precisions.
"""

import argparse
import time
from pathlib import Path
from typing import Dict, List

import torch
from ultralytics import YOLO


def benchmark_model(
    model_path: str,
    test_image: str,
    iterations: int = 100,
    warmup: int = 10,
) -> Dict[str, float]:
    """
    Benchmark model inference performance.
    
    Args:
        model_path: Path to model (.pt or .engine)
        test_image: Path to test image
        iterations: Number of inference iterations
        warmup: Number of warmup iterations
        
    Returns:
        Dictionary with benchmark metrics
    """
    
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Load model
    print(f"Loading model: {model_path.name}")
    model = YOLO(str(model_path))
    
    # Warmup
    print(f"Warming up ({warmup} iterations)...")
    for _ in range(warmup):
        _ = model(test_image, verbose=False)
    
    # Benchmark
    print(f"Benchmarking ({iterations} iterations)...")
    times = []
    
    for i in range(iterations):
        start = time.perf_counter()
        results = model(test_image, verbose=False)
        end = time.perf_counter()
        times.append(end - start)
        
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{iterations}")
    
    # Calculate metrics
    times_ms = [t * 1000 for t in times]
    avg_time = sum(times_ms) / len(times_ms)
    min_time = min(times_ms)
    max_time = max(times_ms)
    fps = 1000 / avg_time
    
    return {
        'model': model_path.name,
        'avg_ms': avg_time,
        'min_ms': min_time,
        'max_ms': max_time,
        'fps': fps,
    }


def benchmark_multiple(
    model_paths: List[str],
    test_image: str,
    iterations: int = 100,
):
    """
    Benchmark multiple models and display comparison.
    """
    
    results = []
    
    for model_path in model_paths:
        try:
            print(f"\n{'='*60}")
            result = benchmark_model(model_path, test_image, iterations)
            results.append(result)
        except Exception as e:
            print(f"ERROR benchmarking {model_path}: {e}")
    
    # Display results
    print(f"\n\n{'='*60}")
    print(f"Benchmark Results")
    print(f"{'='*60}")
    print(f"{'Model':<30} {'Avg (ms)':<12} {'FPS':<10} {'Min/Max (ms)'}")
    print(f"{'-'*60}")
    
    for r in results:
        print(f"{r['model']:<30} {r['avg_ms']:>10.2f}  {r['fps']:>8.1f}  "
              f"{r['min_ms']:.2f}/{r['max_ms']:.2f}")
    
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark model inference performance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Benchmark single model
  python benchmark.py --model outputs/trained/ppe_v1/weights/best.pt --image test.jpg
  
  # Compare multiple models
  python benchmark.py --models \\
      outputs/trained/ppe_v1_n/weights/best.pt \\
      outputs/trained/ppe_v1_s/weights/best.pt \\
      outputs/trained/ppe_v1_m/weights/best.pt \\
      --image test.jpg
        """
    )
    
    parser.add_argument('--model', help='Single model to benchmark')
    parser.add_argument('--models', nargs='+', help='Multiple models to compare')
    parser.add_argument('--image', required=True, help='Test image path')
    parser.add_argument('--iterations', type=int, default=100, 
                        help='Number of iterations')
    parser.add_argument('--warmup', type=int, default=10, 
                        help='Warmup iterations')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.model and not args.models:
        parser.error("Either --model or --models must be specified")
    
    if args.model and args.models:
        parser.error("Cannot specify both --model and --models")
    
    # Benchmark
    if args.model:
        result = benchmark_model(args.model, args.image, args.iterations, args.warmup)
        
        print(f"\n{'='*60}")
        print(f"Benchmark Result")
        print(f"{'='*60}")
        print(f"Model: {result['model']}")
        print(f"Average: {result['avg_ms']:.2f} ms")
        print(f"FPS: {result['fps']:.1f}")
        print(f"Min/Max: {result['min_ms']:.2f} / {result['max_ms']:.2f} ms")
        print(f"{'='*60}\n")
    else:
        benchmark_multiple(args.models, args.image, args.iterations)


if __name__ == '__main__':
    main()
