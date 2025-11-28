#!/usr/bin/env python3
"""
Deployment Validation Tool
Verify deployment setup is correct before going to production.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple


def check_file_exists(path: str, description: str) -> Tuple[bool, str]:
    """Check if file exists."""
    if Path(path).exists():
        return True, f"✓ {description}: {path}"
    else:
        return False, f"✗ {description} NOT FOUND: {path}"


def validate_onnx_models(models_dir: str) -> bool:
    """Validate ONNX models exist."""
    models_dir = Path(models_dir)
    
    print(f"\n{'='*60}")
    print("Validating ONNX Models")
    print(f"{'='*60}")
    
    if not models_dir.exists():
        print(f"✗ Models directory not found: {models_dir}")
        return False
    
    onnx_files = list(models_dir.glob("*.onnx"))
    
    if not onnx_files:
        print(f"✗ No ONNX files found in {models_dir}")
        return False
    
    all_valid = True
    for onnx_file in onnx_files:
        size_mb = onnx_file.stat().st_size / (1024 * 1024)
        print(f"  ✓ {onnx_file.name} ({size_mb:.1f} MB)")
    
    print(f"\n✓ Found {len(onnx_files)} ONNX model(s)")
    return all_valid


def validate_labels(labels_dir: str, expected_labels: List[str] = None) -> bool:
    """Validate label files exist."""
    labels_dir = Path(labels_dir)
    
    print(f"\n{'='*60}")
    print("Validating Label Files")
    print(f"{'='*60}")
    
    if not labels_dir.exists():
        print(f"✗ Labels directory not found: {labels_dir}")
        return False
    
    label_files = list(labels_dir.glob("*.txt"))
    
    if not label_files:
        print(f"✗ No label files found in {labels_dir}")
        return False
    
    for label_file in label_files:
        with open(label_file) as f:
            classes = [line.strip() for line in f if line.strip()]
        print(f"  ✓ {label_file.name} ({len(classes)} classes)")
        
        if expected_labels:
            for label in expected_labels:
                if label not in classes:
                    print(f"    ⚠ Warning: Expected class '{label}' not found")
    
    print(f"\n✓ Found {len(label_files)} label file(s)")
    return True


def validate_deepstream_configs(configs_dir: str) -> bool:
    """Validate DeepStream config files."""
    configs_dir = Path(configs_dir)
    
    print(f"\n{'='*60}")
    print("Validating DeepStream Configs")
    print(f"{'='*60}")
    
    if not configs_dir.exists():
        print(f"✗ Configs directory not found: {configs_dir}")
        return False
    
    config_files = list(configs_dir.glob("*.txt"))
    
    if not config_files:
        print(f"✗ No config files found in {configs_dir}")
        return False
    
    all_valid = True
    required_keys = ['onnx-file', 'model-engine-file', 'labelfile-path', 
                     'num-detected-classes', 'network-mode']
    
    for config_file in config_files:
        print(f"\n  Checking: {config_file.name}")
        
        with open(config_file) as f:
            content = f.read()
        
        for key in required_keys:
            if key in content:
                print(f"    ✓ {key}")
            else:
                print(f"    ✗ Missing: {key}")
                all_valid = False
    
    if all_valid:
        print(f"\n✓ All {len(config_files)} config(s) valid")
    else:
        print(f"\n✗ Some configs have missing keys")
    
    return all_valid


def validate_docker_setup() -> bool:
    """Validate Docker setup."""
    import subprocess
    
    print(f"\n{'='*60}")
    print("Validating Docker Setup")
    print(f"{'='*60}")
    
    # Check if Docker is installed
    try:
        result = subprocess.run(['docker', '--version'], 
                              capture_output=True, text=True)
        print(f"  ✓ Docker: {result.stdout.strip()}")
    except FileNotFoundError:
        print("  ✗ Docker not installed")
        return False
    
    # Check if NVIDIA runtime is available
    try:
        result = subprocess.run(['docker', 'run', '--rm', '--gpus', 'all', 
                               'nvidia/cuda:12.0-base', 'nvidia-smi'],
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("  ✓ NVIDIA Container Runtime working")
        else:
            print("  ✗ NVIDIA Container Runtime not working")
            print(f"    Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"  ✗ Error testing NVIDIA runtime: {e}")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Validate deployment setup',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument('--models-dir', default='deployment/common/models',
                       help='Models directory')
    parser.add_argument('--labels-dir', default='deployment/common/labels',
                       help='Labels directory')
    parser.add_argument('--configs-dir', required=True,
                       help='DeepStream configs directory')
    parser.add_argument('--skip-docker', action='store_true',
                       help='Skip Docker validation')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("AF-Training Deployment Validation")
    print(f"{'='*60}")
    
    checks = []
    
    # Validate ONNX models
    checks.append(('ONNX Models', validate_onnx_models(args.models_dir)))
    
    # Validate labels
    checks.append(('Labels', validate_labels(args.labels_dir)))
    
    # Validate DeepStream configs
    checks.append(('DeepStream Configs', validate_deepstream_configs(args.configs_dir)))
    
    # Validate Docker
    if not args.skip_docker:
        checks.append(('Docker Setup', validate_docker_setup()))
    
    # Summary
    print(f"\n\n{'='*60}")
    print("Validation Summary")
    print(f"{'='*60}")
    
    all_passed = True
    for name, passed in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:<25} {status}")
        if not passed:
            all_passed = False
    
    print(f"{'='*60}\n")
    
    if all_passed:
        print("✓ All validation checks passed!")
        print("  Deployment setup looks good.\n")
        sys.exit(0)
    else:
        print("✗ Some validation checks failed.")
        print("  Please fix the issues before deploying.\n")
        sys.exit(1)


if __name__ == '__main__':
    main()
