#!/bin/bash
# DeepStream TensorRT Engine Builder for Jetson Orin Nano
# Builds INT8 optimized engines for edge deployment

set -e

MODELS_DIR="${MODELS_DIR:-/app/models}"
ENGINES_DIR="${ENGINES_DIR:-/app/engines}"
CALIB_DIR="${CALIB_DIR:-/app/calibration}"
WORKSPACE_MB="${WORKSPACE_MB:-2048}"

echo "=========================================="
echo "TensorRT Engine Builder - Jetson Orin Nano"
echo "=========================================="
echo "Models directory: $MODELS_DIR"
echo "Engines directory: $ENGINES_DIR"
echo "Calibration directory: $CALIB_DIR"
echo "Workspace: $WORKSPACE_MB MB"
echo "=========================================="
echo ""

# Create engines directory
mkdir -p "$ENGINES_DIR"

# Check for trtexec
if ! command -v trtexec &> /dev/null; then
    echo "ERROR: trtexec not found"
    echo "Using alternative path: /usr/src/tensorrt/bin/trtexec"
    TRTEXEC="/usr/src/tensorrt/bin/trtexec"
else
    TRTEXEC="trtexec"
fi

# Build engines for each ONNX model
for onnx_file in "$MODELS_DIR"/*.onnx; do
    [ -f "$onnx_file" ] || continue
    
    base_name=$(basename "$onnx_file" .onnx)
    engine_file="$ENGINES_DIR/${base_name}_orin_int8.engine"
    
    # Skip if engine already exists
    if [ -f "$engine_file" ]; then
        echo "✓ Engine exists: $(basename $engine_file)"
        continue
    fi
    
    echo ""
    echo "Building INT8 engine for: $base_name"
    echo "----------------------------------------"
    
    # Check for calibration cache
    calib_file="$CALIB_DIR/CalibrationTable"
    
    if [ ! -f "$calib_file" ]; then
        echo "WARNING: No calibration cache found at $calib_file"
        echo "         Building without explicit calibration (may be slower)"
        calib_arg=""
    else
        echo "Using calibration cache: $calib_file"
        calib_arg="--calib=$calib_file"
    fi
    
    # Build engine with INT8 + FP16 fallback
    $TRTEXEC \
        --onnx="$onnx_file" \
        --saveEngine="$engine_file" \
        --int8 \
        --fp16 \
        --workspace=$WORKSPACE_MB \
        $calib_arg \
        --verbose \
        2>&1 | tee "$ENGINES_DIR/${base_name}_build.log"
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully built: $(basename $engine_file)"
        
        # Print engine size
        size=$(du -h "$engine_file" | cut -f1)
        echo "  Size: $size"
    else
        echo "✗ Failed to build engine for $base_name"
    fi
done

echo ""
echo "=========================================="
echo "Engine Build Complete"
echo "=========================================="
echo "Engines directory: $ENGINES_DIR"
ls -lh "$ENGINES_DIR"/*.engine 2>/dev/null || echo "No engines built"
echo "=========================================="

# If this is an entrypoint script, continue to main application
if [ $# -gt 0 ]; then
    echo ""
    echo "Executing: $@"
    exec "$@"
fi
