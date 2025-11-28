# Multi-Platform Deployment Guide

Guide for deploying trained models to Cloud, x86 PCs, and Jetson edge devices.

---

## 🎯 Overview

The deployment pipeline follows this architecture:

```
Training (RTX 5090)
        │
        ▼
   ONNX Export (Portable)
        │
        ├──────────┬──────────┬──────────┐
        ▼          ▼          ▼          ▼
     Cloud       x86 PC    Jetson     Other
        │          │          │          │
        ▼          ▼          ▼          ▼
   Build .engine for each platform
```

**Key Principle**: ONNX is portable, TensorRT engines are NOT. Each platform must build its own optimized engine.

---

## ☁️ Cloud Deployment

### Target Hardware
- NVIDIA T4, A10, A100, L4
- High-throughput scenarios
- Multi-stream processing

### Configuration
```ini
[property]
network-mode=2  # FP16
batch-size=4    # Higher batch for throughput
```

### Deployment Steps

1. **Prepare Models**
   ```bash
   # Copy ONNX models to deployment server
   scp training/outputs/onnx/*.onnx user@cloud-server:/app/models/
   ```

2. **Build Container**
   ```bash
   docker build -t af-deploy-cloud -f docker/Dockerfile.x86 .
   ```

3. **Run Container**
   ```bash
   docker run --gpus all \
       -v $(pwd)/deployment/common/models:/app/models:ro \
       -v $(pwd)/deployment/common/labels:/app/labels:ro \
       -v $(pwd)/deployment/x86/configs:/app/configs:ro \
       -v engines:/app/engines \
       af-deploy-cloud
   ```

4. **Verify Engine Build**
   ```bash
   docker exec af-deploy-cloud ls -lh /app/engines/
   ```

### Performance Expectations
| GPU | YOLOv8s FPS | YOLOv8m FPS | YOLOv8l FPS |
|-----|-------------|-------------|-------------|
| T4  | 180+        | 120+        | 80+         |
| A10 | 300+        | 200+        | 150+        |
| A100| 500+        | 350+        | 250+        |

---

## 💻 x86 PC Deployment

### Target Hardware
- RTX 3060, 3070, 3080, 3090, 4070, 4080, 4090
- RTX 5000 series (Ada Lovelace)
- Workstation GPUs: A5000, A6000

### Configuration
```ini
[property]
network-mode=2  # FP16
batch-size=1    # Single stream for low latency
```

### Deployment Steps

1. **Install NVIDIA Container Runtime**
   ```bash
   # Ubuntu/Debian
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
       sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

2. **Deploy**
   ```bash
   cd af-training
   docker-compose --profile x86 up -d x86-deploy
   ```

3. **Check Logs**
   ```bash
   docker-compose logs x86-deploy
   ```

### Performance Expectations
| GPU     | YOLOv8s FPS | YOLOv8m FPS | YOLOv8l FPS |
|---------|-------------|-------------|-------------|
| RTX 3060| 200+        | 150+        | 100+        |
| RTX 4070| 350+        | 250+        | 180+        |
| RTX 4090| 600+        | 450+        | 350+        |

---

## 🤖 Jetson Edge Deployment

### Target Hardware
- **Jetson Orin Nano** (minimum recommended)
- Jetson Orin NX
- Jetson AGX Orin

**Note**: Jetson Nano (Maxwell) is too limited. Orin family (Ampere) is required.

### Configuration
```ini
[property]
network-mode=1  # INT8 required for performance
batch-size=1
workspace=2048  # MB, adjust based on available memory
```

### Deployment Steps

1. **Flash JetPack**
   - Minimum JetPack 6.0
   - Ensure CUDA, cuDNN, TensorRT installed

2. **Transfer Files to Jetson**
   ```bash
   # From development machine
   scp -r deployment/common/models/*.onnx jetson@192.168.1.100:/home/jetson/models/
   scp -r deployment/common/labels/*.txt jetson@192.168.1.100:/home/jetson/labels/
   scp -r deployment/common/calibration/* jetson@192.168.1.100:/home/jetson/calibration/
   scp -r deployment/edge/configs/* jetson@192.168.1.100:/home/jetson/configs/
   scp deployment/edge/scripts/build_engines.sh jetson@192.168.1.100:/home/jetson/
   ```

3. **Build Engines on Jetson**
   ```bash
   # SSH into Jetson
   ssh jetson@192.168.1.100
   
   # Build engines
   cd /home/jetson
   chmod +x build_engines.sh
   ./build_engines.sh
   ```

   This will take 5-15 minutes per model.

4. **Verify**
   ```bash
   ls -lh /home/jetson/engines/
   ```

### Jetson Docker Deployment (Alternative)

```bash
# On Jetson
docker pull nvcr.io/nvidia/deepstream-l4t:7.0-samples

docker run --runtime nvidia -it \
    -v /home/jetson/models:/app/models:ro \
    -v /home/jetson/labels:/app/labels:ro \
    -v /home/jetson/calibration:/app/calibration:ro \
    -v /home/jetson/configs:/app/configs:ro \
    -v /home/jetson/engines:/app/engines \
    nvcr.io/nvidia/deepstream-l4t:7.0-samples \
    /bin/bash
```

### Performance Expectations

| Device | YOLOv8n INT8 | YOLOv8s INT8 | YOLOv8m INT8 |
|--------|--------------|--------------|--------------|
| Orin Nano 4GB | 80-100 | 50-70 | 25-40 |
| Orin Nano 8GB | 80-100 | 50-70 | 25-40 |
| Orin NX | 120+ | 80+ | 50+ |
| AGX Orin | 200+ | 150+ | 100+ |

---

## 🔍 Deployment Verification

### Test Inference

```bash
# Create test script
cat > test_inference.py << 'EOF'
from ultralytics import YOLO

# Load TensorRT engine
model = YOLO('engines/ppe_best_fp16.engine')

# Run inference on test image
results = model('test_image.jpg')

# Print results
for r in results:
    print(f"Detected {len(r.boxes)} objects")
    for box in r.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        print(f"  Class: {cls}, Confidence: {conf:.2f}")
EOF

python test_inference.py
```

### Benchmark Performance

```bash
# Using trtexec
trtexec \
    --loadEngine=engines/ppe_best_fp16.engine \
    --iterations=100 \
    --avgRuns=100
```

### DeepStream Pipeline Test

```bash
# Simple test pipeline
deepstream-app -c configs/config_infer_ppe.txt
```

---

## 🔧 Optimization Tips

### Edge Optimization
1. **Use INT8** - 3-4x faster than FP16
2. **Reduce resolution** - 416x416 instead of 640x640 if acceptable
3. **Use smaller models** - YOLOv8n or YOLOv8s
4. **Lower frame rate** - Process every Nth frame
5. **Batch preprocessing** - If processing multiple streams

### Cloud Optimization
1. **Increase batch size** - 4-8 for throughput
2. **Multi-stream** - Process multiple videos concurrently
3. **Dynamic batching** - With Triton Inference Server
4. **FP16 precision** - Good balance
5. **Async processing** - Non-blocking inference

---

## 🐛 Troubleshooting

### Engine Build Fails

**Error**: `Out of memory during TensorRT build`

**Solution**:
```bash
# Reduce workspace
export WORKSPACE_MB=1024

# Or build with lower precision
# Use FP16 instead of INT8
```

**Error**: `Unsupported ONNX opset`

**Solution**:
```bash
# Re-export ONNX with compatible opset
python scripts/export_onnx.py --model best.pt --opset 12
```

### Slow Inference

**Issue**: FPS much lower than expected

**Checks**:
1. Verify GPU utilization: `nvidia-smi`
2. Confirm engine is used (not ONNX fallback)
3. Check precision mode matches config
4. Ensure no CPU throttling (on Jetson)

### Engine Not Found

**Error**: `Could not find engine file`

**Solution**:
```bash
# Set correct paths in DeepStream config
onnx-file=/full/path/to/model.onnx
model-engine-file=/full/path/to/model.engine

# Or set environment variables
export MODELS_DIR=/app/models
export ENGINES_DIR=/app/engines
```

---

## 📊 Multi-Platform Comparison

| Factor | Cloud | x86 PC | Jetson Edge |
|--------|-------|--------|-------------|
| **Precision** | FP16 | FP16 | INT8 |
| **Batch Size** | 4-8 | 1-2 | 1 |
| **Throughput** | Highest | High | Medium |
| **Latency** | Medium | Low | Low |
| **Power** | High (250W+) | Medium (100-300W) | Low (10-25W) |
| **Cost** | Pay-per-use | One-time hardware | One-time hardware |
| **Use Case** | Multi-stream, batch | Single/multi camera | Single camera, remote |

---

## 🚀 Production Checklist

- [ ] TensorRT engines built for target platform
- [ ] Inference tested with sample images
- [ ] Performance benchmarked (meets requirements)
- [ ] DeepStream config validated
- [ ] Error handling implemented
- [ ] Monitoring/logging configured
- [ ] Fallback mechanism tested
- [ ] Security hardening (if network-exposed)
- [ ] Documentation updated

---

## 📚 Additional Resources

- [NVIDIA DeepStream Documentation](https://docs.nvidia.com/metropolis/deepstream/dev-guide/index.html)
- [TensorRT Documentation](https://docs.nvidia.com/deepstream/dev-guide/text/DS_using_custom_model.html)
- [Jetson Developer Guide](https://docs.nvidia.com/jetson/)
- [Triton Inference Server](https://github.com/triton-inference-server) (for advanced cloud deployments)
