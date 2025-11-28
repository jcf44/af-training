# AF-Training: ML Training Pipeline for DeepStream

A comprehensive machine learning training and deployment pipeline optimized for NVIDIA DeepStream across multiple hardware platforms (Cloud, Jetson Orin Nano, x86 PCs).

## 🎯 Overview

Train YOLO models on your GPU (RTX 5090, 4090, etc.), export to ONNX, and deploy anywhere with platform-specific TensorRT optimization.

**Supported Domains:**
- 🦺 **PPE Detection** - Helmets, vests, harness for safety compliance
- 🚗 **Traffic Monitoring** - Vehicles, pedestrians, traffic analysis
- 🦌 **Wildlife** - Camera trap and conservation monitoring
- 🏗️ **Industrial/Mining** - Heavy equipment tracking

**Supported Platforms:**
- ☁️ **Cloud** - High-throughput GPU servers (FP16)
- 💻 **x86 PCs** - Workstation deployments (FP16)
- 🤖 **Jetson Orin Nano** - Edge devices (INT8)

**Key Features:**
- 🖥️ **Web UI** - Full dashboard for training, monitoring, and management
- 📡 **Real-time Logs** - Live streaming logs for long-running jobs
- ⚡ **INT8 Calibration** - Auto-generate calibration caches for edge deployment
- 📦 **Deployment Bundles** - One-click download of ONNX + Config + Labels + Cache
- 🐳 **Docker & Host Support** - Flexible environment setup

---

## 🚀 Quick Start

### Setup (Choose One)

**Option A: Host-based (Recommended for Training)**
```bash
# Install with uv (modern, fast)
uv venv && source .venv/bin/activate
uv pip install -e .

# Verify
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

**Option B: Docker (Reproducibility)**
```bash
docker-compose build training

```

### Training & Deployment (UI Workflow)

1.  **Start the UI**:
    ```bash
    # Terminal 1: API
    uv run af-training-api
    
    # Terminal 2: UI
    cd ui && npm run dev
    ```
    Open [http://localhost:3000](http://localhost:3000).

2.  **Train**: Go to **Training**, select dataset and model size, click **Start Training**.
3.  **Export**: Go to **Models**, click **Export ONNX**.
4.  **Calibrate**: Click **Calibrate INT8** (auto-generates config).
5.  **Bundle**: Click **Bundle** to download a ZIP with everything needed for deployment.

### CLI Workflow (Alternative)

**Host (Recommended)**:
```bash
source .venv/bin/activate
cd training

# Train model
python scripts/train.py --data configs/datasets/ppe.yaml --name ppe_v1 --size s

# Export to ONNX
python scripts/export_onnx.py --all

# Generate Calibration
python scripts/generate_calibration.py --model outputs/trained/ppe_v1/weights/best.pt --data configs/calibration/ppe_calib.yaml --output outputs/calibration
```

**Docker**:
```bash
docker-compose run training python scripts/train.py --data configs/datasets/ppe.yaml --name ppe_v1 --size s
```

### Deployment

**Using the Bundle:**
1.  Download the **Deployment Bundle** from the UI.
2.  Upload/Extract to your target device.
3.  Run with DeepStream (using the included `config_infer_primary.txt`).

---

## 📁 Project Structure

```
af-training/
├── training/          # Training scripts & configs
│   ├── scripts/      # Train, export, calibration, dataset prep
│   ├── configs/      # Dataset & training configurations
│   ├── datasets/     # Your datasets (raw, processed, calibration)
│   └── outputs/      # Trained models, ONNX exports, logs
├── deployment/        # Platform-specific deployment
│   ├── common/       # Shared models, labels, configs
│   ├── edge/         # Jetson Orin Nano configs
│   ├── x86/          # PC/cloud configs
│   └── cloud/        # Cloud-specific
├── docker/           # Dockerfiles (train, edge, x86)
├── docs/             # 📚 Comprehensive documentation
│   ├── INDEX.md      # Documentation index (START HERE)
│   ├── QUICKSTART.md # Quick reference
│   ├── SETUP.md      # Environment setup
│   ├── DATASETS.md   # Dataset preparation
│   └── DEPLOYMENT.md # Multi-platform deployment
├── tools/            # Benchmarking & validation scripts
├── pyproject.toml    # Python project config (uv)
├── requirements.txt  # Pip dependencies
├── docker-compose.yml # Container orchestration
└── README.md         # This file
```

**📖 Documentation Index**: [docs/INDEX.md](docs/INDEX.md)

---

## 📚 Complete Workflows

### Setup Your Environment

**See [SETUP.md](SETUP.md) for detailed setup instructions.**

Quick setup:
```bash
# With uv (recommended)
uv venv && source .venv/bin/activate
uv pip install -e .

# Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Training a New Model

1. **Prepare Dataset**
   ```bash
   # Activate environment
   source .venv/bin/activate
   cd training
   
   # Extract frames from video
   python scripts/prepare_dataset.py extract \
       --video footage.mp4 \
       --output datasets/raw/ppe/images \
       --fps 1

   # Annotate with CVAT, Label Studio, or Roboflow
   # Export in YOLO format

   # Split into train/val/test
   python scripts/prepare_dataset.py split \
       --images datasets/raw/ppe/images \
       --labels datasets/raw/ppe/labels \
       --output datasets/processed/ppe

   # Create dataset YAML
   python scripts/prepare_dataset.py yaml \
       --dataset-dir datasets/processed/ppe \
       --classes person helmet no_helmet vest no_vest harness \
       --output configs/datasets/ppe.yaml
   ```

2. **Train Model**
   ```bash
   # Single model
   python scripts/train.py \
       --data configs/datasets/ppe.yaml \
       --name ppe_v1 \
       --size s \
       --epochs 200

   # Multiple sizes for different deployment targets
   python scripts/train.py \
       --data configs/datasets/ppe.yaml \
       --name ppe_v1 \
       --multi \
       --sizes n s m
   ```

3. **Export to ONNX**
   ```bash
   # Single model
   python scripts/export_onnx.py \
       --model outputs/trained/ppe_v1/weights/best.pt \
       --output outputs/onnx

   # Batch export all
   python scripts/export_onnx.py --all
   ```

4. **Generate INT8 Calibration** (optional, for edge)
   ```bash
   python scripts/generate_calibration.py \
       --model outputs/trained/ppe_v1_s/weights/best.pt \
       --data configs/calibration/ppe_calib.yaml \
       --output outputs/calibration
   ```

### Deploying to Edge (Jetson Orin Nano)

```bash
# 1. Copy files to Jetson
scp training/outputs/onnx/ppe_v1_best.onnx jetson@192.168.1.100:/home/jetson/models/
scp deployment/common/labels/ppe.txt jetson@192.168.1.100:/home/jetson/labels/

# 2. On Jetson, build TensorRT engine
ssh jetson@192.168.1.100
cd /home/jetson
./build_engines.sh

# Or use Docker
docker run --runtime nvidia -v $(pwd):/app af-deploy-edge:latest
```

### Deploying to x86/Cloud

```bash
# Using docker-compose
docker-compose --profile x86 up x86-deploy

# Or manual
docker build -t af-deploy -f docker/Dockerfile.x86 .
docker run --runtime nvidia -v $(pwd)/deployment:/app af-deploy
```

---

## 🔧 Configuration

### Model Size Selection

| Model | Parameters | Speed | Accuracy | Use Case |
|-------|-----------|-------|----------|----------|
| YOLOv8n | 3.2M | Fastest | Good | Ultra-low-power edge |
| YOLOv8s | 11.2M | Fast | Better | **Recommended for most cases** |
| YOLOv8m | 25.9M | Medium | Great | Server/cloud |
| YOLOv8l | 43.7M | Slow | Excellent | High-accuracy requirements |

### Precision Selection

| Precision | Speed | Accuracy | Memory | Use Case |
|-----------|-------|----------|--------|----------|
| **FP32** | 1x | 100% | 4x | Training only |
| **FP16** | ~2x | 99.9% | 2x | **Recommended for cloud/x86** |
| **INT8** | ~4x | 98-99% | 1x | **Required for edge performance** |

---

## 🎯 Domain-Specific Guidelines

### PPE Detection
- **Model**: YOLOv8s
- **Precision**: INT8 (edge) or FP16 (cloud)
- **Datasets**: Roboflow PPE, Kaggle Safety Helmet
- **Tips**: Ensure diverse lighting conditions, various distances

### Traffic Monitoring
- **Model**: YOLOv8s or fine-tune NVIDIA TrafficCamNet
- **Precision**: FP16
- **Datasets**: BDD100K, COCO vehicles subset
- **Tips**: Include day/night, various weather conditions

### Wildlife
- **Model**: YOLOv8m (better for distant/small animals)
- **Precision**: FP16
- **Datasets**: Microsoft MegaDetector, LILA BC, iNaturalist
- **Tips**: Consider MegaDetector as first-stage detector

### Industrial/Mining
- **Model**: YOLOv8s (equipment is large/distinct)
- **Precision**: FP16 or INT8 depending on camera count
- **Datasets**: Custom annotation required
- **Tips**: Include dust/weather effects, partial occlusions

---

## 🐋 Docker Commands

```bash
# Build all containers
docker-compose build

# Training
docker-compose run training /bin/bash

# Edge deployment
docker-compose --profile edge up edge-deploy

# x86/Cloud deployment
docker-compose --profile x86 up x86-deploy

# Clean up
docker-compose down -v
```

---

## 📊 Performance Expectations

### Jetson Orin Nano (INT8)
- YOLOv8n: 80-100 FPS
- YOLOv8s: 50-70 FPS
- YOLOv8m: 25-40 FPS

### RTX 3060 (FP16)
- YOLOv8s: 200+ FPS
- YOLOv8m: 150+ FPS
- YOLOv8l: 100+ FPS

### Cloud T4 (FP16)
- YOLOv8s: 180+ FPS
- YOLOv8m: 120+ FPS

---

## 📖 Documentation

- [CONVERSATION_ANALYSIS.md](docs/CONVERSATION_ANALYSIS.md) - Analysis of the original discussion
- [DATASETS.md](docs/DATASETS.md) - Dataset preparation guide
- [DEPLOYMENT.md](docs/DEPLOYMENT.md) - Multi-platform deployment guide

---

## 🛠️ Requirements

### Training (Host-Based)
- **NVIDIA GPU** with CUDA support (RTX 5090, 4090, 3090, etc.)
- **Python 3.10+**
- **CUDA 12.0+** and drivers
- **Ubuntu 22.04+** or **WSL2** on Windows
- **16GB+ RAM** (32GB+ recommended for large datasets)

### Training (Docker-Based)
- **NVIDIA GPU** with CUDA support
- **Docker** with NVIDIA Container Runtime
- **32GB+ RAM** recommended

### Deployment
- **Edge**: Jetson Orin Nano with JetPack 6.0+
- **x86**: NVIDIA GPU with CUDA 12.0+
- **Cloud**: Any CUDA-capable GPU

---

## 💡 Tips & Best Practices

### Environment & Workflow
1. **Use host-based training** for development (faster iteration, better debugging)
2. **Use Docker** for final runs and reproducibility
3. **Use `uv`** for package management (10-100x faster than pip)

### Training
1. **Start with YOLOv8s** - Best balance for most use cases
2. **Train on FP32, deploy on FP16/INT8** - Let training use full precision
3. **Use INT8 only for edge** - FP16 is usually sufficient for cloud/x86
4. **Calibrate with 500-1000 images** - Enough for good INT8 accuracy
5. **ONNX is portable, engines are not** - Always build engines on target device
6. **Batch export models** - Train multiple sizes, choose best for deployment

---

## 🐛 Troubleshooting

### Training OOM (Out of Memory)
- Reduce batch size: `--batch 16` or `--batch 8`
- Use smaller model: `--size s` instead of `--size m`
- Reduce image size: `--imgsz 416`

### Engine Build Fails on Jetson
- Reduce workspace: `WORKSPACE_MB=1024`
- Ensure calibration cache exists
- Check JetPack version (requires 6.0+)

### Slow Inference
- Verify INT8 mode is enabled (edge)
- Check if engine is actually being used (not ONNX fallback)
- Confirm GPU is being utilized: `nvidia-smi`


---

## 📚 Documentation

This README provides an overview and quick start. For comprehensive documentation:

**→ [docs/INDEX.md](docs/INDEX.md) - Complete Documentation Index**

Individual guides:
- **[docs/QUICKSTART.md](docs/QUICKSTART.md)** - Quick reference
- **[docs/SETUP.md](docs/SETUP.md)** - Environment setup
- **[docs/DATASETS.md](docs/DATASETS.md)** - Dataset preparation  
- **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)** - Multi-platform deployment
- **[docs/CONVERSATION_ANALYSIS.md](docs/CONVERSATION_ANALYSIS.md)** - Project background

---

## 📝 License

This project is for educational and commercial use. Models trained with this pipeline can be deployed freely.

---

## 🙏 Acknowledgments

- **NVIDIA** - DeepStream, TensorRT, TAO Toolkit
- **Ultralytics** - YOLO implementation
- **Community** - Open datasets and annotation tools
