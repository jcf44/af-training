# Setup Guide

Choose your preferred setup method: **Host-based** (recommended for development) or **Docker** (reproducibility).

---

## 🚀 Host-Based Setup (Recommended for Training)

### Prerequisites

- **NVIDIA GPU** with CUDA support (RTX 5090, 4090, 3090, etc.)
- **Ubuntu 22.04+** or **WSL2** on Windows
- **Python 3.10+**
- **CUDA 12.0+** installed
- **uv** (modern Python package manager) or **pip**

### Option A: Using `uv` (Modern, Fast - **Recommended**)

**Why uv?** 10-100x faster than pip, better dependency resolution.

```bash
# 1. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with pip
# pip install uv

# 2. Clone/navigate to project
cd af-training

# 3. Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows WSL2
## 1. Prerequisites

- **OS**: Linux (Ubuntu 22.04+ recommended) or Windows (WSL2)
- **GPU**: NVIDIA GPU with CUDA support
- **Python**: 3.12 (Recommended)
- **CUDA**: 12.0+

## 2. Installation (Host)

We recommend using `uv` for fast, reliable dependency management.

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create virtual environment (Python 3.12)
uv venv --python 3.12
source .venv/bin/activate

# 3. Install dependencies
# This installs torch, ultralytics, onnxruntime-gpu, tensorrt, etc.
uv pip install -e .
python -c "import ultralytics; print('✓ Ultralytics installed')"
```

### 4. Setup UI & API

The project includes a web interface for easier management.

**Prerequisites**:
- **Node.js 18+**
- **npm**

```bash
# 1. Start the API (Terminal 1)
uv run af-training-api

# 2. Setup & Start UI (Terminal 2)
cd ui
npm install
npm run dev

# 3. Access
# Open http://localhost:3000
```

### Install TensorRT (Optional, for local ONNX→TensorRT conversion)

TensorRT is optional for training but needed for local engine building.

```bash
# For Ubuntu/WSL2
# Download from: https://developer.nvidia.com/tensorrt
# Or use NVIDIA's pip package (if available for your CUDA version)

pip install tensorrt

# Verify
python -c "import tensorrt; print(f'✓ TensorRT {tensorrt.__version__}')"
```

**Note**: If TensorRT installation is complex, skip it. You can build engines on deployment targets (Jetson/cloud) instead.

---

### Verify GPU Access

```bash
# Check NVIDIA driver
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"

# Should output: CUDA: True, Devices: 1 (or more)
```

---

## 🐋 Docker-Based Setup (Reproducibility)

For perfect reproducibility or if you prefer containers.

### Prerequisites

- **Docker** with **NVIDIA Container Runtime**
- **NVIDIA GPU** drivers on host

### Setup

```bash
# 1. Install NVIDIA Container Toolkit (if not already)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# 2. Verify GPU access in Docker
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi

# 3. Build training container
cd af-training
docker-compose build training

# 4. Test
docker-compose run training python -c "import torch; print(torch.cuda.is_available())"
```

---

## 📁 Directory Setup

After installation, set up your workspace:

```bash
# Create dataset directories
mkdir -p training/datasets/{raw,processed,calibration}

# Create output directories
mkdir -p training/outputs/{trained,onnx,calibration,logs}

# Optional: symlink your data
ln -s /path/to/your/datasets training/datasets/raw/ppe
```

---

## 🎯 Quick Verification

### Host-Based

```bash
# Activate environment
source .venv/bin/activate

# Run a quick training test (1 epoch)
cd training
python scripts/train.py \
    --data configs/datasets/ppe.yaml \
    --name test_run \
    --size n \
    --epochs 1

# Should start training and complete quickly
```

### Docker-Based

```bash
docker-compose run training python scripts/train.py \
    --data configs/datasets/ppe.yaml \
    --name test_run \
    --size n \
    --epochs 1
```

---

## 💡 Recommended Workflow

### For Development & Experimentation (Daily Work)

**Use host-based training**:
```bash
source .venv/bin/activate
cd training
python scripts/train.py --data configs/datasets/ppe.yaml --name ppe_v1 --size s
```

**Benefits**:
- ⚡ Faster iteration
- 🐛 Better debugging (IDE integration)
- 📁 Direct file access
- 🎯 Simpler workflow

### For Reproducibility & Final Runs

**Use Docker**:
```bash
docker-compose run training python scripts/train.py \
    --data configs/datasets/ppe.yaml --name ppe_v1 --size s
```

**Benefits**:
- 📦 Perfect reproducibility
- 🔒 No dependency conflicts
- 👥 Same environment for team
- 🚢 Deployment-ready

---

## 🔧 Troubleshooting

### Issue: `torch.cuda.is_available()` returns False

**Check**:
```bash
# 1. NVIDIA driver
nvidia-smi

# 2. CUDA toolkit
nvcc --version

# 3. PyTorch CUDA build
python -c "import torch; print(torch.version.cuda)"
```

**Solution**: Reinstall PyTorch with correct CUDA version:
```bash
# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Issue: `uv` not found

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add to PATH (if needed)
export PATH="$HOME/.cargo/bin:$PATH"
```

### Issue: Out of memory during training

**Solution**: Reduce batch size
```bash
python scripts/train.py ... --batch 8  # or --batch 4
```

---

## 📚 Next Steps

1. **Prepare your dataset** - See [DATASETS.md](docs/DATASETS.md)
2. **Train your model** - See [README.md](README.md#training)
3. **Export to ONNX** - See [README.md](README.md#export)
4. **Deploy** - See [DEPLOYMENT.md](docs/DEPLOYMENT.md)

---

## 🎓 Learning Resources

- [Ultralytics YOLO Docs](https://docs.ultralytics.com/)
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
- [uv Documentation](https://github.com/astral-sh/uv)
