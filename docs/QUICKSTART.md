# AF-Training: Quick Start

Complete ML training and deployment pipeline for NVIDIA DeepStream.

📖 **[Full Documentation](README.md)** | 🔧 **[Setup Guide](SETUP.md)**

## 30-Second Setup

```bash
# Install with uv (modern, fast)
uv venv && source .venv/bin/activate
uv pip install -e .

# Or with Docker
docker-compose build training
```

## 30-Second Training

**Host (Recommended)**:
```bash
source .venv/bin/activate
cd training
python scripts/train.py --data configs/datasets/ppe.yaml --name ppe_v1 --size s
python scripts/export_onnx.py --all
```

**Docker (Alternative)**:
```bash
docker-compose run training python scripts/train.py \
    --data configs/datasets/ppe.yaml --name ppe_v1 --size s
docker-compose run training python scripts/export_onnx.py --all
```

## 30-Second Deploy

```bash
# Deploy to Edge
docker-compose --profile edge up edge-deploy

# Deploy to x86/Cloud
docker-compose --profile x86 up x86-deploy
```

## Project Structure

```
af-training/
├── training/          # Training scripts & configs
├── deployment/        # Platform-specific deployment
├── docker/           # Containers (train, edge, x86)
├── docs/             # Comprehensive guides
└── tools/            # Benchmarking & validation
```

## Supported Platforms

- ☁️ **Cloud** (T4, A10, A100) - FP16, high throughput
- 💻 **x86 PCs** (RTX GPUs) - FP16, balanced
- 🤖 **Jetson Orin** - INT8, edge deployment

## Supported Domains

- 🦺 PPE Detection (helmets, vests, harness)
- 🚗 Traffic Monitoring (vehicles, pedestrians)
- 🦌 Wildlife (camera traps, conservation)
- 🏗️ Industrial/Mining (heavy equipment)

## Documentation

- [README.md](README.md) - Complete guide
- [docs/DATASETS.md](docs/DATASETS.md) - Dataset preparation
- [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) - Multi-platform deployment
- [docs/CONVERSATION_ANALYSIS.md](docs/CONVERSATION_ANALYSIS.md) - Project insights

Created from conversation analysis - production-ready ML pipeline for DeepStream.
