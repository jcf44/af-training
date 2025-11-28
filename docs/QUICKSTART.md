# AF-Training: Quick Start

Complete ML training and deployment pipeline for NVIDIA DeepStream.

📖 **[Full Documentation](README.md)** | 🔧# Quick Start Guide

## 1. Setup

### Option A: Host (Recommended)
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Setup project
uv venv --python 3.12
source .venv/bin/activate
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
