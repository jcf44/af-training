# AF-Training Documentation Index

Welcome to the AF-Training documentation! This index will guide you through all available documentation.

---

## 🚀 Getting Started

### [QUICKSTART.md](QUICKSTART.md)
**Quick reference for immediate setup and usage**
- 30-second setup with uv
- Quick training examples
- Quick deployment commands
- Perfect for returning users

### [SETUP.md](SETUP.md)
**Comprehensive environment setup guide**
- Host-based setup (recommended)
  - uv installation and usage
  - pip alternative
- Docker setup (optional)
- GPU verification
- TensorRT installation
- Troubleshooting
- **Start here if you're setting up for the first time**

---

## 📚 Core Documentation

### [DATASETS.md](DATASETS.md)
**Complete dataset preparation guide**
- Dataset requirements per domain
- Public dataset sources
  - PPE detection datasets
  - Traffic monitoring datasets
  - Wildlife datasets
  - Industrial/mining datasets
- Annotation tools comparison (CVAT, Label Studio, Roboflow)
- Dataset preparation workflow
  - Frame extraction
  - Train/val/test splitting
  - YAML configuration
- Data quality checks
- INT8 calibration dataset creation

### [DEPLOYMENT.md](DEPLOYMENT.md)
**Multi-platform deployment guide**
- Cloud deployment (T4, A10, A100)
  - FP16 configuration
  - High-throughput setup
- x86 PC deployment (RTX GPUs)
  - Workstation setup
  - Performance optimization
- Jetson Edge deployment (Orin Nano+)
  - INT8 optimization
  - Engine building on device
  - Power efficiency tips
- Platform comparison matrix
- Performance expectations
- Troubleshooting deployment issues

---

## 📖 Reference Documentation

### [CONVERSATION_ANALYSIS.md](CONVERSATION_ANALYSIS.md)
**Project background and design decisions**
- Original conversation analysis
- Strengths identified
- Improvements implemented
- Key insights incorporated
- How this project addresses gaps in the original discussion
- Recommended future enhancements

### [PYTHON_ENV_SUMMARY.md](PYTHON_ENV_SUMMARY.md)
**Python environment setup summary**
- Modern uv-based setup
- Traditional pip setup
- Benefits and comparisons
- Quick reference commands

---

## 📂 Documentation Organization

```
docs/
├── INDEX.md                    # This file - documentation index
├── QUICKSTART.md              # Quick reference guide
├── SETUP.md                   # Environment setup guide
├── DATASETS.md                # Dataset preparation guide
├── DEPLOYMENT.md              # Multi-platform deployment guide
├── CONVERSATION_ANALYSIS.md   # Project background
└── PYTHON_ENV_SUMMARY.md      # Python environment summary
```

---

## 🎯 Documentation by Use Case

### "I'm new to this project"
1. Read [../README.md](../README.md) for overview
2. Follow [SETUP.md](SETUP.md) for environment setup
3. Check [DATASETS.md](DATASETS.md) for dataset preparation
4. Start training with examples in [../README.md](../README.md)

### "I need to prepare my dataset"
1. [DATASETS.md](DATASETS.md) - Complete guide
2. Use `training/scripts/prepare_dataset.py`

### "I'm ready to deploy"
1. [DEPLOYMENT.md](DEPLOYMENT.md) - Platform-specific guides
2. Choose your target: Cloud, x86, or Jetson Edge
3. Follow platform-specific instructions

### "I want a quick reference"
1. [QUICKSTART.md](QUICKSTART.md) - Quick commands
2. [../README.md](../README.md) - Example workflows

### "I want to understand the project design"
1. [CONVERSATION_ANALYSIS.md](CONVERSATION_ANALYSIS.md) - Background
2. [../README.md](../README.md) - Project overview

---

## 📋 Additional Resources

### Root Directory Files

- **[../README.md](../README.md)** - Main project documentation
  - Project overview
  - Complete workflows
  - Model/precision selection guides
  - Performance expectations
  - Tips & best practices

- **[../pyproject.toml](../pyproject.toml)** - Python project configuration
  - Dependencies
  - Development tools
  - Build configuration

- **[../requirements.txt](../requirements.txt)** - Pip dependencies
  - For traditional pip-based installs

- **[../docker-compose.yml](../docker-compose.yml)** - Container orchestration
  - Training container
  - Edge deployment
  - x86/Cloud deployment

### Configuration Files

- `training/configs/datasets/` - Dataset YAML configurations
  - `ppe.yaml` - PPE detection
  - `traffic.yaml` - Traffic monitoring
  - `wildlife.yaml` - Wildlife detection
  - `industrial.yaml` - Industrial equipment
  - `calibration/` - INT8 calibration configs

- `deployment/common/configs/` - DeepStream configurations
  - Universal templates
  - Platform-specific configs

### Scripts

- `training/scripts/` - Training pipeline scripts
  - `train.py` - Model training
  - `export_onnx.py` - ONNX export
  - `generate_calibration.py` - INT8 calibration
  - `prepare_dataset.py` - Dataset utilities

- `tools/` - Utility scripts
  - `benchmark.py` - Performance benchmarking
  - `validate_deployment.py` - Deployment validation

---

## 🔍 Quick Reference

### Training Commands
```bash
# Setup
source .venv/bin/activate
cd training

# Train
python scripts/train.py --data configs/datasets/ppe.yaml --name ppe_v1 --size s

# Export
python scripts/export_onnx.py --all
```

### Deployment Commands
```bash
# Copy models
cp training/outputs/onnx/*.onnx deployment/common/models/

# Deploy to x86/Cloud
docker-compose --profile x86 up x86-deploy

# Deploy to Jetson Edge
docker-compose --profile edge up edge-deploy
```

---

## 🆘 Need Help?

1. **Setup issues** → [SETUP.md](SETUP.md) troubleshooting section
2. **Dataset questions** → [DATASETS.md](DATASETS.md)
3. **Deployment problems** → [DEPLOYMENT.md](DEPLOYMENT.md) troubleshooting
4. **Understanding the project** → [CONVERSATION_ANALYSIS.md](CONVERSATION_ANALYSIS.md)

---

## 📝 Documentation Standards

All documentation in this project follows:
- **Markdown format** with GitHub Flavored Markdown
- **Clear structure** with table of contents where appropriate
- **Code examples** with syntax highlighting
- **Tables** for comparisons and specifications
- **Links** to related documentation and files

---

**Last Updated**: 2025-11-27  
**Project Version**: 1.0.0
