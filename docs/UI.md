# Web Interface Guide

The AF-Training Web UI provides a user-friendly dashboard to manage the entire training and deployment pipeline without touching the command line.

---

## 🖥️ Dashboard Overview

The dashboard is divided into three main sections:
1.  **Training**: Configure and start model training jobs.
2.  **Models**: Manage trained models, export to ONNX, calibrate for INT8, and download deployment bundles.
3.  **Datasets**: View available datasets (configured in `training/configs/datasets`).

---

## 🚀 Training Models

Navigate to the **Training** page to start a new training job.

### 1. Configuration
*   **Dataset**: Select a dataset from the dropdown. These correspond to YAML files in `training/configs/datasets/`.
*   **Model Size**: Choose the YOLOv8 model size:
    *   **Nano (n)**: Fastest, lowest accuracy. Best for very constrained edge devices.
    *   **Small (s)**: **Recommended**. Good balance of speed and accuracy.
    *   **Medium (m)**: Higher accuracy, slower. Good for cloud/x86.
    *   **Large (l)**: Best accuracy, slowest.
*   **Epochs**: Number of training cycles (default: 100).
*   **Batch Size**: Images per batch (default: 16). Reduce if you hit Out of Memory (OOM) errors.

### 2. Monitoring Progress
*   Click **Start Training** to begin.
*   **Real-time Logs**: A terminal window will appear showing the live output from the training script.
*   **Job Status**: You can see active jobs in the sidebar or top status bar.

---

## 📦 Managing Models

Navigate to the **Models** page to manage your trained artifacts.

### 1. Export to ONNX
*   Find your trained model in the "Trained Models" list.
*   Click **Export ONNX**.
*   This converts the PyTorch (`.pt`) model to a standard ONNX format optimized for inference.

### 2. INT8 Calibration (For Edge)
*   **Required for Jetson/DeepStream INT8 mode.**
*   Click **Calibrate INT8** on an ONNX model.
*   Select **"Auto-generate from Training Data"** (recommended).
*   This runs a calibration job using a subset of your validation data to generate a `.cache` file.
*   *Note: This process can take 5-10 minutes.*

### 3. Deployment Bundles
*   Once you have an ONNX model (and optional calibration cache), click the **Bundle** button.
*   This downloads a `.zip` file containing:
    *   `model.onnx`
    *   `model.calibration.cache` (if available)
    *   `labels.txt` (Auto-generated class names)
    *   `config_infer_primary.txt` (DeepStream config)

---

## 📡 Real-time Log Viewer

The UI includes a floating **Log Viewer** (terminal icon in the bottom right).
*   **Toggle**: Click the terminal icon to show/hide logs.
*   **Streaming**: Logs stream in real-time for all active background jobs (Training, Export, Calibration).
*   **History**: Scroll up to see past output.

---

## 🛠️ Troubleshooting UI Issues

*   **"Failed to fetch"**: Ensure the API server is running (`uv run af-training-api`).
*   **Logs not showing**: Refresh the page. The log viewer connects via Server-Sent Events (SSE).
*   **Training fails immediately**: Check the log viewer for Python errors (often missing data or OOM).
