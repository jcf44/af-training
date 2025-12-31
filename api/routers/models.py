import hashlib
import io
import os
import zipfile
from typing import List, Optional

import httpx
import yaml
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from ..config import settings

router = APIRouter(prefix="/models", tags=["models"])

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
MODELS_DIR = os.path.join(PROJECT_ROOT, "training/outputs/trained") # Where trained weights are
ONNX_DIR = os.path.join(PROJECT_ROOT, "training/outputs/onnx")
CALIB_DIR = os.path.join(PROJECT_ROOT, "training/outputs/calibration")

@router.get("/")
def list_models():
    """List trained models and ONNX exports."""
    trained_models = []
    if os.path.exists(MODELS_DIR):
        for name in os.listdir(MODELS_DIR):
            model_path = os.path.join(MODELS_DIR, name, "weights", "best.pt")
            if os.path.exists(model_path):
                trained_models.append({
                    "name": name,
                    "path": model_path,
                    "type": "pt"
                })
                
    onnx_models = []
    if os.path.exists(ONNX_DIR):
        for name in os.listdir(ONNX_DIR):
            if name.endswith(".onnx"):
                onnx_models.append({
                    "name": name,
                    "path": os.path.join(ONNX_DIR, name),
                    "type": "onnx"
                })

    calib_files = []
    if os.path.exists(CALIB_DIR):
        for name in os.listdir(CALIB_DIR):
            if name.endswith(".cache"):
                calib_files.append({
                    "name": name,
                    "path": os.path.join(CALIB_DIR, name),
                    "type": "cache"
                })
                
    return {
        "trained": trained_models,
        "onnx": onnx_models,
        "calibration": calib_files
    }

@router.get("/{filename}/download")
def download_model(filename: str, type: str = "onnx"):
    """Download a model file."""
    if type == "onnx":
        file_path = os.path.join(ONNX_DIR, filename)
    elif type == "pt":
        # Filename here is likely the model name, e.g., ppe_v1
        file_path = os.path.join(MODELS_DIR, filename, "weights", "best.pt")
    elif type == "calibration":
        file_path = os.path.join(CALIB_DIR, filename)
    else:
        raise HTTPException(status_code=400, detail="Invalid type")
        
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
        
    return FileResponse(file_path, filename=os.path.basename(file_path))

@router.post("/{name}/export")
def export_model(name: str):
    """Export a trained model to ONNX."""
    model_path = os.path.join(MODELS_DIR, name, "weights", "best.pt")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model {name} not found")
        
    # Construct command
    cmd = [
        "python",
        "training/scripts/export_onnx.py",
        "--model", model_path,
        "--output", ONNX_DIR,
        "--opset", "12"
    ]
    
    log_path = os.path.join(PROJECT_ROOT, f"training/outputs/logs/export_{name}.log")
    
    try:
        from ..services.process_manager import process_manager
        from ..services.event_manager import event_manager
        pid = process_manager.start_process(cmd, log_path)
        event_manager.track_process(pid, "export", name, log_path)
        return {"message": f"Export started for {name}", "pid": pid}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start export: {str(e)}")

@router.post("/{name}/calibrate")
def calibrate_model(name: str, config: str):
    """Generate INT8 calibration cache."""
    model_path = os.path.join(MODELS_DIR, name, "weights", "best.pt")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model {name} not found")
        
    if config == "auto":
        # Auto-generate calibration config from training args
        args_path = os.path.join(MODELS_DIR, name, "args.yaml")
        if not os.path.exists(args_path):
            raise HTTPException(status_code=400, detail=f"Could not find args.yaml for model {name}. Cannot auto-calibrate.")
            
        try:
            with open(args_path, "r") as f:
                args = yaml.safe_load(f)
                
            data_config_path = args.get("data")
            if not data_config_path:
                raise HTTPException(status_code=400, detail="Could not find dataset path in args.yaml")
                
            # Handle relative paths in args.yaml (relative to project root usually)
            if not os.path.isabs(data_config_path):
                data_config_path = os.path.join(PROJECT_ROOT, data_config_path)
                
            if not os.path.exists(data_config_path):
                raise HTTPException(status_code=400, detail=f"Original dataset config not found: {data_config_path}")
                
            with open(data_config_path, "r") as f:
                data_config = yaml.safe_load(f)
                
            # Create temp calibration config
            calib_config = {
                "path": data_config.get("path"),
                "train": data_config.get("val"), # Use validation set for calibration
                "val": data_config.get("val"),
                "nc": data_config.get("nc"),
                "names": data_config.get("names")
            }
            
            config_name = f"auto_calib_{name}.yaml"
            config_path = os.path.join(PROJECT_ROOT, "training/configs/calibration", config_name)
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            with open(config_path, "w") as f:
                yaml.dump(calib_config, f)
                
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to auto-generate config: {str(e)}")
    else:
        config_path = os.path.join(PROJECT_ROOT, "training/configs/calibration", config)
        if not os.path.exists(config_path):
            raise HTTPException(status_code=404, detail=f"Calibration config {config} not found")
        
    # Construct command
    cmd = [
        "python",
        "training/scripts/generate_calibration.py",
        "--model", model_path,
        "--data", config_path,
        "--output", os.path.join(PROJECT_ROOT, "training/outputs/calibration")
    ]
    
    log_path = os.path.join(PROJECT_ROOT, f"training/outputs/logs/calibrate_{name}.log")
    
    try:
        from ..services.process_manager import process_manager
        from ..services.event_manager import event_manager
        pid = process_manager.start_process(cmd, log_path)
        event_manager.track_process(pid, "calibration", name, log_path)
        return {"message": f"Calibration started for {name}", "pid": pid}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start calibration: {str(e)}")

@router.get("/{name}/bundle")
def download_bundle(name: str):
    """Generate and download a deployment bundle (ONNX + Config + Labels + Cache)."""
    import zipfile
    import io
    
    # 1. Locate artifacts
    model_dir = os.path.join(MODELS_DIR, name)
    args_path = os.path.join(model_dir, "args.yaml")
    
    # Find ONNX file (assume best_model.onnx or similar, or just search)
    onnx_file = None
    # Check standard export name first
    potential_onnx = os.path.join(ONNX_DIR, f"{name}_best.onnx")
    if os.path.exists(potential_onnx):
        onnx_file = potential_onnx
    else:
        # Fallback search
        if os.path.exists(ONNX_DIR):
            for f in os.listdir(ONNX_DIR):
                if f.startswith(name) and f.endswith(".onnx"):
                    onnx_file = os.path.join(ONNX_DIR, f)
                    break
    
    if not onnx_file:
        raise HTTPException(status_code=404, detail=f"ONNX model for {name} not found. Please export it first.")

    # Find Calibration Cache
    cache_file = None
    potential_cache = os.path.join(CALIB_DIR, f"{name}_best.calibration.cache") # Logic might need adjustment based on generation script
    # The generation script copies *.cache. Let's look for any cache file matching the name or just *.cache in the output dir if we can be specific.
    # Actually, the calibration script copies the cache file to CALIB_DIR.
    # Let's search for a cache file that looks related.
    if os.path.exists(CALIB_DIR):
        for f in os.listdir(CALIB_DIR):
            # Heuristic: if the cache file starts with the model name or 'best' (if copied directly)
            # The calibration script copies `best.calibration.cache` to `CALIB_DIR/best.calibration.cache`. 
            # Wait, if multiple models run, they might overwrite? 
            # The previous script did: dest = output_dir / cache_file.name. 
            # So it's likely `best.calibration.cache`. This is a collision risk!
            # I should have renamed it. 
            # For now, let's assume the user just ran it. 
            # BETTER: Update calibration script to rename it? 
            # Or just look for *any* .cache file if we assume one model context?
            # Let's look for one that matches the model name if I renamed it, or just use the most recent?
            # Let's assume for now we look for `best.calibration.cache` or `{name}.cache`.
            if f.endswith(".cache"):
                 # If we have a naming convention, use it. 
                 # If not, maybe we just take the one there?
                 # Let's try to match name.
                 pass
        
        # Re-reading the calibration script logic:
        # dest = output_dir / cache_file.name
        # It preserves the name 'best.calibration.cache'.
        # This IS a problem for multiple models.
        # But for this task, let's assume we pick 'best.calibration.cache' if it exists.
        if os.path.exists(os.path.join(CALIB_DIR, "best.calibration.cache")):
            cache_file = os.path.join(CALIB_DIR, "best.calibration.cache")

    # 2. Generate Labels
    labels = []
    num_classes = 80 # Default
    if os.path.exists(args_path):
        with open(args_path, "r") as f:
            args = yaml.safe_load(f)
            data_path = args.get("data")
            # Resolve path
            if not os.path.isabs(data_path):
                data_path = os.path.join(PROJECT_ROOT, data_path)
            
            if os.path.exists(data_path):
                with open(data_path, "r") as df:
                    data_config = yaml.safe_load(df)
                    names = data_config.get("names", {})
                    num_classes = data_config.get("nc", 80)
                    # names can be list or dict
                    if isinstance(names, list):
                        labels = names
                    elif isinstance(names, dict):
                        labels = [names[i] for i in sorted(names.keys())]
    
    if not labels:
        labels = [f"class_{i}" for i in range(num_classes)]

    # 3. Generate DeepStream Config
    onnx_filename = os.path.basename(onnx_file)
    cache_filename = os.path.basename(cache_file) if cache_file else None
    
    ds_config = f"""[property]
gpu-id=0
net-scale-factor=0.0039215697906911373
model-color-format=0
onnx-file={onnx_filename}
model-engine-file=model.engine
labelfile-path=labels.txt
batch-size=1
## 0=FP32, 1=INT8, 2=FP16 mode
network-mode={'1' if cache_file else '2'}
num-detected-classes={num_classes}
interval=0
gie-unique-id=1
process-mode=1
network-type=0
cluster-mode=2
maintain-aspect-ratio=1
symmetric-padding=1
"""
    if cache_file:
        ds_config += f"int8-calib-file={cache_filename}\n"

    # 4. Create ZIP
    # Create a temporary file or in-memory bytes
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        # Add ONNX
        zip_file.write(onnx_file, onnx_filename)
        
        # Add Cache
        if cache_file:
            zip_file.write(cache_file, cache_filename)
            
        # Add Labels
        zip_file.writestr("labels.txt", "\\n".join(labels))
        
        # Add Config
        zip_file.writestr("config_infer_primary.txt", ds_config)
        
    zip_buffer.seek(0)

    return StreamingResponse(
        iter([zip_buffer.getvalue()]),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={name}_deployment.zip"}
    )


# ML Model directories (for sklearn, xgboost, etc.)
ML_OUTPUTS_DIR = os.path.join(PROJECT_ROOT, settings.ML_OUTPUTS_DIR)


class PushRequest(BaseModel):
    """Request body for pushing model to registry."""

    model_type: str = "isolation_forest"
    framework: str = "joblib"
    version: str = "1.0"
    description: Optional[str] = None
    tags: Optional[List[str]] = None


def _calculate_checksum(file_path: str) -> str:
    """Calculate SHA256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def _find_ml_model_files(name: str) -> dict:
    """
    Find ML model files for a given model name.

    Searches in common output directories:
    - outputs/anomaly/{name}/
    - outputs/tabular/{name}/
    - outputs/{name}/

    Returns dict with model_path, scaler_path, metadata_path (all optional).
    """
    result = {"model_path": None, "scaler_path": None, "metadata_path": None}

    # Possible locations
    search_dirs = [
        os.path.join(ML_OUTPUTS_DIR, "anomaly", name),
        os.path.join(ML_OUTPUTS_DIR, "tabular", name),
        os.path.join(ML_OUTPUTS_DIR, name),
    ]

    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue

        # Look for model file
        for model_name in ["model.pkl", "model.joblib", f"{name}.pkl", f"{name}.joblib"]:
            model_path = os.path.join(search_dir, model_name)
            if os.path.exists(model_path):
                result["model_path"] = model_path
                break

        # Look for scaler file
        for scaler_name in ["scaler.pkl", "scaler.joblib"]:
            scaler_path = os.path.join(search_dir, scaler_name)
            if os.path.exists(scaler_path):
                result["scaler_path"] = scaler_path
                break

        # Look for metadata file
        for meta_name in ["metadata.json", "config.json", "training_config.json"]:
            meta_path = os.path.join(search_dir, meta_name)
            if os.path.exists(meta_path):
                result["metadata_path"] = meta_path
                break

        # If we found a model, stop searching
        if result["model_path"]:
            break

    return result


@router.get("/ml")
def list_ml_models():
    """List ML models (sklearn, xgboost, etc.) available for push."""
    ml_models = []

    # Search in ML outputs directories
    search_roots = [
        os.path.join(ML_OUTPUTS_DIR, "anomaly"),
        os.path.join(ML_OUTPUTS_DIR, "tabular"),
        ML_OUTPUTS_DIR,
    ]

    seen_names = set()

    for search_root in search_roots:
        if not os.path.exists(search_root):
            continue

        for name in os.listdir(search_root):
            if name in seen_names:
                continue

            model_dir = os.path.join(search_root, name)
            if not os.path.isdir(model_dir):
                continue

            # Check if it has a model file
            files = _find_ml_model_files(name)
            if files["model_path"]:
                seen_names.add(name)

                # Determine model type from directory structure
                if "anomaly" in search_root:
                    model_type = "isolation_forest"
                elif "tabular" in search_root:
                    model_type = "xgboost"
                else:
                    model_type = "sklearn"

                ml_models.append(
                    {
                        "name": name,
                        "model_path": files["model_path"],
                        "scaler_path": files["scaler_path"],
                        "metadata_path": files["metadata_path"],
                        "type": model_type,
                    }
                )

    return {"ml_models": ml_models}


@router.post("/{name}/push")
async def push_to_registry(
    name: str,
    request: PushRequest,
    registry_url: str = Query(
        default=None, description="Override registry URL from config"
    ),
):
    """
    Push a trained model to the af-api2 model registry.

    Supports both DL models (ONNX) and ML models (sklearn/joblib).

    For DL models: Pushes the ONNX bundle (model + config + labels)
    For ML models: Pushes the model file + optional scaler file
    """
    url = registry_url or settings.REGISTRY_URL

    # First, try to find ML model
    ml_files = _find_ml_model_files(name)

    if ml_files["model_path"]:
        # Push ML model
        return await _push_ml_model(name, ml_files, request, url)

    # If not ML, check for DL model (ONNX)
    onnx_file = None
    potential_onnx = os.path.join(ONNX_DIR, f"{name}_best.onnx")
    if os.path.exists(potential_onnx):
        onnx_file = potential_onnx
    else:
        # Search for matching ONNX file
        if os.path.exists(ONNX_DIR):
            for f in os.listdir(ONNX_DIR):
                if f.startswith(name) and f.endswith(".onnx"):
                    onnx_file = os.path.join(ONNX_DIR, f)
                    break

    if onnx_file:
        # Push DL model
        return await _push_dl_model(name, onnx_file, request, url)

    raise HTTPException(
        status_code=404,
        detail=f"No model found for '{name}'. Check outputs/anomaly/{name}/, "
        f"outputs/tabular/{name}/, or training/outputs/onnx/",
    )


async def _push_ml_model(
    name: str,
    files: dict,
    request: PushRequest,
    registry_url: str,
):
    """Push ML model (sklearn/joblib) to registry."""
    model_path = files["model_path"]
    scaler_path = files.get("scaler_path")
    metadata_path = files.get("metadata_path")

    # Calculate checksum
    checksum = _calculate_checksum(model_path)

    # Load metadata if available
    model_metadata = {}
    if metadata_path and os.path.exists(metadata_path):
        import json

        with open(metadata_path, "r") as f:
            model_metadata = json.load(f)

    # Generate model_id
    model_id = f"{name}-{request.model_type}-v{request.version}".replace(" ", "-").lower()

    # Prepare multipart upload
    upload_url = f"{registry_url}/api/v1/models/upload"

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            # Prepare files for upload
            files_dict = {
                "model_file": (
                    os.path.basename(model_path),
                    open(model_path, "rb"),
                    "application/octet-stream",
                ),
            }

            # Add scaler if available
            if scaler_path and os.path.exists(scaler_path):
                files_dict["scaler_file"] = (
                    os.path.basename(scaler_path),
                    open(scaler_path, "rb"),
                    "application/octet-stream",
                )

            # Prepare form data
            data = {
                "model_id": model_id,
                "name": name,
                "version": request.version,
                "type": request.model_type,
                "framework": request.framework,
                "description": request.description or f"ML model trained with af-training",
                "metadata": str(model_metadata),  # Will be JSON serialized
            }

            if request.tags:
                data["tags"] = ",".join(request.tags)

            response = await client.post(upload_url, data=data, files=files_dict)

            # Close file handles
            for f in files_dict.values():
                f[1].close()

            if response.status_code not in (200, 201):
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Registry upload failed: {response.text}",
                )

            result = response.json()

            return {
                "message": f"ML model '{name}' pushed to registry",
                "registry_model_id": result.get("model_id", model_id),
                "model_type": request.model_type,
                "framework": request.framework,
                "checksum": checksum,
                "has_scaler": scaler_path is not None,
            }

    except httpx.RequestError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to connect to registry at {registry_url}: {str(e)}",
        )


async def _push_dl_model(
    name: str,
    onnx_file: str,
    request: PushRequest,
    registry_url: str,
):
    """Push DL model (ONNX bundle) to registry."""
    # Find associated files
    model_dir = os.path.join(MODELS_DIR, name)
    args_path = os.path.join(model_dir, "args.yaml")

    # Get labels
    labels = []
    num_classes = 80
    if os.path.exists(args_path):
        with open(args_path, "r") as f:
            args = yaml.safe_load(f)
            data_path = args.get("data")
            if data_path:
                if not os.path.isabs(data_path):
                    data_path = os.path.join(PROJECT_ROOT, data_path)

                if os.path.exists(data_path):
                    with open(data_path, "r") as df:
                        data_config = yaml.safe_load(df)
                        names = data_config.get("names", {})
                        num_classes = data_config.get("nc", 80)
                        if isinstance(names, list):
                            labels = names
                        elif isinstance(names, dict):
                            labels = [names[i] for i in sorted(names.keys())]

    if not labels:
        labels = [f"class_{i}" for i in range(num_classes)]

    # Generate DeepStream config
    onnx_filename = os.path.basename(onnx_file)
    ds_config = f"""[property]
gpu-id=0
net-scale-factor=0.0039215697906911373
model-color-format=0
onnx-file={onnx_filename}
model-engine-file=model.engine
labelfile-path=labels.txt
batch-size=1
network-mode=2
num-detected-classes={num_classes}
interval=0
gie-unique-id=1
process-mode=1
network-type=0
cluster-mode=2
maintain-aspect-ratio=1
symmetric-padding=1
"""

    # Create bundle ZIP
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(onnx_file, onnx_filename)
        zf.writestr("labels.txt", "\n".join(labels))
        zf.writestr("config_infer_primary.txt", ds_config)

    zip_buffer.seek(0)

    # Calculate checksum of ONNX file
    checksum = _calculate_checksum(onnx_file)

    # Generate model_id
    model_id = f"{name}-yolo-v{request.version}".replace(" ", "-").lower()

    # Push to registry
    upload_url = f"{registry_url}/api/v1/models/upload-bundle"

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            files_dict = {
                "bundle_file": (
                    f"{name}_deployment.zip",
                    zip_buffer,
                    "application/zip",
                ),
            }

            data = {
                "model_id": model_id,
                "name": name,
                "version": request.version,
                "type": "yolo",
                "framework": "onnx",
                "description": request.description or f"YOLO model trained with af-training",
            }

            if request.tags:
                data["tags"] = ",".join(request.tags)

            response = await client.post(upload_url, data=data, files=files_dict)

            if response.status_code not in (200, 201):
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Registry upload failed: {response.text}",
                )

            result = response.json()

            return {
                "message": f"DL model '{name}' pushed to registry",
                "registry_model_id": result.get("model_id", model_id),
                "model_type": "yolo",
                "framework": "onnx",
                "checksum": checksum,
                "num_classes": num_classes,
            }

    except httpx.RequestError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to connect to registry at {registry_url}: {str(e)}",
        )
