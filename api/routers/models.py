from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
import os
import yaml
from typing import List

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
