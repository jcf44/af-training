from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import List, Dict, Any
import os
import yaml
import shutil
import zipfile
from pathlib import Path
from pydantic import BaseModel

router = APIRouter(prefix="/datasets", tags=["datasets"])

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
DATASETS_DIR = os.path.join(PROJECT_ROOT, "training/datasets")
CONFIGS_DIR = os.path.join(PROJECT_ROOT, "training/configs/datasets")

class DatasetPrepareRequest(BaseModel):
    dataset_name: str
    train_split: float = 0.7
    val_split: float = 0.2
    test_split: float = 0.1

@router.get("/")
def list_datasets():
    """List available datasets (raw and processed) and configs."""
    raw_dir = os.path.join(DATASETS_DIR, "raw")
    processed_dir = os.path.join(DATASETS_DIR, "processed")
    
    raw_datasets = [d for d in os.listdir(raw_dir) if not d.startswith(".")] if os.path.exists(raw_dir) else []
    processed_datasets = [d for d in os.listdir(processed_dir) if not d.startswith(".")] if os.path.exists(processed_dir) else []
    configs = [d for d in os.listdir(CONFIGS_DIR) if not d.startswith(".")] if os.path.exists(CONFIGS_DIR) else []
    
    calib_dir = os.path.join(PROJECT_ROOT, "training/configs/calibration")
    calib_configs = [d for d in os.listdir(calib_dir) if not d.startswith(".")] if os.path.exists(calib_dir) else []
    
    return {
        "raw": raw_datasets,
        "processed": processed_datasets,
        "configs": configs,
        "calibration": calib_configs
    }

@router.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    name: str = Form(...)
):
    """Upload and extract a dataset zip file."""
    if not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only .zip files are allowed")

    # Create processed directory for this dataset
    dataset_path = os.path.join(DATASETS_DIR, "processed", name)
    if os.path.exists(dataset_path):
        raise HTTPException(status_code=400, detail=f"Dataset {name} already exists")
    
    os.makedirs(dataset_path, exist_ok=True)
    
    # Save zip temporarily
    temp_zip = os.path.join(dataset_path, "temp.zip")
    try:
        with open(temp_zip, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Extract zip
        with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
            zip_ref.extractall(dataset_path)
    finally:
        if os.path.exists(temp_zip):
            os.remove(temp_zip)

    # Find data.yaml and dataset root
    yaml_path = None
    dataset_root = dataset_path
    
    # Check if extracted into a subdirectory (common in zips)
    items = os.listdir(dataset_path)
    if len(items) == 1 and os.path.isdir(os.path.join(dataset_path, items[0])):
        # Move contents up
        subdir = os.path.join(dataset_path, items[0])
        for item in os.listdir(subdir):
            shutil.move(os.path.join(subdir, item), dataset_path)
        os.rmdir(subdir)

    # Search for data.yaml (or similar)
    for root, dirs, files in os.walk(dataset_path):
        for f in files:
            if f in ["data.yaml", "data.yml", "dataset.yaml"]:
                yaml_path = os.path.join(root, f)
                break
        if yaml_path:
            break
            
    if not yaml_path:
        # Cleanup
        shutil.rmtree(dataset_path)
        raise HTTPException(status_code=400, detail="Could not find data.yaml in zip")

    # Process YAML
    try:
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Update paths to be absolute or relative to the new location
        # We set 'path' to the absolute path of the dataset directory
        config['path'] = os.path.abspath(dataset_path)
        
        # Ensure train/val/test paths are correct relative to dataset_path
        # Common variations: 'train/images', 'train', './train'
        def fix_path(p):
            if not p: return None
            # Remove ../ or ./ prefixes
            clean_p = p.replace('../', '').replace('./', '')
            # If it points to images, keep it. If just folder, append /images if it exists
            if os.path.exists(os.path.join(dataset_path, clean_p)):
                return clean_p
            return p # Fallback

        if 'train' in config: config['train'] = fix_path(config['train'])
        if 'val' in config: config['val'] = fix_path(config['val'])
        if 'test' in config: config['test'] = fix_path(config['test'])
        
        # Save new config to configs dir
        new_config_path = os.path.join(CONFIGS_DIR, f"{name}.yaml")
        with open(new_config_path, 'w') as f:
            yaml.dump(config, f, sort_keys=False)
            
        # Remove original yaml from dataset dir to avoid confusion
        os.remove(yaml_path)
        
    except Exception as e:
        # Cleanup
        shutil.rmtree(dataset_path)
        raise HTTPException(status_code=500, detail=f"Failed to process dataset config: {str(e)}")

    return {"message": f"Dataset {name} uploaded and configured successfully", "config": f"{name}.yaml"}
