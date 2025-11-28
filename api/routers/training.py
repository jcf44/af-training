from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlmodel import Session, select
from typing import List, Optional
from datetime import datetime
import json
import os
from pydantic import BaseModel

from ..database import get_session
from ..models.db import TrainingJob, TrainingJobRead
from ..services.process_manager import process_manager

router = APIRouter(prefix="/train", tags=["training"])

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

class TrainRequest(BaseModel):
    name: str
    dataset_config: str # e.g., "ppe.yaml"
    model_size: str = "s" # n, s, m, l, x
    model_type: str = "YOLO"
    epochs: int = 100
    batch_size: int = 16
    imgsz: int = 640
    
    class Config:
        extra = "allow"

@router.get("/schema/{model_type}")
def get_training_schema(model_type: str):
    """Get the configuration schema for a specific model type."""
    try:
        from ..services.training_strategies import TrainingStrategyFactory
        strategy = TrainingStrategyFactory.get_strategy(model_type)
        return strategy.get_schema()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/", response_model=TrainingJobRead)
def start_training(req: TrainRequest, session: Session = Depends(get_session)):
    # Create DB record
    job = TrainingJob(
        name=req.name,
        version="v1", # TODO: Auto-increment version
        status="running",
        config=json.dumps(req.dict()),
        start_time=datetime.utcnow()
    )
    session.add(job)
    session.commit()
    session.refresh(job)
    
    # Get strategy
    try:
        from ..services.training_strategies import TrainingStrategyFactory
        strategy = TrainingStrategyFactory.get_strategy(req.model_type)
        strategy.validate_config(req.dict())
        cmd = strategy.get_command(req.dict())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    log_path = os.path.join(PROJECT_ROOT, f"training/outputs/logs/{job.id}_{req.name}.log")
    
    try:
        pid = process_manager.start_process(cmd, log_path)
        job.pid = pid
        job.log_path = log_path
        session.add(job)
        session.commit()
        session.refresh(job)
    except Exception as e:
        job.status = "failed"
        session.add(job)
        session.commit()
        raise HTTPException(status_code=500, detail=f"Failed to start training: {str(e)}")
        
    return job

@router.get("/jobs", response_model=List[TrainingJobRead])
def list_jobs(session: Session = Depends(get_session)):
    jobs = session.exec(select(TrainingJob)).all()
    # Update status if process finished
    for job in jobs:
        if job.status == "running" and job.pid:
            if not process_manager.is_process_running(job.pid):
                job.status = "completed" # Or failed, would need to check exit code or log
                job.end_time = datetime.utcnow()
                session.add(job)
    session.commit()
    # Refresh to get updated fields
    for job in jobs:
        session.refresh(job)
    return jobs

@router.get("/{job_id}", response_model=TrainingJobRead)
def get_job(job_id: int, session: Session = Depends(get_session)):
    job = session.get(TrainingJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status == "running" and job.pid:
        if not process_manager.is_process_running(job.pid):
            job.status = "completed"
            job.end_time = datetime.utcnow()
            session.add(job)
            session.commit()
            session.refresh(job)
            
    return job

@router.post("/{job_id}/stop")
def stop_job(job_id: int, session: Session = Depends(get_session)):
    job = session.get(TrainingJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
        
    if job.status == "running" and job.pid:
        if process_manager.stop_process(job.pid):
            job.status = "stopped"
            job.end_time = datetime.utcnow()
            session.add(job)
            session.commit()
            return {"message": "Job stopped"}
        else:
            raise HTTPException(status_code=500, detail="Failed to stop process")
            
    return {"message": "Job is not running"}

@router.get("/{job_id}/logs")
def get_job_logs(job_id: int, session: Session = Depends(get_session)):
    job = session.get(TrainingJob, job_id)
    if not job or not job.log_path or not os.path.exists(job.log_path):
        return {"logs": ""}
        
    with open(job.log_path, "r") as f:
        return {"logs": f.read()}

@router.delete("/jobs")
def clear_jobs(session: Session = Depends(get_session)):
    """Clear all training jobs."""
    jobs = session.exec(select(TrainingJob)).all()
    for job in jobs:
        # Stop if running
        if job.status == "running" and job.pid:
            process_manager.stop_process(job.pid)
        session.delete(job)
    session.commit()
    return {"message": "All jobs cleared"}
