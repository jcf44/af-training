import asyncio
import logging
import json
import os
from typing import List, Dict, Any
from sse_starlette.sse import ServerSentEvent
from sqlmodel import Session, select
from datetime import datetime

from ..database import engine
from ..models.db import TrainingJob
from .process_manager import process_manager

logger = logging.getLogger(__name__)

class EventManager:
    _instance = None
    _subscribers: List[asyncio.Queue] = []

    _tracked_processes: Dict[int, Dict[str, Any]] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EventManager, cls).__new__(cls)
            cls._instance._subscribers = []
            cls._instance._tracked_processes = {}
        return cls._instance

    async def subscribe(self):
        queue = asyncio.Queue()
        self._subscribers.append(queue)
        try:
            while True:
                data = await queue.get()
                yield ServerSentEvent(data=json.dumps(data), event="message")
        except asyncio.CancelledError:
            self._subscribers.remove(queue)

    async def broadcast(self, event_type: str, data: Dict[str, Any]):
        message = {"type": event_type, "data": data}
        for queue in self._subscribers:
            await queue.put(message)

    def track_process(self, pid: int, job_type: str, name: str, log_path: str = None):
        """Track a process that isn't in the database (e.g., export)."""
        self._tracked_processes[pid] = {
            "type": job_type,
            "name": name,
            "start_time": datetime.utcnow(),
            "log_path": log_path,
            "log_pos": 0
        }

    async def monitor_jobs(self):
        """Background task to monitor running jobs."""
        logger.info("Starting background job monitor")
        while True:
            try:
                # 1. Check DB jobs (Training)
                with Session(engine) as session:
                    statement = select(TrainingJob).where(TrainingJob.status == "running")
                    running_jobs = session.exec(statement).all()
                    
                    for job in running_jobs:
                        if job.pid:
                            if not process_manager.is_process_running(job.pid):
                                job.status = "completed"
                                job.end_time = datetime.utcnow()
                                session.add(job)
                                session.commit()
                                
                                logger.info(f"Job {job.id} ({job.name}) completed")
                                await self.broadcast("job_completed", {
                                    "id": job.id,
                                    "name": job.name,
                                    "status": "completed"
                                })

                # 2. Check in-memory tracked processes (Export/Calibration)
                completed_pids = []
                for pid, info in self._tracked_processes.items():
                    # Stream logs if log_path is set
                    if "log_path" in info and os.path.exists(info["log_path"]):
                        try:
                            with open(info["log_path"], "r") as f:
                                f.seek(info.get("log_pos", 0))
                                new_lines = f.readlines()
                                if new_lines:
                                    info["log_pos"] = f.tell()
                                    # Broadcast new lines
                                    await self.broadcast("log_update", {
                                        "pid": pid,
                                        "name": info["name"],
                                        "type": info["type"],
                                        "lines": [line.rstrip() for line in new_lines]
                                    })
                        except Exception as e:
                            logger.error(f"Error reading logs for {pid}: {e}")

                    if not process_manager.is_process_running(pid):
                        logger.info(f"{info['type']} job {info['name']} (PID {pid}) completed")
                        await self.broadcast(f"{info['type']}_completed", {
                            "name": info['name'],
                            "status": "completed"
                        })
                        completed_pids.append(pid)
                
                for pid in completed_pids:
                    del self._tracked_processes[pid]

            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
            
            await asyncio.sleep(2)

event_manager = EventManager()
