from typing import Optional
from sqlmodel import Field, SQLModel
from datetime import datetime

class TrainingJobBase(SQLModel):
    name: str
    version: str
    status: str  # running, completed, failed, stopped
    start_time: datetime = Field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    config: str  # JSON string of the config used
    pid: Optional[int] = None
    log_path: Optional[str] = None

class TrainingJob(TrainingJobBase, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)

class TrainingJobRead(TrainingJobBase):
    id: int
