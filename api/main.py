from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from .database import create_db_and_tables
from .routers import datasets, training, models, events
from .services.event_manager import event_manager
import asyncio

@asynccontextmanager
async def lifespan(app: FastAPI):
    create_db_and_tables()
    # Start background monitor
    asyncio.create_task(event_manager.monitor_jobs())
    yield

app = FastAPI(title="AF-Training API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(datasets.router)
app.include_router(training.router)
app.include_router(models.router)
app.include_router(events.router)

def start_dev():
    import uvicorn
    from .config import settings
    uvicorn.run("api.main:app", host="0.0.0.0", port=settings.API_PORT, reload=True)

if __name__ == "__main__":
    start_dev()
