from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse
from ..services.event_manager import event_manager

router = APIRouter(tags=["events"])

@router.get("/events")
async def events(request: Request):
    """SSE endpoint for global status updates."""
    return EventSourceResponse(event_manager.subscribe())
