
import logging
from typing import Dict, Set
from fastapi import WebSocket

logger = logging.getLogger(__name__)

class WebSocketConnectionManager:
    def __init__(self):
        self.active: Dict[str, Set[WebSocket]] = {}

    async def connect(self, session_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active.setdefault(session_id, set()).add(websocket)
        logger.info("WebSocket connected for session %s. Total connections: %d",
                    session_id, len(self.active.get(session_id, [])))

    def disconnect(self, session_id: str, websocket: WebSocket):
        group = self.active.get(session_id)
        if group and websocket in group:
            group.remove(websocket)
        logger.info("WebSocket disconnected for session %s", session_id)

    async def broadcast(self, session_id: str, message: dict):
        # Optionally log for debugging
        # logger.info("Broadcasting to %s: %s", session_id, message.get("event"))
        for ws in list(self.active.get(session_id, [])):
            try:
                await ws.send_json(message)
            except Exception:
                self.disconnect(session_id, ws)

# Create ONE singleton here
manager = WebSocketConnectionManager()
logger.info("WS manager singleton created id=%s", id(manager))
