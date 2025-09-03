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
        """Broadcast message to all connections for a session."""
        if session_id in self.active:
            disconnected = []
            for connection in self.active[session_id]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error(f"Error sending message to WebSocket: {str(e)}")
                    disconnected.append(connection)

            # Remove disconnected connections
            for conn in disconnected:
                self.active[session_id].remove(conn)

    # ✅ Legacy alias so existing calls keep working
    async def send_message(self, session_id: str, message: dict):
        await self.broadcast(session_id, message)

# Create ONE singleton here
manager = WebSocketConnectionManager()
logger.info("WS manager singleton created id=%s", id(manager))