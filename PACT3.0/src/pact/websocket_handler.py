"""
WebSocket Management for Real-Time Progress Updates

Handles WebSocket connections and broadcasts progress updates to connected clients.
"""

import json
from typing import Dict, List, Any
from fastapi import WebSocket, WebSocketDisconnect

class WebSocketManager:
    """
    Manages WebSocket connections for real-time progress updates.
    """
    
    def __init__(self):
        # Map of session_id to list of WebSocket connections
        self.connections: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        """
        Accept a new WebSocket connection for a session.
        """
        await websocket.accept()
        
        if session_id not in self.connections:
            self.connections[session_id] = []
        
        self.connections[session_id].append(websocket)
        print(f"WebSocket connected for session {session_id}. Total connections: {len(self.connections[session_id])}")
    
    async def disconnect(self, session_id: str):
        """
        Remove all WebSocket connections for a session.
        """
        if session_id in self.connections:
            connection_count = len(self.connections[session_id])
            del self.connections[session_id]
            print(f"WebSocket disconnected for session {session_id}. Removed {connection_count} connections.")
    
    async def disconnect_websocket(self, websocket: WebSocket, session_id: str):
        """
        Remove a specific WebSocket connection.
        """
        if session_id in self.connections:
            if websocket in self.connections[session_id]:
                self.connections[session_id].remove(websocket)
                print(f"WebSocket disconnected for session {session_id}. Remaining connections: {len(self.connections[session_id])}")
                
                # Clean up empty session lists
                if not self.connections[session_id]:
                    del self.connections[session_id]
    
    async def send_to_session(self, session_id: str, data: Dict[str, Any]):
        """
        Send data to all WebSocket connections for a specific session.
        """
        if session_id not in self.connections:
            return
        
        # Convert data to JSON
        message = json.dumps(data)
        
        # Send to all connections for this session
        disconnected_sockets = []
        
        for websocket in self.connections[session_id]:
            try:
                await websocket.send_text(message)
            except WebSocketDisconnect:
                disconnected_sockets.append(websocket)
            except Exception as e:
                print(f"Error sending message to WebSocket: {e}")
                disconnected_sockets.append(websocket)
        
        # Clean up disconnected sockets
        for websocket in disconnected_sockets:
            await self.disconnect_websocket(websocket, session_id)
    
    async def broadcast(self, data: Dict[str, Any]):
        """
        Broadcast data to all connected WebSocket clients.
        """
        message = json.dumps(data)
        
        for session_id, websockets in list(self.connections.items()):
            disconnected_sockets = []
            
            for websocket in websockets:
                try:
                    await websocket.send_text(message)
                except WebSocketDisconnect:
                    disconnected_sockets.append(websocket)
                except Exception as e:
                    print(f"Error broadcasting to WebSocket: {e}")
                    disconnected_sockets.append(websocket)
            
            # Clean up disconnected sockets
            for websocket in disconnected_sockets:
                await self.disconnect_websocket(websocket, session_id)
    
    def get_connection_count(self, session_id: str = None) -> int:
        """
        Get the number of active connections.
        
        Args:
            session_id: If provided, get count for specific session. Otherwise, get total count.
        """
        if session_id:
            return len(self.connections.get(session_id, []))
        else:
            return sum(len(websockets) for websockets in self.connections.values())
    
    def get_active_sessions(self) -> List[str]:
        """
        Get list of session IDs with active WebSocket connections.
        """
        return list(self.connections.keys())
    
    async def send_message(self, session_id: str, message: dict):
        """
        Send a message to all WebSocket connections for a session.
        Alias for send_to_session for compatibility.
        """
        await self.send_to_session(session_id, message)