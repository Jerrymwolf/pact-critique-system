"""
Session Management for PACT Critique System

Handles critique sessions, progress tracking, and state management.
"""

import uuid
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
import os

from .utils.enum_safety import safe_status_value

def _coerce_enums(obj):
    """
    Recursively coerce Enums to strings for safe JSON serialization.

    Args:
        obj: Any object that may contain Enums

    Returns:
        Object with all Enums converted to strings
    """
    if isinstance(obj, Enum):
        # choose .value if defined, otherwise name
        return getattr(obj, "value", obj.name)
    if isinstance(obj, dict):
        return {k: _coerce_enums(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_coerce_enums(v) for v in obj]
    return obj

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CritiqueStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    PLANNING = "planning"
    EVALUATING = "evaluating"
    SYNTHESIZING = "synthesizing"
    COMPLETED = "completed"
    FAILED = "failed"

class AgentStatus(str, Enum):
    WAITING = "waiting"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class AgentProgress:
    agent_id: str
    agent_name: str
    status: AgentStatus
    message: str = ""
    progress: float = 0.0  # 0-100
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

@dataclass
class CritiqueSession:
    session_id: str
    status: CritiqueStatus
    paper_content: str
    created_at: datetime
    updated_at: datetime
    paper_title: Optional[str] = None
    paper_type: Optional[str] = None
    mode: str = "STANDARD"

    # Progress tracking
    overall_progress: float = 0.0
    agents: Dict[str, AgentProgress] = None
    current_stage: str = ""

    # Results
    dimension_critiques: Dict[str, Any] = None
    final_critique: Optional[str] = None
    overall_score: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    original_text: Optional[str] = None  # Added for original text storage

    # Error handling
    error_message: Optional[str] = None
    retry_count: int = 0

    def __post_init__(self):
        if self.agents is None:
            self.agents = {
                "supervisor": AgentProgress("supervisor", "Supervisor", AgentStatus.WAITING),
                "1": AgentProgress("1", "Research Foundations", AgentStatus.WAITING),
                "2": AgentProgress("2", "Methodological Rigor", AgentStatus.WAITING),
                "3": AgentProgress("3", "Structure & Coherence", AgentStatus.WAITING),
                "4": AgentProgress("4", "Academic Precision", AgentStatus.WAITING),
                "5": AgentProgress("5", "Critical Sophistication", AgentStatus.WAITING),
            }
        if self.dimension_critiques is None:
            self.dimension_critiques = {}

    def __getitem__(self, key):
        """Allow dict-like access for backward compatibility."""
        return getattr(self, key)

    def __setitem__(self, key, value):
        """Allow dict-like assignment for backward compatibility."""
        setattr(self, key, value)

    def get(self, key, default=None):
        """Dict-like get method for backward compatibility."""
        return getattr(self, key, default)

class SessionManager:
    """
    Manages PACT critique sessions and their progress.
    """

    def __init__(self, storage_dir: str = "critique_sessions"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.sessions: Dict[str, CritiqueSession] = {}
        self.load_sessions()

    def create_session(self, title: str, mode: str = "STANDARD", **kwargs) -> CritiqueSession:
        """Create a new critique session."""
        session_id = str(uuid.uuid4())

        session = CritiqueSession(
            session_id=session_id,
            status=CritiqueStatus.PENDING,
            paper_content=kwargs.get('paper_content', ''),
            paper_title=title,
            paper_type=kwargs.get('paper_type'),
            created_at=datetime.now(),
            updated_at=datetime.now(),
            mode=mode
        )

        # Handle any additional kwargs
        for key, value in kwargs.items():
            if hasattr(session, key):
                setattr(session, key, value)

        self.sessions[session_id] = session
        logger.info(f"Created session {session_id} with title: {title}, mode: {mode}")

        return session

    def get_session(self, session_id: str) -> Optional[CritiqueSession]:
        """Get a session by ID."""
        return self.sessions.get(session_id)

    def get(self, session_id: str) -> Optional[CritiqueSession]:
        """Get a session by ID - alias for backward compatibility."""
        return self.sessions.get(session_id)

    def update_session_status(self, session_id: str, status,
                            current_stage: str = "", overall_progress: float = None, **kwargs) -> bool:
        """Update session status and progress."""
        session = self.sessions.get(session_id)
        if not session:
            return False

        # Handle both string and enum status values with mapping for common aliases
        if isinstance(status, str):
            status_map = {
                "running": CritiqueStatus.PROCESSING,
                "complete": CritiqueStatus.COMPLETED,
                "error": CritiqueStatus.FAILED
            }
            # Use mapped value if exists, otherwise try to create enum directly
            mapped_status = status_map.get(status)
            if mapped_status:
                session.status = mapped_status
            else:
                try:
                    session.status = CritiqueStatus(status)
                except ValueError:
                    # If status string doesn't match any enum value, default to PENDING
                    session.status = CritiqueStatus.PENDING
        else:
            session.status = status
        session.updated_at = datetime.now()

        if current_stage:
            session.current_stage = current_stage

        if overall_progress is not None:
            session.overall_progress = overall_progress

        # Handle additional keyword arguments
        for key, value in kwargs.items():
            if hasattr(session, key):
                setattr(session, key, value)

        self.save_session(session)
        return True

    def update_agent_status(self, session_id: str, agent_id: str,
                           status: AgentStatus, message: str = "", progress: float = None) -> bool:
        """Update individual agent status and message."""
        session = self.sessions.get(session_id)
        if not session or agent_id not in session.agents:
            return False

        agent = session.agents[agent_id]
        agent.status = status
        agent.message = message
        session.updated_at = datetime.now()

        if progress is not None:
            agent.progress = progress

        if status == AgentStatus.ACTIVE and agent.start_time is None:
            agent.start_time = datetime.now()
        elif status in [AgentStatus.COMPLETED, AgentStatus.FAILED]:
            agent.end_time = datetime.now()

        self.save_session(session)
        return True

    def set_critique_results(self, session_id: str, dimension_critiques: Dict[str, Any] = None,
                            final_critique: str = None, overall_score: float = None) -> bool:
        """Set the critique results for a session."""
        session = self.sessions.get(session_id)
        if not session:
            return False

        if dimension_critiques:
            session.dimension_critiques.update(dimension_critiques)

        if final_critique:
            session.final_critique = final_critique

        if overall_score is not None:
            session.overall_score = overall_score

        session.updated_at = datetime.now()
        self.save_session(session)
        return True

    def update_progress(self, session_id: str, progress: int, status: str = None):
        """Update session progress and optionally status."""
        session = self.sessions.get(session_id)
        if not session:
            return

        session.overall_progress = float(progress)
        if status:
            # Handle both string and enum status values with mapping for common aliases
            if isinstance(status, str):
                status_map = {
                    "running": CritiqueStatus.PROCESSING,
                    "complete": CritiqueStatus.COMPLETED,
                    "error": CritiqueStatus.FAILED
                }
                # Use mapped value if exists, otherwise try to create enum directly
                mapped_status = status_map.get(status)
                if mapped_status:
                    session.status = mapped_status
                else:
                    try:
                        session.status = CritiqueStatus(status)
                    except ValueError:
                        # If status string doesn't match any enum value, default to PENDING
                        session.status = CritiqueStatus.PENDING
            else:
                session.status = status

        if status == "completed":
            session.updated_at = datetime.now()

        session.updated_at = datetime.now()
        self.save_session(session)

    def update_session_results(self, session_id: str, result: Dict[str, Any]):
        """Update session with critique results."""
        session = self.sessions.get(session_id)
        if not session:
            raise KeyError(f"Unknown session_id: {session_id}")

        # Store the complete result with enums coerced to strings
        session.result = _coerce_enums(result)

        # Update individual result fields for backward compatibility
        if 'overall_score' in result:
            session.overall_score = result['overall_score']
        if 'final_critique' in result:
            session.final_critique = result['final_critique']
        if 'dimension_critiques' in result:
            session.dimension_critiques = result['dimension_critiques']

        session.updated_at = datetime.now()
        self.save_session(session)

    def set_error(self, session_id: str, error_message: str) -> bool:
        """Set error state for a session."""
        session = self.sessions.get(session_id)
        if not session:
            return False

        session.status = CritiqueStatus.FAILED
        session.error_message = error_message
        session.retry_count += 1
        session.updated_at = datetime.now()

        self.save_session(session)
        return True

    def get_session_progress(self, session_id: str) -> Dict[str, Any]:
        """Get detailed progress information for a session."""
        session = self.sessions.get(session_id)
        if not session:
            return {"error": "Session not found"}

        return {
            "session_id": session_id,
            "status": session.status,  # String enum, no .value needed
            "overall_progress": session.overall_progress,
            "current_stage": session.current_stage,
            "agents": {
                agent_id: {
                    "name": agent.agent_name,
                    "status": agent.status,  # String enum, no .value needed
                    "message": agent.message,
                    "progress": agent.progress
                }
                for agent_id, agent in session.agents.items()
            },
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "error_message": session.error_message
        }

    def get_session_results(self, session_id: str) -> Dict[str, Any]:
        """Get the final results for a completed session."""
        session = self.sessions.get(session_id)
        if not session:
            return {"error": "Session not found"}

        if session.status != CritiqueStatus.COMPLETED:
            return {"error": "Session not completed yet"}

        return {
            "session_id": session_id,
            "paper_title": session.paper_title,
            "overall_score": session.overall_score,
            "final_critique": session.final_critique,
            "dimension_critiques": session.dimension_critiques,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat()
        }

    def attach_text(self, session_id: str, text: str):
        """Attach original text to session for comprehensive reports."""
        s = self.sessions.get(session_id)
        if s:
            s.original_text = text # Directly assign to the attribute
            self._save_session(session_id)

    def cleanup_old_sessions(self, max_age_hours: int = 24) -> int:
        """Remove sessions older than max_age_hours."""
        if not os.path.exists(self.storage_dir): # Corrected attribute name
            return 0

        current_time = datetime.now()
        removed_count = 0

        for filename in os.listdir(self.storage_dir): # Corrected attribute name
            if not filename.endswith('.json'):
                continue

            file_path = os.path.join(self.storage_dir, filename) # Corrected attribute name
            try:
                # Check file modification time
                file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                age_hours = (current_time - file_mtime).total_seconds() / 3600

                if age_hours > max_age_hours:
                    os.remove(file_path)
                    session_id = filename[:-5]  # Remove .json extension
                    if session_id in self.sessions:
                        del self.sessions[session_id]
                    removed_count += 1
                    logger.info(f"Cleaned up old session: {session_id}")

            except Exception as e:
                logger.error(f"Error cleaning up session file {filename}: {e}")

        return removed_count

    def get_session_results(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get the final results for a session."""
        session = self.sessions.get(session_id)
        return session.result if session else None

    def save_session(self, session: CritiqueSession):
        """Save session to disk."""
        session_file = self.storage_dir / f"{session.session_id}.json"

        # Convert session to dict, handling dataclasses and enums
        session_data = asdict(session)
        # String-based enums are already JSON-serializable, no need for enum_value
        session_data['status'] = session.status
        session_data['created_at'] = session.created_at.isoformat()
        session_data['updated_at'] = session.updated_at.isoformat()
        session_data['mode'] = getattr(session, 'mode', 'STANDARD')
        # Ensure original_text is included and serializable
        session_data['original_text'] = session.original_text

        # Convert agent data - handle both AgentProgress objects and other types
        for agent_id, agent in session_data['agents'].items():
            if isinstance(agent, dict):
                # Already a dict, just convert times (status is already string-serializable)
                if agent.get('start_time'):
                    agent['start_time'] = agent['start_time'].isoformat() if hasattr(agent['start_time'], 'isoformat') else agent['start_time']
                if agent.get('end_time'):
                    agent['end_time'] = agent['end_time'].isoformat() if hasattr(agent['end_time'], 'isoformat') else agent['end_time']
            elif hasattr(agent, '__dict__'):
                # Convert object to dict first if it's not already a dict
                agent_dict = asdict(agent) if hasattr(agent, '__dataclass_fields__') else agent.__dict__
                # String-based enums are already JSON-serializable
                if agent_dict.get('start_time'):
                    agent_dict['start_time'] = agent_dict['start_time'].isoformat() if hasattr(agent_dict['start_time'], 'isoformat') else agent_dict['start_time']
                if agent_dict.get('end_time'):
                    agent_dict['end_time'] = agent_dict['end_time'].isoformat() if hasattr(agent_dict['end_time'], 'isoformat') else agent_dict['end_time']
                session_data['agents'][agent_id] = agent_dict

        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2)

    def _save_session(self, session_id: str):
        """Internal helper to save a specific session."""
        session = self.sessions.get(session_id)
        if session:
            self.save_session(session)
        else:
            logger.warning(f"Attempted to save non-existent session: {session_id}")


    def load_sessions(self):
        """Load all sessions from disk."""
        for session_file in self.storage_dir.glob("*.json"):
            try:
                with open(session_file, 'r') as f:
                    data = json.load(f)

                # Convert back to proper types
                data['status'] = CritiqueStatus(data['status'])
                data['created_at'] = datetime.fromisoformat(data['created_at'])
                data['updated_at'] = datetime.fromisoformat(data['updated_at'])
                data['mode'] = data.get('mode', 'STANDARD')
                # Load original_text if it exists
                data['original_text'] = data.get('original_text')

                # Convert agent data
                agents = {}
                for agent_id, agent_data in data['agents'].items():
                    agent = AgentProgress(
                        agent_id=agent_data['agent_id'],
                        agent_name=agent_data['agent_name'],
                        status=AgentStatus(agent_data['status']),
                        message=agent_data['message'],
                        progress=agent_data['progress'],
                        start_time=datetime.fromisoformat(agent_data['start_time']) if agent_data.get('start_time') else None,
                        end_time=datetime.fromisoformat(agent_data['end_time']) if agent_data.get('end_time') else None
                    )
                    agents[agent_id] = agent

                data['agents'] = agents
                session = CritiqueSession(**data)
                self.sessions[session.session_id] = session

            except Exception as e:
                logger.error(f"Failed to load session from {session_file}: {e}")

# Global session manager instance
session_manager = SessionManager()