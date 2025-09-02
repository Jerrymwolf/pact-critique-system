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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CritiqueStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    PLANNING = "planning"
    EVALUATING = "evaluating"
    SYNTHESIZING = "synthesizing"
    COMPLETED = "completed"
    FAILED = "failed"

class AgentStatus(Enum):
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

    # Progress tracking
    overall_progress: float = 0.0
    agents: Dict[str, AgentProgress] = None
    current_stage: str = ""

    # Results
    dimension_critiques: Dict[str, Any] = None
    final_critique: Optional[str] = None
    overall_score: Optional[float] = None

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

    def create_session(self, paper_content: str, paper_title: Optional[str] = None,
                      paper_type: Optional[str] = None) -> str:
        """Create a new critique session."""
        session_id = str(uuid.uuid4())

        session = CritiqueSession(
            session_id=session_id,
            status=CritiqueStatus.PENDING,
            paper_content=paper_content,
            paper_title=paper_title,
            paper_type=paper_type,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

        self.sessions[session_id] = session
        logger.info(f"Created session {session_id} with title: {paper_title}")

        return session_id

    def get_session(self, session_id: str) -> Optional[CritiqueSession]:
        """Get a session by ID."""
        return self.sessions.get(session_id)

    def update_session_status(self, session_id: str, status,
                            current_stage: str = "", overall_progress: float = None, **kwargs) -> bool:
        """Update session status and progress."""
        session = self.sessions.get(session_id)
        if not session:
            return False

        # Handle both string and enum status values
        if isinstance(status, str):
            status_map = {
                "pending": CritiqueStatus.PENDING,
                "processing": CritiqueStatus.PROCESSING,
                "running": CritiqueStatus.PROCESSING,
                "planning": CritiqueStatus.PLANNING,
                "evaluating": CritiqueStatus.EVALUATING,
                "synthesizing": CritiqueStatus.SYNTHESIZING,
                "completed": CritiqueStatus.COMPLETED,
                "complete": CritiqueStatus.COMPLETED,
                "failed": CritiqueStatus.FAILED,
                "error": CritiqueStatus.FAILED
            }
            status = status_map.get(status, CritiqueStatus.PENDING)

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
            "status": session.status.value,
            "overall_progress": session.overall_progress,
            "current_stage": session.current_stage,
            "agents": {
                agent_id: {
                    "name": agent.agent_name,
                    "status": agent.status.value,
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

    def cleanup_old_sessions(self, max_age_hours: int = 24) -> int:
        """Remove sessions older than max_age_hours."""
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        to_remove = []

        for session_id, session in self.sessions.items():
            if session.updated_at < cutoff:
                to_remove.append(session_id)

        for session_id in to_remove:
            del self.sessions[session_id]
            session_file = self.storage_dir / f"{session_id}.json"
            if session_file.exists():
                session_file.unlink()

        return len(to_remove)

    def save_session(self, session: CritiqueSession):
        """Save session to disk."""
        session_file = self.storage_dir / f"{session.session_id}.json"

        # Convert session to dict, handling dataclasses and enums
        session_data = asdict(session)
        session_data['status'] = session.status.value
        session_data['created_at'] = session.created_at.isoformat()
        session_data['updated_at'] = session.updated_at.isoformat()

        # Convert agent data
        for agent_id, agent in session_data['agents'].items():
            agent['status'] = agent['status'].value if hasattr(agent['status'], 'value') else agent['status']
            if agent['start_time']:
                agent['start_time'] = agent['start_time'].isoformat() if hasattr(agent['start_time'], 'isoformat') else agent['start_time']
            if agent['end_time']:
                agent['end_time'] = agent['end_time'].isoformat() if hasattr(agent['end_time'], 'isoformat') else agent['end_time']

        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2)

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

                # Convert agent data
                agents = {}
                for agent_id, agent_data in data['agents'].items():
                    agent = AgentProgress(
                        agent_id=agent_data['agent_id'],
                        agent_name=agent_data['agent_name'],
                        status=AgentStatus(agent_data['status']),
                        message=agent_data['message'],
                        progress=agent_data['progress'],
                        start_time=datetime.fromisoformat(agent_data['start_time']) if agent_data['start_time'] else None,
                        end_time=datetime.fromisoformat(agent_data['end_time']) if agent_data['end_time'] else None
                    )
                    agents[agent_id] = agent

                data['agents'] = agents
                session = CritiqueSession(**data)
                self.sessions[session.session_id] = session

            except Exception as e:
                logger.error(f"Failed to load session from {session_file}: {e}")

# Global session manager instance
session_manager = SessionManager()