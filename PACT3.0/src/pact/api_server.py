"""
FastAPI Backend Server for PACT Critique System

Provides REST API endpoints and WebSocket support for real-time progress tracking.
"""

import asyncio
import os
import uuid
import json
from typing import Dict, Any, Optional
from datetime import datetime
from contextlib import asynccontextmanager
from enum import Enum

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect, Query, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from starlette.responses import PlainTextResponse
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field, model_validator
import uvicorn
import io
from docx import Document
import PyPDF2
import logging
from langchain_core.messages import HumanMessage

# Configure logger
logger = logging.getLogger("pact.api_server")

def _err_payload(exc: Exception):
    return {"detail": "internal_error",
            "error": {"type": exc.__class__.__name__, "message": str(exc)}}

# Local imports for PACT components
try:
    from .pact_critique_agent import pact_critique_agent
    from .session_manager import session_manager
    from .websocket_manager import manager
    from .pdf_generator import generate_pact_pdf_report
    from .mode_config import AgentMode, mode_config
    from .supervisors.real_supervisor import RealCritiqueSupervisor
    from .supervisors.mock_supervisor import MockCritiqueSupervisor
    from .utils.enum_safety import enum_value
    logger.info("WS manager id (api_server)=%s", id(manager))
    MOCK_MODE = False
except ImportError:
    logger.warning("PACT modules not available, using mock mode")
    MOCK_MODE = True

if MOCK_MODE:
    # Fallback mock implementations
    class MockWebSocketManager:
        def __init__(self):
            self.connections = {}
            self.event_queues = {}

        async def connect(self, websocket: WebSocket, session_id: str):
            await websocket.accept()
            self.connections[session_id] = websocket
            # Flush any queued events
            if session_id in self.event_queues:
                for event in self.event_queues[session_id]:
                    try:
                        await websocket.send_json(event)
                    except Exception as e:
                        logger.error(f"Error flushing event: {e}")
                del self.event_queues[session_id]
            logger.info(f"WebSocket connected for session {session_id}")

        async def disconnect(self, session_id: str): # Changed method signature to be async
            if session_id in self.connections:
                del self.connections[session_id]
            logger.info(f"WebSocket disconnected for session {session_id}")

        async def send_message(self, session_id: str, message: dict):
            if session_id in self.connections:
                try:
                    await self.connections[session_id].send_json(message)
                except Exception as e:
                    logger.error(f"Error sending message to {session_id}: {e}")
                    await self.disconnect(session_id) # Changed to await disconnect
            else:
                # Queue the message for when WebSocket connects
                if session_id not in self.event_queues:
                    self.event_queues[session_id] = []
                self.event_queues[session_id].append(message)
                logger.debug(f"Queued message for session {session_id}")

        async def broadcast(self, session_id: str, message: dict):
            """ Broadcast message to all connected clients except the sender if session_id is provided.
                If session_id is None, broadcast to all clients.
            """
            if session_id:
                for sess_id, conn in self.connections.items():
                    if sess_id != session_id:
                        try:
                            await conn.send_json(message)
                        except Exception as e:
                            logger.error(f"Error broadcasting message to {sess_id}: {e}")
                            await self.disconnect(sess_id)
            else: # Broadcast to all
                for sess_id, conn in self.connections.items():
                    try:
                        await conn.send_json(message)
                    except Exception as e:
                        logger.error(f"Error broadcasting message to {sess_id}: {e}")
                        await self.disconnect(sess_id)

    class MockSessionManager:
        def __init__(self):
            self.sessions = {}

        def create_session(self, title: str, mode: str = "STANDARD", paper_content: str = None, paper_type: str = None, **kwargs):
            session_id = str(uuid.uuid4())
            session_dict = {
                "session_id": session_id,
                "id": session_id,  # Add id property for compatibility
                "paper_title": title,
                "mode": mode,
                "status": "pending", # Default status, matches enum
                "progress": 0,  # Add progress property
                "paper_content": paper_content, # Added for completeness
                "paper_type": paper_type, # Added for completeness
                "agents": {},
                "result": None,
                "error": None,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            self.sessions[session_id] = session_dict

            # Create a mock session object with dict-like access
            class MockSession:
                def __init__(self, data):
                    self.__dict__.update(data)
                def __getitem__(self, key):
                    return getattr(self, key)
                def __setitem__(self, key, value):
                    setattr(self, key, value)
                def get(self, key, default=None):
                    return getattr(self, key, default)

            return MockSession(session_dict)

        def get_session(self, session_id: str):
            return self.sessions.get(session_id)

        def update_session_status(self, session_id: str, status: str, **kwargs):
            if session_id in self.sessions:
                self.sessions[session_id]["status"] = status
                self.sessions[session_id]["updated_at"] = datetime.now().isoformat()
                for key, value in kwargs.items():
                    self.sessions[session_id][key] = value

        def update_session_results(self, session_id: str, result_data: Dict[str, Any]):
            """Updates session with critique results."""
            if session_id in self.sessions:
                self.sessions[session_id].update(result_data)
                self.sessions[session_id]["updated_at"] = datetime.now().isoformat()

        def update_progress(self, session_id: str, progress: int, status: str, error_message: Optional[str] = None):
            """Updates session progress and status."""
            if session_id in self.sessions:
                self.sessions[session_id]["overall_progress"] = progress
                self.sessions[session_id]["status"] = status
                if error_message:
                    self.sessions[session_id]["error_message"] = error_message
                self.sessions[session_id]["updated_at"] = datetime.now().isoformat()

        def update_agent_status(self, session_id: str, agent_id: str, agent_status: str, message: str):
            """Updates the status of a specific agent within a session."""
            if session_id in self.sessions:
                if "agents" not in self.sessions[session_id]:
                    self.sessions[session_id]["agents"] = {}
                self.sessions[session_id]["agents"][agent_id] = {"status": agent_status, "message": message}
                self.sessions[session_id]["updated_at"] = datetime.now().isoformat()


        def cleanup_old_sessions(self, max_age_hours: int = 24) -> int:
            return 0

    # Create mock implementations
    class MockCritiqueSupervisor:
        async def ainvoke(self, inputs, session_id=None):
            """Mock supervisor that returns sample results."""
            return {
                "overall_score": 78.5,
                "final_critique": "This is a mock comprehensive critique of the paper.",
                "dimension_critiques": {
                    "1.0.0": {"dimension_name": "Research Foundations", "dimension_score": 80},
                    "2.0.0": {"dimension_name": "Methodological Rigor", "dimension_score": 75},
                    "3.0.0": {"dimension_name": "Structure & Coherence", "dimension_score": 82},
                    "4.0.0": {"dimension_name": "Academic Precision", "dimension_score": 77},
                    "5.0.0": {"dimension_name": "Critical Sophistication", "dimension_score": 79}
                }
            }
    
    session_manager = MockSessionManager()
    manager = MockWebSocketManager()  # Set manager for compatibility
else:
    # Assuming 'manager' is correctly imported from websocket_manager
    # If not, this will need to be defined or imported correctly.
    # For now, we'll assume 'manager' is available and properly initialized.
    # If 'manager' is not globally available or imported, this will cause an error.
    # It's better to explicitly import or initialize it here if it's not in the global scope.
    try:
        from .websocket_manager import manager # Ensure manager is imported
        manager = manager  # manager is already imported
        logger.info("Using real WebSocket manager.")
    except ImportError:
        logger.error("Failed to import WebSocket manager.")
        # Provide a fallback or raise an error if the manager is critical
        raise


def create_comprehensive_critique(result: Dict[str, Any]) -> Dict[str, Any]:
    """Mock function to create comprehensive critique."""
    logger.info("Creating comprehensive critique...")
    return {
        "document_title": result.get("paper_title", "Untitled Paper"),
        "analysis_date": datetime.now().isoformat(),
        "analysis_type": "Comprehensive",
        "overall_assessment": "Strong" if result.get("overall_score", 0) >= 80 else "Competent",
        "overall_score": result.get("overall_score", 75),
        "submission_readiness": {
            "overall_readiness": "Ready" if result.get("overall_score", 0) >= 85 else "Minor Revisions Needed",
            "readiness_score": min(5, max(1, result.get("overall_score", 75) / 20)),
            "recommendation": "Accept" if result.get("overall_score", 0) >= 85 else "Revise",
            "justification": "Based on comprehensive PACT analysis",
            "estimated_revision_time": "1-2 weeks"
        },
        "dimension_analyses": result.get("dimension_critiques", {}),
        "executive_summary": result.get("final_critique", "")[:500] + ("..." if len(result.get("final_critique", "")) > 500 else ""),
        "key_findings": ["Analysis completed successfully"],
        "checklist_items": [],
        "next_steps": ["Review detailed feedback", "Address priority improvements"],
        "total_strengths_identified": len(result.get("dimension_critiques", {})) + 1, # Mock value
        "total_improvements_needed": len(result.get("dimension_critiques", {})) -1, # Mock value
        "critical_issues_count": 0 # Mock value
    }


# Mock definitions for AgentMode and mode_config
class AgentMode:
    APA7 = "APA7"
    STANDARD = "STANDARD"
    COMPREHENSIVE = "COMPREHENSIVE"

    @classmethod
    def values(cls):
        return [cls.APA7, cls.STANDARD, cls.COMPREHENSIVE]

# Dummy mode_config, actual implementation would define agent configurations per mode
mode_config = {
    AgentMode.APA7: {"agents": ["APAFormatterAgent", "ClarityCheckerAgent"]},
    AgentMode.STANDARD: {"agents": ["ClarityCheckerAgent", "CoherenceCheckerAgent"]},
    AgentMode.COMPREHENSIVE: {"agents": ["ClarityCheckerAgent", "CoherenceCheckerAgent", "MethodologyCheckerAgent", "ContributionCheckerAgent", "WritingStyleCheckerAgent"]},
}

# Supervisor classes are now imported from separate modules

def make_supervisor():
    """Create appropriate supervisor based on configuration."""
    # Check if we should use mock mode based on environment variable or MOCK_MODE flag
    use_mock_env = os.getenv("USE_MOCK", "false").lower() == "true"
    real_mode = not (use_mock_env or MOCK_MODE)

    if real_mode:
        logger.info("Creating real critique supervisor")
        # Create agents and LLM for real supervisor
        from langchain.chat_models import init_chat_model
        llm = init_chat_model(model="openai:gpt-4o", temperature=0.1)
        agents = {}  # Will be populated by the supervisor
        return RealCritiqueSupervisor(agents, llm)
    else:
        logger.info("Creating mock critique supervisor...")
        return MockCritiqueSupervisor()

# Mock update_session_status
async def update_session_status(session_id: str, status: str, progress: int, message: str = ""):
    """Mock function to update session status and notify clients."""
    session_manager.update_session_status(session_id, status, progress=progress, current_stage=message)
    await notify_websocket_clients(session_id)

# Mock generate_html_report and generate_markdown_report for compatibility
def generate_html_report(results_data: Dict[str, Any]) -> str:
    """Generate an HTML report from results data."""
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>PACT Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
            .header {{ background: #2c4866; color: white; padding: 20px; border-radius: 8px; }}
            .section {{ margin: 20px 0; }}
            .dimension {{ border-left: 4px solid #2c4866; padding-left: 15px; margin: 15px 0; }}
            .score {{ font-weight: bold; color: #2c4866; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>PACT Analysis Report</h1>
            <p>Generated: {datetime.now().strftime('%m/%d/%Y, %I:%M:%S %p')}</p>
            {f"<p>Paper: {results_data.get('paper_title', 'Untitled')}</p>" if results_data.get('paper_title') else ""}
        </div>

        <div class="section">
            <h2>Overall Assessment</h2>
            <p class="score">Overall Score: {results_data.get('overall_score', 'N/A')}/100</p>
            <p>{results_data.get('final_critique', '')}</p>
        </div>

        <div class="section">
            <h2>Dimension Analyses</h2>
            {generate_dimension_html(results_data.get('dimension_critiques', {}))}
        </div>
    </body>
    </html>
    """
    return html_template

def generate_markdown_report(results_data: Dict[str, Any]) -> str:
    """Generate a Markdown report from results data."""
    md_content = f"""
# PACT Analysis Report

**Generated:** {datetime.now().strftime('%m/%d/%Y, %I:%M:%S %p')}
{f"**Paper:** {results_data.get('paper_title', 'Untitled')}" if results_data.get('paper_title') else ""}

## Overall Assessment

**Overall Score:** {results_data.get('overall_score', 'N/A')}/100

{results_data.get('final_critique', '')}

## Dimension Analyses

{generate_dimension_markdown(results_data.get('dimension_critiques', {}))}

---
*Generated by PACT Academic Analysis Tool*
    """
    return md_content.strip()

def generate_dimension_html(dimension_critiques: Dict[str, Any]) -> str:
    """Generate HTML for dimension critiques."""
    html = ""
    for dim_id, critique in dimension_critiques.items():
        html += f"""
        <div class="dimension">
            <h3>{critique.get('dimension_name', dim_id)}</h3>
            <p class="score">Score: {critique.get('dimension_score', 'N/A')}/100</p>
            <p><strong>Strengths:</strong></p>
            <ul>
                {generate_list_html(critique.get('strengths', []))}
            </ul>
            <p><strong>Areas for Improvement:</strong></p>
            <ul>
                {generate_list_html(critique.get('weaknesses', []))}
            </ul>
        </div>
        """
    return html

def generate_dimension_markdown(dimension_critiques: Dict[str, Any]) -> str:
    """Generate Markdown for dimension critiques."""
    md = ""
    for dim_id, critique in dimension_critiques.items():
        # Handle both dimension_name and dimension_label for compatibility
        dimension_name = critique.get('dimension_name') or critique.get('dimension_label', dim_id)
        score = critique.get('dimension_score') or critique.get('overall_score', 'N/A')

        md += f"""
### {dimension_name}

**Score:** {score}/100

**Strengths:**
{generate_list_markdown(critique.get('strengths', []))}

**Areas for Improvement:**
{generate_list_markdown(critique.get('weaknesses', []))}

        """
    return md

def generate_list_html(items: list) -> str:
    """Generate HTML list items."""
    return ''.join([f"<li>{item}</li>" for item in items])

def generate_list_markdown(items: list) -> str:
    """Generate Markdown list items."""
    return '\n'.join([f"- {item}" for item in items])

def convert_to_comprehensive_critique(results_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert API results to comprehensive critique format."""
    from datetime import datetime

    # This is a simplified conversion - in practice, you'd want to
    # map all the detailed fields from the enhanced analysis
    return {
        "document_title": results_data.get("paper_title"),
        "analysis_date": datetime.now().isoformat(),
        "analysis_type": "Comprehensive",
        "overall_assessment": "Strong" if results_data.get("overall_score", 0) >= 80 else "Competent",
        "overall_score": results_data.get("overall_score", 75),
        "submission_readiness": {
            "overall_readiness": "Ready" if results_data.get("overall_score", 0) >= 85 else "Minor Revisions Needed",
            "readiness_score": min(5, max(1, results_data.get("overall_score", 75) / 20)),
            "recommendation": "Accept" if results_data.get("overall_score", 0) >= 85 else "Revise",
            "justification": "Based on comprehensive PACT analysis",
            "estimated_revision_time": "1-2 weeks"
        },
        "dimension_analyses": results_data.get("dimension_critiques", {}),
        "executive_summary": results_data.get("final_critique", "")[:500] + "...",
        "key_findings": ["Analysis completed successfully"],
        "checklist_items": [],
        "next_steps": ["Review detailed feedback", "Address priority improvements"],
        "total_strengths_identified": 5,
        "total_improvements_needed": 3,
        "critical_issues_count": 0
    }


# ===== API MODELS =====

# Define RunStatus Enum as per the user message
class RunStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    error = "error"

class CritiqueRequest(BaseModel):
    content: str = Field(description="The academic paper content to critique")
    title: Optional[str] = Field(default=None, description="Title of the paper")
    paper_type: Optional[str] = Field(default=None, description="Type of the paper")
    mode: str = Field(default="STANDARD", description="Analysis mode: APA7, STANDARD, or COMPREHENSIVE")

class StartCritiqueRequest(BaseModel):
    title: str
    paper_text: Optional[str] = None
    paper_content: Optional[str] = None
    mode: str = "STANDARD"
    paper_type: str = "research"

    @model_validator(mode="after")
    def ensure_text_present(self):
        if not (self.paper_text or self.paper_content):
            raise ValueError("Provide paper_text or paper_content")
        return self

    @property
    def text(self) -> str:
        return (self.paper_text or self.paper_content or "").strip()

class CritiqueResponse(BaseModel):
    session_id: str
    status: str
    message: str

class ProgressResponse(BaseModel):
    session_id: str
    status: str
    overall_progress: float
    current_stage: str
    agents: Dict[str, Any]
    created_at: str
    updated_at: str
    error_message: Optional[str] = None

class ResultsResponse(BaseModel):
    session_id: str
    paper_title: Optional[str]
    overall_score: Optional[float]
    final_critique: Optional[str]
    dimension_critiques: Dict[str, Any]
    created_at: str
    updated_at: str

# Global WebSocket manager
# websocket_manager = WebSocketManager() # Replaced by MockWebSocketManager

# Cleanup task
async def cleanup_old_sessions():
    """Background task to cleanup old sessions."""
    while True:
        try:
            # Assuming session_manager has a cleanup_old_sessions method
            removed = session_manager.cleanup_old_sessions(max_age_hours=24)
            if removed > 0:
                logger.info(f"Cleaned up {removed} old sessions")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        await asyncio.sleep(3600)  # Run every hour

# App lifespan manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    cleanup_task = asyncio.create_task(cleanup_old_sessions())
    logger.info("Starting PACT Critique API server...")
    yield
    # Shutdown
    logger.info("Shutting down PACT Critique API server...")
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass

# Create FastAPI app
app = FastAPI(
    title="PACT Critique API",
    description="Multi-Agent Academic Paper Critique System using PACT Taxonomy",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error("422 on %s: %s", request.url, exc.errors())
    return JSONResponse(status_code=422,
                        content=jsonable_encoder({"detail": "validation_error", "errors": exc.errors()}))

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("500 on %s", request.url)
    return JSONResponse(status_code=500, content=jsonable_encoder(_err_payload(exc)))

# Serve static files (the HTML app)
# Assuming pact_critique_app.html is in the root directory
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/")
async def serve_app():
    """Serve the main PACT critique application."""
    try:
        return FileResponse("pact_critique_app.html")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="PACT critique app interface not found. Ensure 'pact_critique_app.html' is in the root directory.")


# ===== API ENDPOINTS =====

@app.post("/api/upload", response_model=Dict[str, str])
async def upload_file(file: UploadFile = File(...)):
    """
    Upload and extract text content from uploaded files (DOCX, PDF, TXT).
    """
    logger.info(f"Processing uploaded file: {file.filename}, content-type: {file.content_type}, size: {file.size}")

    try:
        # Validate file size
        if file.size and file.size > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(status_code=400, detail="File size exceeds 10MB limit.")

        content = await file.read()
        logger.info(f"Read {len(content)} bytes from file {file.filename}")

        if file.filename.lower().endswith('.docx'):
            # Extract text from DOCX
            try:
                doc = Document(io.BytesIO(content))
                text_content = []
                for paragraph in doc.paragraphs:
                    if paragraph.text.strip():
                        text_content.append(paragraph.text)
                extracted_text = '\n'.join(text_content)
                logger.info(f"Extracted {len(extracted_text)} characters from DOCX file")
            except Exception as docx_error:
                logger.error(f"Error processing DOCX file: {docx_error}")
                raise HTTPException(status_code=400, detail="Failed to process DOCX file. Please ensure it's a valid Word document.")

        elif file.filename.lower().endswith('.pdf'):
            # Extract text from PDF
            try:
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
                text_content = []
                for i, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text_content.append(page_text)
                extracted_text = '\n'.join(text_content)
                logger.info(f"Extracted {len(extracted_text)} characters from PDF file with {len(pdf_reader.pages)} pages")
            except Exception as pdf_error:
                logger.error(f"Error processing PDF file: {pdf_error}")
                raise HTTPException(status_code=400, detail="Failed to process PDF file. Please ensure it's a valid PDF document.")

        elif file.filename.lower().endswith('.txt'):
            # Handle plain text
            try:
                extracted_text = content.decode('utf-8')
                logger.info(f"Processed TXT file with {len(extracted_text)} characters")
            except UnicodeDecodeError:
                try:
                    extracted_text = content.decode('latin-1')
                    logger.info(f"Processed TXT file with latin-1 encoding, {len(extracted_text)} characters")
                except Exception as txt_error:
                    logger.error(f"Error decoding TXT file: {txt_error}")
                    raise HTTPException(status_code=400, detail="Failed to decode text file. Please ensure it's a valid text file.")

        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload TXT, DOCX, or PDF files.")

        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail="No text content could be extracted from the file.")

        logger.info(f"Successfully processed file {file.filename}")
        return {
            "content": extracted_text,
            "filename": file.filename,
            "message": "File processed successfully"
        }

    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        logger.error(f"Unexpected error processing file {file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")

@app.post("/api/critique/start")
async def start_critique(req: StartCritiqueRequest, request: Request):
    try:
        logger.info(f"Starting critique for paper: {req.title[:50]}...")

        # Check required fields
        if not req.text or not req.text.strip():
            raise HTTPException(status_code=400, detail="Paper text is required")

        if len(req.text.strip()) < 100:
            raise HTTPException(status_code=400, detail="Paper text is too short (minimum 100 characters)")

        # Create session
        session = session_manager.create_session(
            title=req.title,
            text=req.text,
            mode=req.mode,
            paper_type=req.paper_type
        )
        
        # Extract session_id properly
        session_id = session.session_id if hasattr(session, 'session_id') else str(session)
        logger.info(f"Created session {session_id} with title: {req.title}, mode: {req.mode}")
        logger.info(f"Created session {session}")

        # Start background analysis
        asyncio.create_task(run_critique_analysis(session_id, req.text, req.title, req.mode))

        return {"session_id": session_id, "status": "started"}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error starting critique")
        return JSONResponse(status_code=500, content=jsonable_encoder(_err_payload(e)))

@app.get("/api/critique/status/{session_id}")
async def get_critique_status(session_id: str):
    """Get the current status of a critique session."""
    try:
        session = session_manager.get_session(session_id)
        if not session:
            return JSONResponse(
                status_code=404,
                content={"detail": "Session not found", "session_id": session_id}
            )

        # Check if session exists but has no results yet
        if session.status == "pending" or (session.status == "running" and not hasattr(session, 'result')):
            return JSONResponse(
                status_code=409,
                content={"detail": "Analysis not ready yet", "session_id": session_id, "status": session.status}
            )

        # String-based enums are directly JSON-serializable
        current_status = {
            "session_id": session_id,
            "title": getattr(session, 'paper_title', 'Untitled'),
            "mode": getattr(session, 'mode', 'STANDARD'),
            "status": session.status,  # No .value needed - string enum is JSON-serializable
            "progress": getattr(session, 'overall_progress', 0),
            "has_result": hasattr(session, 'result') and session.result is not None,
        }

        return JSONResponse(content=current_status)

    except Exception as e:
        logger.exception("Error getting critique status for session %s", session_id)
        return JSONResponse(
            status_code=500,
            content={"detail": f"Internal error: {str(e)}", "session_id": session_id}
        )

@app.get("/api/critique/results/{session_id}", response_model=ResultsResponse)
async def get_critique_results(session_id: str):
    """
    Get the final results of a completed critique session.
    """
    logger.info(f"Getting results for session {session_id}")
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    logger.info(f"Session status: {session.status}")
    # String-based enums can be compared directly with strings
    if session.status != "completed":
        raise HTTPException(status_code=400, detail="Session not yet completed")

    # Check both individual fields and result object
    has_score = getattr(session, 'overall_score', None) is not None
    has_critique = getattr(session, 'final_critique', None) is not None
    has_result = getattr(session, 'result', None) is not None
    
    logger.info(f"Has score: {has_score}, has critique: {has_critique}, has result: {has_result}")
    
    if not has_score and not has_critique and not has_result:
        raise HTTPException(status_code=500, detail="Results not found for completed session")

    # Get data from either individual fields or result object
    result_data = getattr(session, 'result', {}) or {}
    
    return ResultsResponse(**{
        "session_id": session_id,
        "paper_title": getattr(session, 'paper_title', None) or result_data.get('paper_title'),
        "overall_score": getattr(session, 'overall_score', None) or result_data.get('overall_score'),
        "final_critique": getattr(session, 'final_critique', None) or result_data.get('final_critique'),
        "dimension_critiques": getattr(session, 'dimension_critiques', {}) or result_data.get('dimension_critiques', {}),
        "created_at": session.created_at.isoformat() if hasattr(session, 'created_at') and session.created_at else None,
        "updated_at": session.updated_at.isoformat() if hasattr(session, 'updated_at') and session.updated_at else None
    })

@app.websocket("/api/critique/live/{session_id}")
async def critique_progress_websocket(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time progress updates.
    """
    logger.info(f"WebSocket connection requested for session: {session_id}")
    
    try:
        await manager.connect(session_id, websocket)
        logger.info(f"WebSocket connected successfully for session: {session_id}")

        # Send initial status
        session = session_manager.get_session(session_id)
        if session:
            logger.info(f"Sending initial status for session {session_id}, status: {session.status}")
            # String-based enums are directly JSON-serializable
            current_status = {
                "event": "status",
                "session_id": session_id,
                "state": session.status,  # No .value needed - string enum
                "status": session.status,
                "paper_title": getattr(session, 'paper_title', 'Unknown'),
                "created_at": session.created_at.isoformat() if hasattr(session, 'created_at') and session.created_at else None,
                "updated_at": session.updated_at.isoformat() if hasattr(session, 'updated_at') and session.updated_at else None,
                "error_message": getattr(session, 'error_message', None),
                "progress": getattr(session, 'overall_progress', 0)
            }
            if session.status == "completed":
                current_status["result"] = {
                    "overall_score": getattr(session, 'overall_score', None),
                    "final_critique": getattr(session, 'final_critique', None),
                    "dimension_critiques": getattr(session, 'dimension_critiques', {})
                }
            await manager.send_message(session_id, current_status)
        else:
            logger.warning(f"Session {session_id} not found for WebSocket connection")
            await websocket.close(code=1000, reason="Session not found")
            return

        # Keep connection alive
        while True:
            try:
                # Wait for ping/pong or other messages from client
                data = await websocket.receive_text()
                logger.debug(f"Received WebSocket message: {data}")
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected cleanly for session {session_id}")
                break
            except Exception as e:
                logger.error(f"Error receiving message on WebSocket for session {session_id}: {e}")
                break

    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}", exc_info=True)
        try:
            await websocket.close(code=1011, reason="Internal error")
        except:
            pass
    finally:
        logger.info(f"Cleaning up WebSocket connection for session {session_id}")
        manager.disconnect(session_id, websocket)

def _build_markdown_report(title: str, result: Dict[str, Any], original_text: str | None) -> str:
    lines = []
    lines.append(f"# PACT Critique — {title}\n")
    if "overall_score" in result:
        lines.append(f"**Overall Score:** {result['overall_score']}\n")
    lines.append("## Executive Summary\n")
    lines.append(result.get("final_critique", "—"))
    lines.append("\n")

    dims = result.get("dimension_critiques") or {}
    if isinstance(dims, dict) and dims:
        lines.append("## Dimension Critiques\n")
        for dim, info in dims.items():
            lines.append(f"### {dim}")
            if isinstance(info, dict):
                score = info.get("score", "—")
                comments = info.get("comments") or info.get("critique") or "—"
                recs = info.get("recommendations") or []
                lines.append(f"- **Score:** {score}")
                lines.append(f"- **Critique:** {comments}")
                if recs:
                    lines.append("- **Recommendations:**")
                    for r in recs:
                        lines.append(f"  - {r}")
            else:
                lines.append(str(info))
            lines.append("")
    if original_text:
        lines.append("## Original Submission\n")
        lines.append("```text")
        lines.append(original_text)
        lines.append("```")
    lines.append(f"\n_Generated: {datetime.utcnow().isoformat()}Z_")
    return "\n".join(lines)

@app.get("/api/critique/download/{session_id}")
async def download_report(session_id: str, format: str = "json"):
    s = session_manager.get_session(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Unknown session")
    if not getattr(s, "result", None):
        # 409: the session exists but isn't ready
        raise HTTPException(status_code=409, detail="No analysis results for this session yet")

    fmt = (format or "json").lower()
    if fmt in ("json", "raw"):
        payload = {
            "session_id": s.get("session_id") or s.get("id"),
            "title": s.get("paper_title"),
            "mode": enum_value(s.get("mode", "STANDARD")),
            "status": enum_value(s.get("status", "completed")),
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "result": s.get("result"),  # s.result should already be JSON-safe
        }
        headers = {"Content-Disposition": f'attachment; filename="PACT-{s.get("session_id", session_id)}.json"'}
        return JSONResponse(payload, headers=headers)

    if fmt in ("md", "markdown", "comprehensive"):
        original_text = s.get("original_text", None)
        md = _build_markdown_report(s.get("paper_title", "Untitled"), s.get("result", {}), original_text)
        headers = {"Content-Disposition": f'attachment; filename="PACT-{s.get("session_id", session_id)}.md"'}
        return StreamingResponse(
            io.StringIO(md),
            headers=headers,
            media_type="text/markdown; charset=utf-8"
        )

    raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")

@app.get("/api/critique/preview/{session_id}")
async def preview_critique_report(session_id: str):
    """
    Get a preview of the critique report (HTML format).
    """
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # String-based enums can be compared directly with strings
    if session.status != "completed":
        raise HTTPException(status_code=400, detail="Session not yet completed")

    results_data = getattr(session, 'result', None)
    if not results_data:
        raise HTTPException(status_code=500, detail="Results not found for completed session")

    try:
        html_content = generate_html_report(results_data)
        return {"preview_html": html_content}

    except Exception as e:
        logger.error(f"Failed to generate preview for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate preview: {str(e)}")

# ===== BACKGROUND TASK FUNCTIONS =====

async def run_critique_analysis(session_id: str, paper_content: str, paper_title: str, mode: str):
    """Background task to run the critique analysis."""
    session = None
    try:
        logger.info("Launching supervisor task for session %s", session_id)

        session = session_manager.get_session(session_id)
        if not session:
            raise Exception(f"Session {session_id} not found")

        supervisor = make_supervisor()

        session_manager.update_session_status(session_id, "running", overall_progress=1)
        await manager.send_message(session_id, {
            "event": "status",
            "status": "running",
            "progress": 1
        })

        result = await supervisor.ainvoke(
            {"paper_content": paper_content, "paper_title": paper_title, "mode": mode},
            session_id=session_id
        )

        # Store the result properly
        session_manager.update_session_results(session_id, result)
        
        # Also update individual fields for compatibility
        session = session_manager.get_session(session_id)
        if session:
            session.result = result
            session.overall_score = result.get('overall_score')
            session.final_critique = result.get('final_critique')
            session.dimension_critiques = result.get('dimension_critiques', {})

        session_manager.update_progress(session_id, 100, status="completed")
        await manager.send_message(session_id, {"event": "status", "status": "completed", "progress": 100})

        await manager.send_message(session_id, {"event": "summary", "payload": result})

        logger.info("Critique completed for session %s", session_id)

    except Exception as e:
        logger.exception("Critique failed for session %s", session_id)
        current_progress = getattr(session, 'overall_progress', 0) if session else 0
        # Use update_session_status which exists in both mock and real managers
        session_manager.update_session_status(session_id, "error", 
                                            overall_progress=current_progress, 
                                            error_message=str(e))
        await manager.send_message(session_id, {
            "event": "status",
            "status": "error",
            "message": str(e)
        })

# Progress callback class
class CritiqueProgressCallback:
    """Callback to track LangGraph execution progress."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.current_node = ""

    async def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs):
        pass

    async def on_chain_end(self, outputs: Dict[str, Any], **kwargs):
        pass

    async def on_llm_start(self, serialized: Dict[str, Any], prompts: list, **kwargs):
        pass

    async def on_llm_end(self, response, **kwargs):
        pass

    async def on_tool_start(self, serialized: Dict[str, Any], args: Dict[str, Any], **kwargs):
        pass

    async def on_tool_end(self, output, **kwargs):
        pass

    async def on_agent_action(self, action, **kwargs):
        logger.debug(f"Agent action: {action}")
        agent_id = action.tool
        message = action.tool_input if isinstance(action.tool_input, str) else str(action.tool_input)
        if agent_id:
            session_manager.update_agent_status(self.session_id, agent_id, "ACTIVE", message)
            await update_session_status(self.session_id, "processing", 50, f"Agent {agent_id} is running...")

    async def on_agent_finish(self, finish, **kwargs):
        pass

async def notify_websocket_clients(session_id: str):
    """Send progress updates to WebSocket clients."""
    session = session_manager.get_session(session_id)
    if session:
        # String-based enums are directly JSON-serializable
        progress_data = {
            "session_id": session_id,
            "state": session.status,  # No .value needed - string enum
            "progress": {
                "overall_progress": getattr(session, 'overall_progress', 0),
                "current_stage": getattr(session, 'current_stage', "Initializing"),
            },
            "agents": getattr(session, 'agents', {}),
            "created_at": session.created_at.isoformat() if session.created_at else None,
            "updated_at": session.updated_at.isoformat() if session.updated_at else None,
            "error_message": getattr(session, 'error_message', None)
        }
        await manager.broadcast(session_id, progress_data)

# ===== HEALTH CHECK =====

@app.post("/api/critique/test-broadcast/{session_id}")
async def test_broadcast(session_id: str):
    """Test endpoint to manually broadcast a message to verify WebSocket manager is working."""
    logger.info("Test broadcast called for session %s, manager id=%s", session_id, id(manager))
    await manager.broadcast(session_id, {
        "event": "progress",
        "progress": 42,
        "message": "Test ping from API server"
    })
    return {"ok": True, "session_id": session_id, "manager_id": id(manager)}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_sessions": len(session_manager.sessions)
    }

# ===== DEVELOPMENT SERVER =====

if __name__ == "__main__":
    logger.info("Starting FastAPI server for PACT Critique System (Development Mode)...")
    uvicorn.run(
        "api_server:app", # Assumes this file is named api_server.py
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )