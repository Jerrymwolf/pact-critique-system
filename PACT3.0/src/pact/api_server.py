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

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect, Query, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
import uvicorn
import io
from docx import Document
import PyPDF2
import logging
from langchain_core.messages import HumanMessage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import real PACT components
try:
    from .pact_critique_agent import pact_critique_agent
    from .session_manager import session_manager
    from .websocket_manager import manager
    from .pdf_generator import generate_pact_pdf_report
    from .mode_config import AgentMode, mode_config
    from .supervisors.real_supervisor import RealCritiqueSupervisor
    from .supervisors.mock_supervisor import MockCritiqueSupervisor
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

        def create_session(self, paper_content: str, paper_title: str = None, paper_type: str = None, mode: str = "STANDARD", **kwargs):
            session_id = str(uuid.uuid4())
            session_dict = {
                "session_id": session_id,
                "paper_title": paper_title,
                "mode": mode,
                "status": "queued", # Default status
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
                # Convert string status to enum if needed
                if isinstance(status, str):
                    status_enum_map = {
                        "queued": "pending",
                        "running": "processing",
                        "complete": "completed",
                        "error": "failed"
                    }
                    status = status_enum_map.get(status, status)

                self.sessions[session_id]["status"] = status
                self.sessions[session_id]["updated_at"] = datetime.now().isoformat()
                for key, value in kwargs.items():
                    self.sessions[session_id][key] = value

        def cleanup_old_sessions(self, max_age_hours: int = 24) -> int:
            return 0

    session_manager = MockSessionManager()
    websocket_manager = MockWebSocketManager()
else:
    websocket_manager = manager


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
    # Check if we should use real mode
    use_mock = os.getenv("USE_MOCK", "false").lower() == "true"
    real_mode = not (use_mock or MOCK_MODE)

    if real_mode:
        logger.info("Creating real critique supervisor")
        return RealCritiqueSupervisor()
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

class CritiqueRequest(BaseModel):
    content: str = Field(description="The academic paper content to critique")
    title: Optional[str] = Field(default=None, description="Title of the paper")
    paper_type: Optional[str] = Field(default=None, description="Type of the paper")
    mode: Optional[str] = Field(default="STANDARD", description="Analysis mode: APA7, STANDARD, or COMPREHENSIVE")

from pydantic import BaseModel, Field, model_validator
from typing import Optional

class StartCritiqueRequest(BaseModel):
    title: str
    paper_text: Optional[str] = None
    paper_content: Optional[str] = None
    mode: str = "STANDARD"

    @model_validator(mode="after")
    def ensure_text_present(self):
        if not (self.paper_text or self.paper_content):
            # This message will show up in the 422 response
            raise ValueError("Provide paper_text or paper_content")
        return self

    # convenience
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
    return JSONResponse(status_code=422, content={"detail": exc.errors()})

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

@app.post("/api/critique/start", response_model=CritiqueResponse)
async def start_critique(req: StartCritiqueRequest):
    """
    Submit a paper for PACT critique analysis.
    """
    try:
        title = (req.title or "").strip() or "Untitled"
        text = req.text
        if not text:
            # Fallback guard; should be caught by validator already
            raise HTTPException(status_code=422, detail="paper_text or paper_content is required")

        # create the session (whichever signature you use)
        session = session_manager.create_session(title=title, mode=req.mode or "STANDARD")

        # initial status -> broadcast
        session.status = "running"
        session.progress = 1
        await manager.broadcast(session.session_id, {"event":"status","status":"running","progress":1})

        # launch background task
        async def run():
            try:
                supervisor = make_supervisor()
                result = await supervisor.ainvoke(
                    {"paper_title": title, "paper_content": text, "mode": session.mode},
                    session_id=session.session_id
                )
                session_manager.update_session_results(session.session_id, result)
                session_manager.update_progress(session.session_id, 100, status="completed")
                await manager.broadcast(session.session_id, {"event":"status","status":"completed","progress":100})
                await manager.broadcast(session.session_id, {"event":"summary","payload":result})
            except Exception as e:
                session_manager.update_progress(session.session_id, session.progress, status="error")
                await manager.broadcast(session.session_id, {"event":"status","status":"error","message":str(e)})

        asyncio.create_task(run(), name=f"critique-{session.session_id}")
        return CritiqueResponse(
            session_id=session.session_id,
            status="processing",
            message="Paper submitted successfully. Analysis started."
        )

    except Exception as e:
        logger.error(f"Error starting critique: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start critique: {str(e)}")

@app.get("/api/critique/status/{session_id}")
async def get_critique_status(session_id: str):
    """Get the current status of a critique session."""
    session = session_manager.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Unknown session")

    # Convert status enum to string
    status = session.status.value if hasattr(session.status, 'value') else str(session.status)

    response = {
        "session_id": session.session_id,
        "title": session.paper_title,
        "mode": getattr(session, 'mode', 'STANDARD'),
        "status": status,
        "progress": session.overall_progress,
        # Optional: include a small summary or a flag that results exist
        "has_result": session.result is not None,
        # "result": session.result,  # include if your UI wants it here
    }

    return response

@app.get("/api/critique/results/{session_id}", response_model=ResultsResponse)
async def get_critique_results(session_id: str):
    """
    Get the final results of a completed critique session.
    """
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    status = session.status.value if hasattr(session.status, 'value') else str(session.status)
    if status != "completed":
        raise HTTPException(status_code=400, detail="Session not yet completed")

    if not session.overall_score and not session.final_critique:
        raise HTTPException(status_code=500, detail="Results not found for completed session")

    return ResultsResponse(**{
        "session_id": session_id,
        "paper_title": session.paper_title,
        "overall_score": session.overall_score,
        "final_critique": session.final_critique,
        "dimension_critiques": session.dimension_critiques or {},
        "created_at": session.created_at.isoformat() if session.created_at else None,
        "updated_at": session.updated_at.isoformat() if session.updated_at else None
    })

@app.websocket("/api/critique/live/{session_id}")
async def critique_progress_websocket(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time progress updates.
    """
    await manager.connect(session_id, websocket)

    try:
        # Send initial status
        session = session_manager.get_session(session_id)
        if session:
            status = session.status.value if hasattr(session.status, 'value') else str(session.status)
            current_status = {
                "session_id": session_id,
                "state": status,
                "paper_title": session.paper_title,
                "created_at": session.created_at.isoformat() if session.created_at else None,
                "updated_at": session.updated_at.isoformat() if session.updated_at else None,
                "error": session.error_message,
                "progress": session.overall_progress
            }
            if status == "completed":
                current_status["result"] = {
                    "overall_score": session.overall_score,
                    "final_critique": session.final_critique,
                    "dimension_critiques": session.dimension_critiques
                }
            await manager.send_message(session_id, current_status)

        # Keep connection alive and handle any messages
        while True:
            try:
                # Receive messages to keep the connection alive or process client commands
                await websocket.receive_json() # Use receive_json for structured data
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected for session {session_id}")
                break
            except Exception as e:
                logger.error(f"Error receiving message on WebSocket for session {session_id}: {e}")
                break

    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")
    finally:
        manager.disconnect(session_id, None)

@app.get("/api/critique/download/{session_id}")
async def download_critique_report(session_id: str, format: str = Query("pdf", description="Report format: pdf, html, md")):
    """
    Download the critique report in the specified format.
    """
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    status = session.status.value if hasattr(session.status, 'value') else str(session.status)
    if status != "completed": # Use the mapped status 'completed'
        raise HTTPException(status_code=400, detail="Session not yet completed")

    report_filename = getattr(session, 'report_filename', None)
    if not report_filename:
        raise HTTPException(status_code=500, detail="Report filename not found for completed session")

    try:
        if format.lower() == "pdf":
            # For PDF, we assume the report is already generated and its path is known
            # In a more complex system, you might regenerate it here or fetch it from storage
            pdf_path = os.path.join("/tmp", report_filename) # Assuming reports are stored in /tmp
            if not os.path.exists(pdf_path):
                 # Fallback to mock generation if file not found
                 if MOCK_MODE:
                     pdf_filename_mock = generate_pact_pdf_report_fallback(session_id, {})
                     pdf_path = os.path.join("/tmp", pdf_filename_mock)
                 else:
                    raise HTTPException(status_code=500, detail="PDF report file not found.")

            return FileResponse(
                path=pdf_path,
                filename=report_filename,
                media_type="application/pdf"
            )

        elif format.lower() == "html":
            # Generate HTML report
            results_data = session.get("result")
            if not results_data:
                 raise HTTPException(status_code=500, detail="Result data missing for HTML report generation.")
            html_content = generate_html_report(results_data)

            return StreamingResponse(
                io.StringIO(html_content),
                media_type="text/html",
                headers={"Content-Disposition": f"attachment; filename=PACT_Report_{session_id[:8]}.html"}
            )

        elif format.lower() == "md":
            # Generate Markdown report
            results_data = session.get("result")
            if not results_data:
                 raise HTTPException(status_code=500, detail="Result data missing for Markdown report generation.")
            md_content = generate_markdown_report(results_data)

            return StreamingResponse(
                io.StringIO(md_content),
                media_type="text/markdown",
                headers={"Content-Disposition": f"attachment; filename=PACT_Report_{session_id[:8]}.md"}
            )

        else:
            raise HTTPException(status_code=400, detail="Unsupported format. Use: pdf, html, or md")

    except Exception as e:
        logger.error(f"Error generating report for session {session_id} in format {format}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")

@app.get("/api/critique/preview/{session_id}")
async def preview_critique_report(session_id: str):
    """
    Get a preview of the critique report (HTML format).
    """
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    status = session.status.value if hasattr(session.status, 'value') else str(session.status)
    if status != "completed": # Use the mapped status 'completed'
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
    try:
        logger.info("Launching supervisor task for session %s", session_id)

        # Get session object
        session = session_manager.get_session(session_id)
        if not session:
            raise Exception(f"Session {session_id} not found")

        # Create supervisor
        supervisor = make_supervisor()

        # Set initial status and broadcast
        session_manager.update_session_status(session_id, "running", overall_progress=1)
        await manager.broadcast(session_id, {
            "event": "status",
            "status": "running",
            "progress": 1
        })

        # Run the analysis
        result = await supervisor.ainvoke(
            {"paper_content": paper_content, "paper_title": paper_title, "mode": mode},
            session_id=session_id
        )

        # ✅ persist results so /status (or later GET) can return them
        session_manager.update_session_results(session_id, result)

        # ✅ set final status/progress and broadcast
        session_manager.update_progress(session_id, 100, status="completed")
        await manager.broadcast(session_id, {"event": "status", "status": "completed", "progress": 100})
        
        # If the supervisor didn't already send a summary, send it here too:
        await manager.broadcast(session_id, {"event": "summary", "payload": result})

        logger.info("Critique completed for session %s", session_id)

    except Exception as e:
        logger.exception("Critique failed for session %s", session_id)
        session_manager.update_session_status(session_id, "error", error_message=str(e))
        await manager.broadcast(session_id, {
            "event": "status",
            "status": "error",
            "message": str(e)
        })


# Progress callback class (can be used if LangGraph is integrated more deeply)
class CritiqueProgressCallback:
    """Callback to track LangGraph execution progress."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.current_node = ""

    async def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs):
        """Called when a chain starts."""
        logger.debug(f"Chain start: {serialized.get('name')}")
        pass

    async def on_chain_end(self, outputs: Dict[str, Any], **kwargs):
        """Called when a chain ends."""
        logger.debug("Chain end.")
        pass

    async def on_llm_start(self, serialized: Dict[str, Any], prompts: list, **kwargs):
        """Called when LLM starts."""
        logger.debug(f"LLM start: {serialized.get('name')}")
        pass

    async def on_llm_end(self, response, **kwargs):
        """Called when LLM ends."""
        logger.debug("LLM end.")
        pass

    async def on_tool_start(self, serialized: Dict[str, Any], args: Dict[str, Any], **kwargs):
        """Called when a tool starts."""
        logger.debug(f"Tool start: {serialized.get('name')}")
        pass

    async def on_tool_end(self, output, **kwargs):
        """Called when a tool ends."""
        logger.debug("Tool end.")
        pass

    async def on_agent_action(self, action, **kwargs):
        """Called when an agent takes an action."""
        logger.debug(f"Agent action: {action}")
        agent_id = action.tool # Assuming tool name is agent ID
        message = action.tool_input if isinstance(action.tool_input, str) else str(action.tool_input)
        if agent_id:
            session_manager.update_agent_status(self.session_id, agent_id, "ACTIVE", message)
            await update_session_status(self.session_id, "processing", 50, f"Agent {agent_id} is running...") # Example progress update

    async def on_agent_finish(self, finish, **kwargs):
        """Called when an agent finishes."""
        logger.debug(f"Agent finish: {finish}")
        # Potentially update session status or agent status based on finish output
        pass

async def notify_websocket_clients(session_id: str):
    """Send progress updates to WebSocket clients."""
    session = session_manager.get_session(session_id)
    if session:
        status = session.status.value if hasattr(session.status, 'value') else str(session.status)
        progress_data = {
            "session_id": session_id,
            "state": status,
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
    # Note: In a real application, you would not run this directly if deploying with a production server like Gunicorn.
    # This is for local development and testing.
    logger.info("Starting FastAPI server for PACT Critique System (Development Mode)...")
    uvicorn.run(
        "api_server:app", # Assumes this file is named api_server.py
        host="0.0.0.0",
        port=8000,
        reload=True, # Enable auto-reloading during development
        log_level="info"
    )