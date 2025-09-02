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

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
import uvicorn
import io
from docx import Document
import PyPDF2
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Assume these are defined in a separate pact module
# from .session_manager import session_manager, CritiqueStatus, AgentStatus
# from .pact_critique_agent import pact_critique_agent
# from .websocket_handler import WebSocketManager
# from .pdf_generator import generate_pact_pdf_report
# from .enhanced_schemas import ComprehensiveCritique
# from .mode_config import AgentMode, mode_config # Added for mode support

# Mock implementations for demonstration purposes if pact module is not available
class MockSessionManager:
    def __init__(self):
        self.sessions = {}
        self.critique_status = {
            "processing": "PROCESSING",
            "running": "RUNNING",
            "completed": "COMPLETED",
            "failed": "FAILED",
            "planning": "PLANNING",
            "evaluating": "EVALUATING",
            "synthesizing": "SYNTHESIZING"
        }
        self.agent_status = {
            "active": "ACTIVE",
            "completed": "COMPLETED"
        }

    def create_session(self, paper_content: str, paper_title: str = None, paper_type: str = None, mode: str = "STANDARD"):
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "id": session_id,
            "status": self.critique_status["processing"],
            "created_at": datetime.now().isoformat(),
            "paper_content": paper_content,
            "paper_title": paper_title,
            "paper_type": paper_type,
            "mode": mode,
            "progress": 0,
            "agents": {},
            "error_message": None
        }
        return session_id

    def get_session(self, session_id: str):
        return self.sessions.get(session_id)

    def update_session_status(self, session_id: str, status: str, progress: int, message: str = ""):
        if session_id in self.sessions:
            self.sessions[session_id]["status"] = self.critique_status.get(status, status)
            self.sessions[session_id]["progress"] = progress
            if message:
                self.sessions[session_id]["current_stage"] = message
            self.sessions[session_id]["updated_at"] = datetime.now().isoformat()

    def update_agent_status(self, session_id: str, agent_id: str, status: str, message: str, progress: int = 100):
        if session_id in self.sessions:
            self.sessions[session_id]["agents"][agent_id] = {
                "status": self.agent_status.get(status, status),
                "message": message,
                "progress": progress
            }

    def get_session_progress(self, session_id: str):
        session = self.sessions.get(session_id)
        if not session:
            return {"error": "Session not found"}
        return {
            "session_id": session_id,
            "status": session.get("status"),
            "overall_progress": session.get("progress", 0),
            "current_stage": session.get("current_stage", "Initializing"),
            "agents": session.get("agents", {}),
            "created_at": session.get("created_at"),
            "updated_at": session.get("updated_at"),
            "error_message": session.get("error_message")
        }

    def get_session_results(self, session_id: str):
        session = self.sessions.get(session_id)
        if not session:
            return {"error": "Session not found"}
        if session.get("status") != self.critique_status["completed"]:
            return {"error": "Session not yet completed"}
        
        # Mock results
        return {
            "session_id": session_id,
            "paper_title": session.get("paper_title"),
            "overall_score": 85.5,
            "final_critique": "This is a mock final critique based on the paper content.",
            "dimension_critiques": {
                "clarity": {"dimension_name": "Clarity", "dimension_score": 90, "strengths": ["Well-organized."], "weaknesses": ["Some jargon could be simplified."]},
                "coherence": {"dimension_name": "Coherence", "dimension_score": 80, "strengths": ["Logical flow is good."], "weaknesses": ["Transitions between sections could be smoother."]}
            },
            "created_at": session.get("created_at"),
            "updated_at": session.get("updated_at")
        }

    def set_critique_results(self, session_id: str, dimension_critiques: Dict[str, Any], final_critique: str, overall_score: float):
        if session_id in self.sessions:
            self.sessions[session_id]["dimension_critiques"] = dimension_critiques
            self.sessions[session_id]["final_critique"] = final_critique
            self.sessions[session_id]["overall_score"] = overall_score
            self.sessions[session_id]["updated_at"] = datetime.now().isoformat()

    def set_error(self, session_id: str, error_message: str):
        if session_id in self.sessions:
            self.sessions[session_id]["error_message"] = error_message
            self.sessions[session_id]["status"] = self.critique_status["failed"]
            self.sessions[session_id]["updated_at"] = datetime.now().isoformat()

    def cleanup_old_sessions(self, max_age_hours: int):
        removed_count = 0
        current_time = datetime.now()
        sessions_to_remove = []
        for session_id, session_data in self.sessions.items():
            created_at = datetime.fromisoformat(session_data.get("created_at"))
            if (current_time - created_at).total_seconds() > max_age_hours * 3600:
                sessions_to_remove.append(session_id)
        
        for session_id in sessions_to_remove:
            del self.sessions[session_id]
            removed_count += 1
        return removed_count

session_manager = MockSessionManager()

class MockWebSocketManager:
    def __init__(self):
        self.active_connections: Dict[str, list[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        if session_id not in self.active_connections:
            self.active_connections[session_id] = []
        self.active_connections[session_id].append(websocket)
        logger.info(f"WebSocket connected for session {session_id}")

    async def disconnect(self, websocket: WebSocket, session_id: str):
        if session_id in self.active_connections and websocket in self.active_connections[session_id]:
            self.active_connections[session_id].remove(websocket)
            if not self.active_connections[session_id]:
                del self.active_connections[session_id]
        logger.info(f"WebSocket disconnected for session {session_id}")

    async def send_to_session(self, session_id: str, message: Any):
        if session_id in self.active_connections:
            disconnected_websockets = []
            for connection in self.active_connections[session_id]:
                try:
                    await connection.send_json(message)
                except RuntimeError as e:
                    logger.error(f"Error sending to WebSocket for session {session_id}: {e}")
                    disconnected_websockets.append(connection)
            
            # Remove disconnected websockets
            for ws in disconnected_websockets:
                if ws in self.active_connections[session_id]:
                    self.active_connections[session_id].remove(ws)
            if not self.active_connections[session_id]:
                del self.active_connections[session_id]
        else:
            logger.warning(f"No active WebSocket connections for session {session_id} to send message.")

websocket_manager = MockWebSocketManager()

# Mocking pact_critique_agent and its components
class MockPactCritiqueAgent:
    async def ainvoke(self, input_data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        await asyncio.sleep(2) # Simulate work
        session_id = config.get("configurable", {}).get("thread_id", "mock_session")
        mode = input_data.get("mode", "STANDARD")

        if mode == "COMPREHENSIVE":
            return {
                "dimension_critiques": {
                    "clarity": {"dimension_name": "Clarity", "dimension_score": 92, "strengths": ["Excellent focus.", "Clear research question."], "weaknesses": ["Minor grammatical errors noted."]},
                    "coherence": {"dimension_name": "Coherence", "dimension_score": 88, "strengths": ["Logical progression of ideas.", "Well-structured arguments."], "weaknesses": ["Some paragraph transitions could be smoother."]},
                    "methodology": {"dimension_name": "Methodology", "dimension_score": 95, "strengths": ["Appropriate methods chosen.", "Detailed explanation of procedures."], "weaknesses": []},
                    "contribution": {"dimension_name": "Contribution", "dimension_score": 85, "strengths": ["Addresses a relevant gap.", "Novel approach."], "weaknesses": ["Impact on the field could be elaborated."]},
                    "writing_style": {"dimension_name": "Writing Style", "dimension_score": 90, "strengths": ["Professional tone.", "Concise language."], "weaknesses": ["Occasional use of passive voice."]}
                },
                "final_critique": "The paper demonstrates strong clarity, coherence, and methodology. The contribution is significant, and the writing style is professional. Minor revisions are recommended for improved transitions and elaboration on the impact.",
                "overall_score": 90.0,
                "agents_status": {
                    "supervisor": "COMPLETED",
                    "dimension_analyzer_clarity": "COMPLETED",
                    "dimension_analyzer_coherence": "COMPLETED",
                    "dimension_analyzer_methodology": "COMPLETED",
                    "dimension_analyzer_contribution": "COMPLETED",
                    "dimension_analyzer_writing_style": "COMPLETED",
                    "synthesizer": "COMPLETED"
                }
            }
        elif mode == "APA7":
             return {
                "dimension_critiques": {
                    "apa_formatting": {"dimension_name": "APA Formatting", "dimension_score": 95, "strengths": ["Correct citation style.", "Proper heading structure."], "weaknesses": ["Minor inconsistencies in reference list formatting."]},
                    "clarity": {"dimension_name": "Clarity", "dimension_score": 85, "strengths": ["Clear communication of ideas."], "weaknesses": ["Some sentences are a bit long."]}
                },
                "final_critique": "The paper generally adheres to APA 7th edition guidelines, with excellent citation and heading structure. Minor improvements are needed in the reference list. Clarity of communication is good, though sentence length could be optimized.",
                "overall_score": 88.0,
                "agents_status": {
                    "supervisor": "COMPLETED",
                    "dimension_analyzer_apa_formatting": "COMPLETED",
                    "dimension_analyzer_clarity": "COMPLETED",
                    "synthesizer": "COMPLETED"
                }
            }
        else: # STANDARD mode
            return {
                "dimension_critiques": {
                    "clarity": {"dimension_name": "Clarity", "dimension_score": 85, "strengths": ["Easy to understand."], "weaknesses": ["Could use more examples."]},
                    "coherence": {"dimension_name": "Coherence", "dimension_score": 80, "strengths": ["Logical flow."], "weaknesses": ["Some connections between paragraphs are weak."]}
                },
                "final_critique": "The paper is clear and coherent, though some areas could benefit from further development and smoother transitions.",
                "overall_score": 82.5,
                "agents_status": {
                    "supervisor": "COMPLETED",
                    "dimension_analyzer_clarity": "COMPLETED",
                    "dimension_analyzer_coherence": "COMPLETED",
                    "synthesizer": "COMPLETED"
                }
            }

pact_critique_agent = MockPactCritiqueAgent()

async def generate_pdf_report(comprehensive_critique: Dict[str, Any], session_id: str) -> Optional[str]:
    """Mock PDF generation function."""
    logger.info(f"Generating PDF for session {session_id}...")
    await asyncio.sleep(1)
    pdf_filename = f"PACT_Analysis_Report_{session_id[:8]}.pdf"
    # In a real scenario, this would create a PDF file.
    # For now, just return a mock path.
    logger.info(f"Mock PDF report generated: {pdf_filename}")
    return pdf_filename

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

# Dummy mode_config, actual implementation would define agent configurations per mode
mode_config = {
    AgentMode.APA7: {"agents": ["APAFormatterAgent", "ClarityCheckerAgent"]},
    AgentMode.STANDARD: {"agents": ["ClarityCheckerAgent", "CoherenceCheckerAgent"]},
    AgentMode.COMPREHENSIVE: {"agents": ["ClarityCheckerAgent", "CoherenceCheckerAgent", "MethodologyCheckerAgent", "ContributionCheckerAgent", "WritingStyleCheckerAgent"]},
}

# Mock create_critique_supervisor
def create_critique_supervisor():
    """Mock function to create a supervisor agent."""
    logger.info("Creating mock critique supervisor...")
    # In a real scenario, this would instantiate and configure the LangGraph supervisor
    # For this mock, we return an object with an ainvoke method
    class MockSupervisor:
        async def ainvoke(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
            logger.info("Mock supervisor ainvoke called with state: %s", initial_state)
            # Simulate agent execution based on mode
            mode = initial_state.get("mode", "STANDARD")
            paper_content = initial_state.get("paper_content")
            
            # Mock the behavior of pact_critique_agent.ainvoke directly
            # This mock supervisor doesn't have complex internal logic, 
            # it just delegates to the mock pact_critique_agent
            return await pact_critique_agent.ainvoke({"messages": [HumanMessage(content=paper_content)], "mode": mode})

    return MockSupervisor()

# Mock update_session_status
async def update_session_status(session_id: str, status: str, progress: int, message: str = ""):
    """Mock function to update session status and notify clients."""
    session_manager.update_session_status(session_id, status, progress, message)
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
            <p>{results_data.get('final_critique', '')[:500]}...</p>
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
    try:
        content = await file.read()

        if file.filename.lower().endswith('.docx'):
            # Extract text from DOCX
            doc = Document(io.BytesIO(content))
            text_content = []
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_content.append(paragraph.text)
            extracted_text = '\n'.join(text_content)

        elif file.filename.lower().endswith('.pdf'):
            # Extract text from PDF
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
            text_content = []
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_content.append(page_text)
            extracted_text = '\n'.join(text_content)

        elif file.filename.lower().endswith('.txt'):
            # Handle plain text
            extracted_text = content.decode('utf-8')

        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload TXT, DOCX, or PDF files.")

        if not extracted_text.strip():
            raise HTTPException(status_code=400, detail="No text content could be extracted from the file.")

        return {
            "content": extracted_text,
            "filename": file.filename,
            "message": "File processed successfully"
        }

    except Exception as e:
        logger.error(f"Failed to process file {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")

@app.post("/api/critique/start", response_model=CritiqueResponse)
async def start_critique(request: CritiqueRequest):
    """
    Submit a paper for PACT critique analysis.
    """
    try:
        # Get mode from request (default to STANDARD)
        mode = getattr(request, 'mode', 'STANDARD')
        
        # Validate mode if it's not recognized, default to STANDARD
        try:
            AgentMode(mode)
        except ValueError:
            logger.warning(f"Unrecognized mode '{mode}'. Defaulting to STANDARD.")
            mode = AgentMode.STANDARD

        # Create session using session_manager
        session_id = session_manager.create_session(
            paper_content=request.content,
            paper_title=request.title,
            paper_type=request.paper_type,
            mode=mode
        )

        # Start critique process in background
        # Pass the mode to the workflow
        asyncio.create_task(run_critique_workflow(session_id, request.content, request.title, mode))

        return CritiqueResponse(
            session_id=session_id,
            status="processing",
            message="Paper submitted successfully. Analysis started."
        )

    except Exception as e:
        logger.error(f"Error starting critique: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start critique: {str(e)}")

@app.get("/api/critique/status/{session_id}", response_model=ProgressResponse)
async def get_critique_status(session_id: str):
    """
    Get the current status and progress of a critique session.
    """
    progress_data = session_manager.get_session_progress(session_id)

    if "error" in progress_data:
        status_code = 404 if progress_data["error"] == "Session not found" else 400
        raise HTTPException(status_code=status_code, detail=progress_data["error"])

    return ProgressResponse(**progress_data)

@app.get("/api/critique/results/{session_id}", response_model=ResultsResponse)
async def get_critique_results(session_id: str):
    """
    Get the final results of a completed critique session.
    """
    results_data = session_manager.get_session_results(session_id)

    if "error" in results_data:
        if results_data["error"] == "Session not found":
            raise HTTPException(status_code=404, detail=results_data["error"])
        elif "not yet completed" in results_data["error"]:
             raise HTTPException(status_code=400, detail=results_data["error"])
        else:
            raise HTTPException(status_code=500, detail=results_data["error"])

    return ResultsResponse(**results_data)

@app.websocket("/api/critique/live/{session_id}")
async def critique_progress_websocket(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time progress updates.
    """
    await websocket_manager.connect(websocket, session_id)

    try:
        # Send initial status
        progress_data = session_manager.get_session_progress(session_id)
        if "error" not in progress_data:
            await websocket_manager.send_to_session(session_id, progress_data)

        # Keep connection alive and handle any messages
        while True:
            try:
                # Receive messages to keep the connection alive or process client commands
                data = await websocket.receive_json() # Use receive_json for structured data
                logger.debug(f"Received message from WebSocket for session {session_id}: {data}")
                # Add logic here to handle messages from the client if needed
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected for session {session_id}")
                break
            except Exception as e:
                logger.error(f"Error receiving message on WebSocket for session {session_id}: {e}")
                break

    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")
    finally:
        await websocket_manager.disconnect(websocket, session_id)

@app.get("/api/critique/download/{session_id}")
async def download_critique_report(session_id: str, format: str = Query("pdf", description="Report format: pdf, html, md")):
    """
    Download the critique report in the specified format.
    """
    # Get session results
    results_data = session_manager.get_session_results(session_id)

    if "error" in results_data:
        if results_data["error"] == "Session not found":
            raise HTTPException(status_code=404, detail=results_data["error"])
        elif "not yet completed" in results_data["error"]:
             raise HTTPException(status_code=400, detail=results_data["error"])
        else:
            raise HTTPException(status_code=500, detail=results_data["error"])

    try:
        if format.lower() == "pdf":
            # Generate PDF report
            # Ensure results_data is converted to the format expected by generate_pact_pdf_report
            # For now, using the mock comprehensive critique conversion
            comprehensive_critique = convert_to_comprehensive_critique(results_data)
            pdf_path = await generate_pdf_report(comprehensive_critique, session_id)

            if pdf_path:
                # Return the PDF file
                return FileResponse(
                    path=pdf_path,
                    filename=f"PACT_Analysis_Report_{session_id[:8]}.pdf",
                    media_type="application/pdf"
                )
            else:
                raise HTTPException(status_code=500, detail="Failed to generate PDF report.")

        elif format.lower() == "html":
            # Generate HTML report
            html_content = generate_html_report(results_data)

            return StreamingResponse(
                io.StringIO(html_content),
                media_type="text/html",
                headers={"Content-Disposition": f"attachment; filename=PACT_Report_{session_id[:8]}.html"}
            )

        elif format.lower() == "md":
            # Generate Markdown report  
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
    results_data = session_manager.get_session_results(session_id)

    if "error" in results_data:
        if results_data["error"] == "Session not found":
            raise HTTPException(status_code=404, detail=results_data["error"])
        elif "not yet completed" in results_data["error"]:
             raise HTTPException(status_code=400, detail=results_data["error"])
        else:
            raise HTTPException(status_code=500, detail=results_data["error"])

    try:
        html_content = generate_html_report(results_data)
        return {"preview_html": html_content}

    except Exception as e:
        logger.error(f"Failed to generate preview for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate preview: {str(e)}")

# ===== BACKGROUND TASK FUNCTIONS =====

async def run_critique_workflow(session_id: str, paper_content: str, paper_title: str = None, mode: str = "STANDARD"):
    """Run the critique workflow asynchronously."""
    try:
        # Import mode config and AgentMode if not already done globally
        # from pact.mode_config import AgentMode, mode_config

        # Validate mode
        try:
            agent_mode = AgentMode(mode)
        except ValueError:
            logger.warning(f"Invalid mode '{mode}' received in workflow. Defaulting to STANDARD.")
            agent_mode = AgentMode.STANDARD
            mode = AgentMode.STANDARD # Ensure mode variable is updated

        # Update session status to running and initialize progress
        await update_session_status(session_id, "running", 10, "Initializing critique workflow")

        # Truncate paper content to prevent token limit issues based on the selected mode
        # These limits are examples and might need tuning based on the LLM's actual token window
        max_content_length = 0
        if agent_mode == AgentMode.COMPREHENSIVE:
            max_content_length = 8000  # Generous limit for comprehensive mode
        elif agent_mode == AgentMode.APA7:
            max_content_length = 7000  # Moderate limit for APA7 mode
        else: # STANDARD mode
            max_content_length = 6000  # Standard limit

        truncated_content = paper_content[:max_content_length]
        if len(paper_content) > max_content_length:
            logger.warning(f"Paper content truncated for session {session_id} due to token limits. Original length: {len(paper_content)}, Truncated length: {max_content_length}")

        # Create the supervisor agent
        supervisor = create_critique_supervisor()

        # Initial state for the LangGraph workflow
        initial_state = {
            "paper_content": truncated_content,
            "paper_title": paper_title,
            "mode": mode, # Include mode in the state
            "dimension_critiques": {},
            "messages": [HumanMessage(content=truncated_content)], # Use truncated content
            "agents": {} # Placeholder for agent statuses if needed internally
        }

        # Update progress to indicate planning phase
        await update_session_status(session_id, "planning", 25, "Planning critique approach")
        # Mock updating agent status for supervisor
        session_manager.update_agent_status(session_id, "supervisor", "ACTIVE", "Planning critique approach", 25)


        # Run the critique workflow via the supervisor agent
        # The ainvoke method will orchestrate the agents based on the initial_state
        result = await supervisor.ainvoke(initial_state)

        # Update progress to indicate evaluation/synthesis phase
        await update_session_status(session_id, "evaluating", 75, "Evaluating dimensions and synthesizing results")
        
        # Update agent statuses based on mock result
        for agent_id, status in result.get("agents_status", {}).items():
             session_manager.update_agent_status(session_id, agent_id, status, f"Agent {status.lower()}", 100)

        # Process the results from the agent execution
        if result.get("dimension_critiques"):
            # Create comprehensive critique using enhanced schemas (or relevant schema based on mode)
            # For simplicity, we use convert_to_comprehensive_critique for all modes here.
            # A more robust implementation would tailor this based on the 'mode'.
            comprehensive_critique = create_comprehensive_critique(result)

            # Generate markdown report
            markdown_report = generate_markdown_report(comprehensive_critique)

            # Generate PDF report
            pdf_path = await generate_pdf_report(comprehensive_critique, session_id)

            # Save final results to the session manager
            final_results = {
                "session_id": session_id,
                "status": "completed",
                "mode": mode,
                "comprehensive_critique": comprehensive_critique,
                "markdown_report": markdown_report,
                "pdf_path": str(pdf_path) if pdf_path else None,
                "completed_at": datetime.now().isoformat()
            }

            # Update session manager with final results
            # This part needs to map correctly to how session_manager stores results
            # Assuming set_critique_results and then updating status to completed
            session_manager.set_critique_results(
                session_id=session_id,
                dimension_critiques=comprehensive_critique.get("dimension_analyses", {}),
                final_critique=comprehensive_critique.get("executive_summary", ""),
                overall_score=comprehensive_critique.get("overall_score", 0)
            )
            
            # Mark all agents as completed if they were tracked
            for agent_id in result.get("agents_status", {}):
                 session_manager.update_agent_status(session_id, agent_id, "COMPLETED", "Analysis complete", 100)

            # Mark session as completed
            await update_session_status(session_id, "completed", 100, "Critique completed successfully")

        else:
            # Handle cases where no critiques were generated
            error_message = "No critiques generated by the agent."
            logger.error(f"Critique workflow failed for session {session_id}: {error_message}")
            session_manager.set_error(session_id, error_message)
            await update_session_status(session_id, "failed", 0, error_message)

    except Exception as e:
        # Catch any unexpected errors during the workflow execution
        logger.error(f"Unexpected error in critique workflow for session {session_id}: {e}", exc_info=True)
        session_manager.set_error(session_id, f"Workflow execution failed: {str(e)}")
        await update_session_status(session_id, "failed", 0, f"Workflow execution failed: {str(e)}")


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
    progress_data = session_manager.get_session_progress(session_id)
    if progress_data and "error" not in progress_data:
        await websocket_manager.send_to_session(session_id, progress_data)

# ===== HEALTH CHECK =====

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