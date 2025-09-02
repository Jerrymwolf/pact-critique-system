"""
FastAPI Backend Server for PACT Critique System

Provides REST API endpoints and WebSocket support for real-time progress tracking.
"""

import asyncio
import os
from typing import Dict, Any, Optional
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
import uvicorn
import io

from .session_manager import session_manager, CritiqueStatus, AgentStatus
from .pact_critique_agent import pact_critique_agent
from .websocket_handler import WebSocketManager
from .pdf_generator import generate_pact_pdf_report
from .enhanced_schemas import ComprehensiveCritique
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

# Request/Response Models
class PaperSubmission(BaseModel):
    content: str = Field(..., description="The paper content to critique")
    title: Optional[str] = Field(None, description="Paper title")
    paper_type: Optional[str] = Field(None, description="Type of paper (research, thesis, etc.)")

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
websocket_manager = WebSocketManager()

# Cleanup task
async def cleanup_old_sessions():
    """Background task to cleanup old sessions."""
    while True:
        try:
            removed = session_manager.cleanup_old_sessions(max_age_hours=24)
            if removed > 0:
                print(f"Cleaned up {removed} old sessions")
        except Exception as e:
            print(f"Error during cleanup: {e}")
        await asyncio.sleep(3600)  # Run every hour

# App lifespan manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    cleanup_task = asyncio.create_task(cleanup_old_sessions())
    yield
    # Shutdown
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
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/")
async def serve_app():
    """Serve the main PACT critique application."""
    return FileResponse("pact_critique_app.html")

# ===== API ENDPOINTS =====

@app.post("/api/critique/start", response_model=CritiqueResponse)
async def start_critique(paper: PaperSubmission, background_tasks: BackgroundTasks):
    """
    Submit a paper for PACT critique analysis.
    """
    try:
        # Create session
        session_id = session_manager.create_session(
            paper_content=paper.content,
            paper_title=paper.title,
            paper_type=paper.paper_type
        )
        
        # Start critique process in background
        background_tasks.add_task(run_critique_workflow, session_id)
        
        return CritiqueResponse(
            session_id=session_id,
            status="processing",
            message="Paper submitted successfully. Analysis started."
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start critique: {str(e)}")

@app.get("/api/critique/status/{session_id}", response_model=ProgressResponse)
async def get_critique_status(session_id: str):
    """
    Get the current status and progress of a critique session.
    """
    progress_data = session_manager.get_session_progress(session_id)
    
    if "error" in progress_data:
        raise HTTPException(status_code=404, detail=progress_data["error"])
    
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
        else:
            raise HTTPException(status_code=400, detail=results_data["error"])
    
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
        await websocket_manager.send_to_session(session_id, progress_data)
        
        # Keep connection alive and handle any messages
        while True:
            try:
                data = await websocket.receive_text()
                # Handle any client messages if needed
            except WebSocketDisconnect:
                break
            
    except Exception as e:
        print(f"WebSocket error for session {session_id}: {e}")
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
        else:
            raise HTTPException(status_code=400, detail=results_data["error"])
    
    try:
        if format.lower() == "pdf":
            # Generate PDF report
            pdf_path = await generate_pdf_report(session_id, results_data)
            
            # Return the PDF file
            return FileResponse(
                path=pdf_path,
                filename=f"PACT_Analysis_Report_{session_id[:8]}.pdf",
                media_type="application/pdf"
            )
        
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
        print(f"Error generating report for session {session_id}: {e}")
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
        else:
            raise HTTPException(status_code=400, detail=results_data["error"])
    
    try:
        html_content = generate_html_report(results_data)
        return {"preview_html": html_content}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate preview: {str(e)}")

# ===== BACKGROUND TASK FUNCTIONS =====

async def run_critique_workflow(session_id: str):
    """
    Run the complete PACT critique workflow for a session.
    """
    try:
        session = session_manager.get_session(session_id)
        if not session:
            return
        
        # Update status to processing
        session_manager.update_session_status(
            session_id, CritiqueStatus.PROCESSING, 
            "Initializing critique workflow", 10
        )
        await notify_websocket_clients(session_id)
        
        # Create messages for LangGraph
        messages = [HumanMessage(content=session.paper_content)]
        
        # Configure for the session
        config = {
            "configurable": {"thread_id": session_id},
            "recursion_limit": 20
        }
        
        # Update status to planning
        session_manager.update_session_status(
            session_id, CritiqueStatus.PLANNING, 
            "Planning critique approach", 20
        )
        session_manager.update_agent_status(
            session_id, "supervisor", AgentStatus.ACTIVE, 
            "Planning critique approach"
        )
        await notify_websocket_clients(session_id)
        
        # Run the critique
        result = await pact_critique_agent.ainvoke(
            {"messages": messages},
            config=config
        )
        
        # Update status to evaluating
        session_manager.update_session_status(
            session_id, CritiqueStatus.EVALUATING, 
            "Evaluating dimensions", 60
        )
        await notify_websocket_clients(session_id)
        
        # Update status to synthesizing
        session_manager.update_session_status(
            session_id, CritiqueStatus.SYNTHESIZING, 
            "Synthesizing final critique", 90
        )
        await notify_websocket_clients(session_id)
        
        # Extract results
        final_critique = result.get('final_critique', '')
        overall_score = result.get('overall_score', 0)
        dimension_critiques = result.get('dimension_critiques', {})
        
        # Update session with results
        session_manager.set_critique_results(
            session_id=session_id,
            dimension_critiques=dimension_critiques,
            final_critique=final_critique,
            overall_score=overall_score
        )
        
        # Mark as completed
        session_manager.update_session_status(
            session_id, CritiqueStatus.COMPLETED, 
            "Critique completed successfully", 100
        )
        
        # Mark all agents as completed
        for agent_id in session.agents.keys():
            session_manager.update_agent_status(
                session_id, agent_id, AgentStatus.COMPLETED, 
                "Analysis complete", 100
            )
        
        await notify_websocket_clients(session_id)
        
    except Exception as e:
        print(f"Error in critique workflow for session {session_id}: {e}")
        session_manager.set_error(session_id, str(e))
        await notify_websocket_clients(session_id)

class CritiqueProgressCallback:
    """Callback to track LangGraph execution progress."""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.current_node = ""
    
    async def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs):
        """Called when a chain starts."""
        pass
    
    async def on_chain_end(self, outputs: Dict[str, Any], **kwargs):
        """Called when a chain ends."""
        pass
    
    async def on_llm_start(self, serialized: Dict[str, Any], prompts: list, **kwargs):
        """Called when LLM starts."""
        pass
    
    async def on_llm_end(self, response, **kwargs):
        """Called when LLM ends."""
        pass

async def notify_websocket_clients(session_id: str):
    """Send progress updates to WebSocket clients."""
    progress_data = session_manager.get_session_progress(session_id)
    await websocket_manager.send_to_session(session_id, progress_data)

# ===== REPORT GENERATION FUNCTIONS =====

async def generate_pdf_report(session_id: str, results_data: Dict[str, Any]) -> str:
    """Generate a PDF report from results data."""
    try:
        # Convert results to comprehensive critique format
        comprehensive_critique = convert_to_comprehensive_critique(results_data)
        
        # Generate PDF
        pdf_path = generate_pact_pdf_report(comprehensive_critique, session_id)
        return pdf_path
    
    except Exception as e:
        print(f"Error generating PDF report: {e}")
        raise

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
        md += f"""
### {critique.get('dimension_name', dim_id)}

**Score:** {critique.get('dimension_score', 'N/A')}/100

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
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )