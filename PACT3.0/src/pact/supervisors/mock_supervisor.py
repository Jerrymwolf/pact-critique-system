
import asyncio
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class MockCritiqueSupervisor:
    """Mock supervisor that emits progress events for testing."""
    
    async def ainvoke(self, state: Dict[str, Any], session_id: Optional[str] = None) -> Dict[str, Any]:
        logger.info("Mock supervisor ainvoke called with state: %s", state)
        
        # Import websocket_manager here to avoid circular imports
        try:
            from ..websocket_manager import manager as websocket_manager
            logger.info("WS manager id (mock_supervisor)=%s", id(websocket_manager))
        except ImportError:
            websocket_manager = None
            logger.warning("WebSocket manager not available")
        
        # Simulate streaming progress with websocket updates
        progress_steps = [
            (5, "Parsing document"),
            (25, "Extracting sections"), 
            (55, "Scoring rubric"),
            (85, "Drafting feedback"),
            (100, "Finalizing")
        ]
        
        for pct, msg in progress_steps:
            await asyncio.sleep(0.2)
            if session_id and websocket_manager:
                await websocket_manager.broadcast(session_id, {
                    "event": "progress", 
                    "progress": pct, 
                    "message": msg
                })
        
        # Create mock result
        result = {
            "paper_title": state.get("paper_title", "Sample Paper"),
            "overall_score": 75.5,
            "final_critique": "Mock analysis result with detailed feedback covering all PACT dimensions. The paper demonstrates adequate understanding but requires refinement in several areas.",
            "dimension_critiques": {
                "1.0.0": {
                    "dimension_score": 78, 
                    "dimension_name": "Research Foundations",
                    "strengths": ["Clear research question", "Relevant literature base"], 
                    "weaknesses": ["Limited scope", "Missing key citations"]
                },
                "2.0.0": {
                    "dimension_score": 73, 
                    "dimension_name": "Methodological Rigor",
                    "strengths": ["Good methodology description", "Appropriate data collection"], 
                    "weaknesses": ["Small sample size", "Limited statistical analysis"]
                },
                "3.0.0": {
                    "dimension_score": 80, 
                    "dimension_name": "Structure & Coherence",
                    "strengths": ["Logical flow", "Clear section organization"], 
                    "weaknesses": ["Transitions could be smoother"]
                }
            }
        }
        
        if session_id and websocket_manager:
            await websocket_manager.broadcast(session_id, {
                "event": "summary", 
                "payload": result
            })
            
        return result
