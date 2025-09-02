
import asyncio
import json
import logging
import re
from typing import Optional, Dict, Any
from openai import OpenAI
from openai import APIError, APIConnectionError, RateLimitError, BadRequestError

logger = logging.getLogger(__name__)

# Use environment variable or default to gpt-5
import os
MODEL = os.getenv("OPENAI_MODEL", "gpt-5")

client = OpenAI(timeout=60)  # hard timeout for network stall

def _safe_extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Try to pull the first JSON object out of an LLM reply."""
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

class RealCritiqueSupervisor:
    """Real supervisor using OpenAI API with robust error handling and non-blocking calls."""
    
    async def ainvoke(self, state: Dict[str, Any], session_id: Optional[str] = None) -> Dict[str, Any]:
        # Import websocket_manager here to avoid circular imports
        try:
            from ..websocket_handler import WebSocketManager
            # Create a temporary manager instance or use the global one
            websocket_manager = WebSocketManager()
        except ImportError:
            websocket_manager = None
            logger.warning("WebSocket manager not available")

        # 1) Early progress ping
        if session_id and websocket_manager:
            await websocket_manager.send_message(session_id, {
                "event": "progress", 
                "progress": 5, 
                "message": "Calling GPT-5"
            })

        paper_title = state.get("paper_title") or "Untitled Paper"
        paper_content = state.get("paper_content") or ""
        mode = state.get("mode", "STANDARD")

        prompt = f"""
You are PACT, an academic critique engine that analyzes papers using the PACT taxonomy.
Critique the paper titled "{paper_title}" below in {mode} mode.

Return **JSON only** with this exact structure:
{{
    "paper_title": "{paper_title}",
    "overall_score": <number between 0-100>,
    "final_critique": "<comprehensive critique text>",
    "dimension_critiques": {{
        "1.0.0": {{
            "dimension_name": "Research Foundations",
            "dimension_score": <number between 0-100>,
            "strengths": ["<strength 1>", "<strength 2>"],
            "weaknesses": ["<weakness 1>", "<weakness 2>"]
        }},
        "2.0.0": {{
            "dimension_name": "Methodological Rigor", 
            "dimension_score": <number between 0-100>,
            "strengths": ["<strength 1>", "<strength 2>"],
            "weaknesses": ["<weakness 1>", "<weakness 2>"]
        }},
        "3.0.0": {{
            "dimension_name": "Structure & Coherence",
            "dimension_score": <number between 0-100>, 
            "strengths": ["<strength 1>", "<strength 2>"],
            "weaknesses": ["<weakness 1>", "<weakness 2>"]
        }}
    }}
}}

--- PAPER START ---
{paper_content}
--- PAPER END ---
"""

        try:
            # 2) Run the sync OpenAI call off the event loop so we don't block
            resp = await asyncio.to_thread(
                client.chat.completions.create,
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are PACT, a precise academic critique engine. Always return valid JSON."},
                    {"role": "user", "content": prompt},
                ],
            )

            # 3) Extract content
            text = (resp.choices[0].message.content or "").strip()
            logger.info(f"Received response from OpenAI: {text[:200]}...")

            if session_id and websocket_manager:
                await websocket_manager.send_message(session_id, {
                    "event": "progress", 
                    "progress": 85, 
                    "message": "Formatting results"
                })

            # 4) Try to parse JSON; if not, wrap a fallback
            result = _safe_extract_json(text)
            
            if not result:
                logger.warning("Could not parse JSON from OpenAI response, creating fallback")
                result = {
                    "paper_title": paper_title,
                    "overall_score": 75,
                    "final_critique": f"Analysis completed for '{paper_title}'. Raw response: {text[:1000]}...",
                    "dimension_critiques": {
                        "1.0.0": {
                            "dimension_name": "Research Foundations",
                            "dimension_score": 75,
                            "strengths": ["Analysis attempted"],
                            "weaknesses": ["Could not parse structured response"]
                        },
                        "2.0.0": {
                            "dimension_name": "Methodological Rigor",
                            "dimension_score": 75,
                            "strengths": ["Analysis attempted"],
                            "weaknesses": ["Could not parse structured response"]
                        },
                        "3.0.0": {
                            "dimension_name": "Structure & Coherence", 
                            "dimension_score": 75,
                            "strengths": ["Analysis attempted"],
                            "weaknesses": ["Could not parse structured response"]
                        }
                    },
                    "raw_response": text[:2000]
                }

            # Ensure all required fields exist
            if "paper_title" not in result:
                result["paper_title"] = paper_title
            if "overall_score" not in result:
                result["overall_score"] = 75
            if "final_critique" not in result:
                result["final_critique"] = "Critique analysis completed."
            if "dimension_critiques" not in result:
                result["dimension_critiques"] = {}

            # 5) Final broadcast
            if session_id and websocket_manager:
                await websocket_manager.send_message(session_id, {
                    "event": "summary", 
                    "payload": result
                })

            logger.info(f"Real supervisor completed analysis for session {session_id}")
            return result

        except (APIConnectionError, RateLimitError, BadRequestError, APIError) as e:
            logger.exception("OpenAI error during critique")
            if session_id and websocket_manager:
                await websocket_manager.send_message(session_id, {
                    "event": "status", 
                    "status": "error", 
                    "message": f"OpenAI API error: {str(e)}"
                })
            raise
        except Exception as e:
            logger.exception("Unexpected error during critique")
            if session_id and websocket_manager:
                await websocket_manager.send_message(session_id, {
                    "event": "status", 
                    "status": "error", 
                    "message": f"Analysis error: {str(e)}"
                })
            raise
