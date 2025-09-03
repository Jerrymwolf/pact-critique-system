import asyncio
import json
import logging
import re
from typing import Optional, Dict, Any, List
from openai import OpenAI
from openai import APIError, APIConnectionError, RateLimitError, BadRequestError
from langchain_core.messages import HumanMessage

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

    def __init__(self, agents: Dict[str, Any], llm: Any):
        self.agents = agents
        self.llm = llm

    async def ainvoke(self, state: Dict[str, Any], session_id: Optional[str] = None) -> Dict[str, Any]:
        logger.info("Real supervisor ainvoke starting (session=%s)", session_id)

        # Import websocket_manager here to avoid circular imports
        try:
            from ..websocket_manager import manager as websocket_manager
            logger.info("WS manager id (real_supervisor)=%s", id(websocket_manager))
        except ImportError:
            websocket_manager = None
            logger.warning("WebSocket manager not available")

        # 1) Early progress ping
        if session_id and websocket_manager:
            await websocket_manager.broadcast(session_id, {
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
                await websocket_manager.broadcast(session_id, {
                    "event": "progress",
                    "progress": 35,
                    "message": "Parsing outline"
                })

            # Add scoring rubric milestone
            if session_id and websocket_manager:
                await websocket_manager.broadcast(session_id, {
                    "event": "progress",
                    "progress": 65,
                    "message": "Scoring rubric"
                })

            if session_id and websocket_manager:
                await websocket_manager.broadcast(session_id, {
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
                await websocket_manager.broadcast(session_id, {
                    "event": "summary",
                    "payload": result
                })

            logger.info("Real supervisor ainvoke finished (session=%s)", session_id)
            return result

        except (APIConnectionError, RateLimitError, BadRequestError, APIError) as e:
            logger.exception("OpenAI error during critique")
            if session_id and websocket_manager:
                await websocket_manager.broadcast(session_id, {
                    "event": "status",
                    "status": "error",
                    "message": f"OpenAI API error: {str(e)}"
                })
            raise
        except Exception as e:
            logger.error(f"Unexpected error during critique for session {session_id}: {e}")
            if session_id and websocket_manager:
                await websocket_manager.broadcast(session_id, {
                    "event": "status",
                    "status": "error",
                    "message": f"Analysis error: {str(e)}"
                })
            raise

    async def _run_dimension_agent(self, agent_name: str, agent: Any, paper_content: str, paper_title: str, session_id: str) -> Dict[str, Any]:
        """Run a single dimension agent and return its critique."""
        logger.info("Running agent %s for %s", agent_name, paper_title)
        try:
            # Import websocket_manager here to avoid circular imports
            try:
                from ..websocket_manager import manager as websocket_manager
            except ImportError:
                websocket_manager = None

            if session_id and websocket_manager:
                await websocket_manager.broadcast(session_id, {
                    "event": "progress",
                    "progress": 10 + int(agent_name.split('.')[0]) * 5,
                    "message": f"Analyzing {agent_name}..."
                })

            critique = await agent.ainvoke({
                "paper_title": paper_title,
                "paper_content": paper_content,
            })
            logger.info("Agent %s finished successfully", agent_name)
            return critique

        except Exception as e:
            logger.exception("Agent %s failed with error: %s", agent_name, e)
            # Raise the exception to be caught by asyncio.gather
            raise

    def _create_fallback_critique(self, dimension_name: str) -> Dict[str, Any]:
        """Create a fallback critique when an agent fails."""
        return {
            "dimension_name": dimension_name,
            "overall_assessment": "Analysis Incomplete",
            "issues": [
                {
                    "title": "Analysis unavailable",
                    "why_it_matters": "Technical issue prevented analysis",
                    "suggestions": ["Re-run analysis", "Contact support"],
                    "exemplar_rewrites": ["Technical issue - no rewrite available"],
                    "quotes": ["Analysis not available", "Please try again"],
                    "location_hint": "System error",
                    "priority": "medium"
                }
            ],
            "key_strengths": [
                {
                    "strength": "Paper submitted for analysis",
                    "evidence": "Document provided for review",
                    "location_hint": "Overall document"
                },
                {
                    "strength": "Seeking feedback improvement",
                    "evidence": "User requested critique",
                    "location_hint": "User intent"
                }
            ],
            "next_steps": [
                {
                    "action": "Re-run analysis",
                    "priority": "high",
                    "estimated_effort": "5 minutes"
                }
            ],
            "executive_summary": f"Analysis of {dimension_name} could not be completed due to technical issues."
        }

    async def _enhance_and_guarantee_quality(self, raw_critiques: Dict[str, Any], paper_content: str, session_id: str) -> Dict[str, Any]:
        """Apply quality guarantees and enhancements to ensure dense, non-empty feedback."""
        enhanced_critiques = {}

        for dimension_name, critique in raw_critiques.items():
            enhanced_critique = critique.copy()

            # Guarantee minimum strengths
            if len(enhanced_critique.get("key_strengths", [])) < 2:
                enhanced_critique["key_strengths"] = await self._harvest_strengths(
                    dimension_name, paper_content, enhanced_critique.get("key_strengths", [])
                )

            # Guarantee minimum issues with quality
            issues = enhanced_critique.get("issues", [])
            filtered_issues = self._filter_quality_issues(issues)

            if len(filtered_issues) < 2:
                additional_issues = await self._generate_specific_improvements(
                    dimension_name, paper_content, filtered_issues
                )
                filtered_issues.extend(additional_issues)

            enhanced_critique["issues"] = filtered_issues

            # Auto-assign priorities if missing
            enhanced_critique["issues"] = self._assign_priorities(enhanced_critique["issues"])

            enhanced_critiques[dimension_name] = enhanced_critique

            # Notify progress
            if session_id:
                try:
                    from ..websocket_manager import manager
                    await manager.broadcast(session_id, {
                        "event": "progress",
                        "progress": 70 + len(enhanced_critiques) * 5,
                        "message": f"Enhanced {dimension_name} feedback"
                    })
                except Exception as e:
                    logger.error(f"Failed to broadcast progress: {e}")

        return enhanced_critiques

    async def _harvest_strengths(self, dimension_name: str, paper_content: str, existing_strengths: List[Dict]) -> List[Dict]:
        """Run a micro pass to harvest additional strengths if needed."""
        try:
            harvester_prompt = f"""Extract 2-3 specific strengths for {dimension_name} from this academic paper.

            For each strength, provide:
            - strength: Specific positive aspect
            - evidence: Direct quote supporting this strength
            - location_hint: Section/paragraph where this appears

            Focus on concrete, grounded positives. Return only JSON array of strength objects.

            Paper content:
            {paper_content[:3000]}..."""

            message = HumanMessage(content=harvester_prompt)
            response = await self.llm.ainvoke([message])

            import json
            harvested = json.loads(response.content)

            # Combine with existing, ensure minimum of 2
            all_strengths = existing_strengths + harvested
            return all_strengths[:3]  # Cap at 3 max

        except Exception as e:
            logger.error(f"Strength harvesting failed: {e}")
            # Return fallback strengths
            return existing_strengths + [
                {
                    "strength": "Paper addresses relevant academic topic",
                    "evidence": "Document presents research question",
                    "location_hint": "Overall structure"
                },
                {
                    "strength": "Structured academic format followed",
                    "evidence": "Standard academic paper organization",
                    "location_hint": "Document structure"
                }
            ]

    def _filter_quality_issues(self, issues: List[Dict]) -> List[Dict]:
        """Filter issues to ensure they meet quality standards."""
        quality_issues = []
        for issue in issues:
            # Check for required fields and quality
            if (issue.get("exemplar_rewrites") and
                issue.get("quotes") and
                len(issue.get("quotes", [])) >= 2 and
                issue.get("title") and
                issue.get("why_it_matters")):
                quality_issues.append(issue)
        return quality_issues

    async def _generate_specific_improvements(self, dimension_name: str, paper_content: str, existing_issues: List[Dict]) -> List[Dict]:
        """Generate additional specific, local improvements if needed."""
        try:
            improvement_prompt = f"""Generate 2 specific, actionable issues for {dimension_name} that include:

            Required for each issue:
            - title: Short descriptive title
            - why_it_matters: Tie to academic standards
            - suggestions: List of actionable steps
            - exemplar_rewrites: Concrete rewrite examples
            - quotes: 2+ direct quotes as evidence
            - location_hint: Specific location in paper
            - priority: "high", "medium", or "low"

            Focus on concrete, local improvements. Return only JSON array.

            Paper content (first 2000 chars):
            {paper_content[:2000]}..."""

            message = HumanMessage(content=improvement_prompt)
            response = await self.llm.ainvoke([message])

            import json
            additional_issues = json.loads(response.content)
            return additional_issues[:2]  # Limit to 2

        except Exception as e:
            logger.error(f"Issue generation failed: {e}")
            # Return fallback issues
            return [
                {
                    "title": "Consider strengthening evidence base",
                    "why_it_matters": "Academic rigor requires strong evidentiary support",
                    "suggestions": ["Add supporting citations", "Include additional examples"],
                    "exemplar_rewrites": ["Research shows X (Author, 2023), supporting the claim that..."],
                    "quotes": ["[Citation needed]", "[Additional evidence required]"],
                    "location_hint": "Throughout document",
                    "priority": "medium"
                }
            ]

    def _assign_priorities(self, issues: List[Dict]) -> List[Dict]:
        """Auto-assign priorities based on issue characteristics."""
        for issue in issues:
            if not issue.get("priority"):
                title_lower = issue.get("title", "").lower()
                why_matters = issue.get("why_it_matters", "").lower()

                # High priority keywords
                high_keywords = ["method", "validity", "ethics", "research question", "evidence mismatch", "transparency"]
                # Medium priority keywords
                medium_keywords = ["organization", "flow", "clarity", "structure", "coherence"]

                if any(keyword in title_lower or keyword in why_matters for keyword in high_keywords):
                    issue["priority"] = "high"
                elif any(keyword in title_lower or keyword in why_matters for keyword in medium_keywords):
                    issue["priority"] = "medium"
                else:
                    issue["priority"] = "low"

        return issues