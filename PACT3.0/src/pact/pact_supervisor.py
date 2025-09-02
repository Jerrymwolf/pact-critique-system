"""
PACT Critique Supervisor Agent

This module implements the supervisor agent that coordinates the specialized
PACT dimension agents and synthesizes their feedback into a cohesive critique.
"""

import os
from typing import Dict, Any, Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage  # Import SystemMessage
from langgraph.types import Command

from .state_pact_critique import (
    PaperCritiqueState, CritiquePlan, FinalCritique
)
from .enhanced_schemas import (
    ComprehensiveCritique, SubmissionReadiness, PACTChecklistItem,
    PACT_DIMENSIONS, AssessmentLevel
)

# Initialize supervisor model with environment configuration
# ChatGPT 5 doesn't support custom temperature settings
supervisor_model = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-5"),
    max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "4000")),
    timeout=int(os.getenv("OPENAI_TIMEOUT", "120"))
)

def create_planning_prompt(paper_content: str) -> str:
    """
    Create a prompt for the supervisor to plan the critique.
    """
    return f"""
You are the lead reviewer coordinating a comprehensive academic paper critique using the PACT taxonomy.

Review the following paper and create a critique plan:

PAPER:
---
{paper_content[:3000]}  # Show first part for initial assessment
---

Create a plan that includes:
1. A brief summary of the paper's content and purpose
2. An initial assessment of overall quality and key areas of concern
3. Which PACT dimensions are most relevant to evaluate (all 5 by default)
4. Any special considerations for this particular paper

The PACT dimensions are:
1.0.0 - Research Foundations (problem, framework, literature)
2.0.0 - Methodological Rigor (methods, data, analysis)
3.0.0 - Structure & Coherence (organization, flow, transitions)
4.0.0 - Academic Precision (terms, citations, grammar)
5.0.0 - Critical Sophistication (reflexivity, originality, theory)
"""

def create_synthesis_prompt(state: PaperCritiqueState) -> str:
    """
    Create a prompt for synthesizing all dimension critiques.
    """
    # Format dimension critiques
    critiques_text = ""
    for dim_id, critique in state['dimension_critiques'].items():
        # Ensure dimension_name exists, default to dim_id if not
        dimension_name = critique.get('dimension_name', dim_id)
        critiques_text += f"\n\n--- {dimension_name} ({dim_id}) ---\n"
        critiques_text += f"Score: {critique['dimension_score']}/100\n"
        critiques_text += f"Severity: {critique['severity']}\n"

        if critique['strengths']:
            critiques_text += f"Strengths:\n"
            for strength in critique['strengths']:
                critiques_text += f"  • {strength}\n"

        if critique['weaknesses']:
            critiques_text += f"Weaknesses:\n"
            for weakness in critique['weaknesses']:
                critiques_text += f"  • {weakness}\n"

        if critique['recommendations']:
            critiques_text += f"Recommendations:\n"
            for rec in critique['recommendations']:
                critiques_text += f"  • {rec}\n"

    return f"""
You are synthesizing feedback from multiple expert reviewers into a cohesive, actionable critique.

CRITIQUE PLAN:
{state.get('critique_plan', 'No plan available')}

INDIVIDUAL DIMENSION CRITIQUES:
{critiques_text}

Create a comprehensive final critique that:
1. Provides an executive summary of the paper's overall quality
2. Synthesizes feedback across all dimensions
3. Identifies the top 3-5 key strengths
4. Identifies the top 3-5 priority areas for improvement
5. Provides specific, actionable next steps
6. Calculates an overall score (weighted average of dimension scores)
7. Makes a final recommendation (Accept, Revise, Major Revision, Reject)

Be constructive and supportive while maintaining academic rigor.
Focus on helping the author improve their work.
"""

async def plan_critique(state: PaperCritiqueState) -> Command[Literal["evaluate_dimensions"]]:
    """
    Supervisor plans the critique approach.
    """
    # Create planning prompt
    prompt = create_planning_prompt(state['paper_content'])

    # Get structured plan from model
    structured_model = supervisor_model.with_structured_output(CritiquePlan)
    plan = await structured_model.ainvoke([HumanMessage(content=prompt)])

    # Format plan as string for state
    plan_text = f"""
Paper Summary: {plan.paper_summary}

Initial Assessment: {plan.initial_assessment}

Dimensions to Evaluate: {', '.join(plan.dimensions_to_evaluate)}

Special Considerations:
{"; ".join(plan.special_considerations) if plan.special_considerations else "None"}
"""

    return Command(
        goto="evaluate_dimensions",
        update={"critique_plan": plan_text}
    )

async def synthesize_critique(state: PaperCritiqueState) -> Dict[str, Any]:
    """
    Supervisor synthesizes all dimension critiques into final feedback.
    """
    # Create synthesis prompt
    prompt = create_synthesis_prompt(state)

    # Get structured final critique from model
    structured_model = supervisor_model.with_structured_output(FinalCritique)
    final_critique = await structured_model.ainvoke([HumanMessage(content=prompt)])

    # Format final critique as markdown
    critique_text = f"""
# Academic Paper Critique Report

## Executive Summary
{final_critique.executive_summary}

## Overall Assessment
{final_critique.overall_assessment}

**Overall Score:** {final_critique.overall_score}/100
**Recommendation:** {final_critique.recommendation}

## Dimension Evaluations
"""

    for dim_id, summary in final_critique.dimension_summaries.items():
        # Ensure dimension_name exists, default to dim_id if not
        dimension_name = dim_id # This is a placeholder, ideally fetch from state if available
        critique_text += f"\n### {dimension_name} ({dim_id})\n{summary}\n"

    critique_text += f"""
## Key Strengths
"""
    for strength in final_critique.key_strengths:
        critique_text += f"- {strength}\n"

    critique_text += f"""
## Priority Areas for Improvement
"""
    for improvement in final_critique.priority_improvements:
        critique_text += f"- {improvement}\n"

    critique_text += f"""
## Actionable Next Steps
"""
    for i, step in enumerate(final_critique.actionable_next_steps, 1):
        critique_text += f"{i}. {step}\n"

    return {
        "final_critique": critique_text,
        "overall_score": final_critique.overall_score,
        "priority_improvements": final_critique.priority_improvements,
        "messages": [f"Critique complete. Overall score: {final_critique.overall_score}/100"]
    }

# Placeholder for logging and constants that might be used in the added code
import logging
logger = logging.getLogger(__name__)

SUPERVISOR_PROMPT = "You are a helpful assistant." # Placeholder

def create_critique_supervisor():
    """Create the supervisor agent workflow."""

    def supervisor_node(state: PaperCritiqueState) -> PaperCritiqueState:
        """Supervisor coordinates the critique process."""
        try:
            # Get mode from state (default to STANDARD)
            mode = state.get('mode', 'STANDARD')

            # Import mode-specific prompts
            # Note: pact.mode_prompts and its functions need to be defined elsewhere
            # For this example, we'll assume they exist and are importable.
            # If they are not, this will raise an ImportError.
            try:
                from pact.mode_prompts import get_supervisor_prompt, SYSTEM_BASE
            except ImportError:
                # Fallback if mode_prompts are not available, or provide a default behavior
                logger.warning("Could not import mode_prompts. Using default supervisor behavior.")
                # Default behavior if mode-specific prompts are not found
                mode_instruction = "" # Or some default instruction
                SYSTEM_BASE = "You are a helpful assistant." # Default system base

            mode_instruction = get_supervisor_prompt(mode)

            # Create critique plan
            plan_response = supervisor_model.invoke([
                SystemMessage(content=f"{SYSTEM_BASE}\n{mode_instruction}"),
                HumanMessage(content=f"Paper to critique:\n\n{state['paper_content'][:4000]}...")
            ])

            state['critique_plan'] = plan_response.content

            return state

        except Exception as e:
            logger.error(f"Error in supervisor node: {e}")
            state['critique_plan'] = f"Error creating critique plan: {str(e)}"
            return state