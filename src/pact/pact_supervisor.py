
"""
PACT Critique Supervisor Agent

This module implements the supervisor agent that coordinates the specialized
PACT dimension agents and synthesizes their feedback into a cohesive critique.
"""

from typing import Dict, Any, Literal
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from pact.state_pact_critique import (
    PaperCritiqueState, CritiquePlan, FinalCritique
)

# Initialize supervisor model
supervisor_model = init_chat_model(model="openai:gpt-4.1", temperature=0.1)

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
        critiques_text += f"\n\n--- {critique['dimension_name']} ({dim_id}) ---\n"
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
        critique_text += f"\n### {dim_id}
{summary}\n"
    
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