"""
Specialized PACT Dimension Critique Agents

This module implements individual agents for each PACT dimension,
each specialized in evaluating specific aspects of academic writing.
"""

import os
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from .state_pact_critique import PaperCritiqueState, DimensionCritique
from .pact_taxonomy import load_pact_taxonomy, get_dimension_details
from .enhanced_schemas import DetailedDimensionCritique, PACT_DIMENSIONS, get_dimension_subsections

# Initialize model for critique agents with environment configuration
# ChatGPT 5 doesn't support custom temperature settings
critique_model = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "chatgpt-5"),
    max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "4000")),
    timeout=int(os.getenv("OPENAI_TIMEOUT", "120"))
)

def create_enhanced_dimension_critique_prompt(paper_content: str, dimension_id: str) -> str:
    """
    Create an enhanced critique prompt for detailed PACT subsection analysis.
    """
    dimension_info = PACT_DIMENSIONS.get(dimension_id, {})
    dimension_name = dimension_info.get('name', '')
    subsections = dimension_info.get('subsections', {})
    
    subsection_details = ""
    for code, name in subsections.items():
        subsection_details += f"\n{code}: {name}\n"
    
    return f"""
You are an expert academic reviewer conducting a comprehensive evaluation of the '{dimension_name}' dimension of academic papers using the PACT (PennCLO Academic Critique Taxonomy).

DIMENSION: {dimension_id} - {dimension_name}

SUBSECTIONS TO EVALUATE:
{subsection_details}

PAPER TO CRITIQUE:
---
{paper_content[:8000]}  # Extended for more detailed analysis
---

EVALUATION INSTRUCTIONS:
Provide a detailed, professional critique that includes:

1. OVERALL DIMENSION ASSESSMENT:
   - Rate the overall dimension: Inadequate, Developing, Competent, Strong, or Exemplary
   - Provide an executive summary (2-3 sentences)

2. SUBSECTION-BY-SUBSECTION ANALYSIS:
   For EACH subsection listed above, provide:
   - Assessment level (Inadequate/Developing/Competent/Strong/Exemplary)
   - Detailed feedback paragraph (100-150 words) explaining your assessment
   - 2-3 specific strengths (with text evidence where possible)
   - 2-3 areas for improvement (with specific recommendations)
   - Examples from the text that support your evaluation
   - Rubric score (1-5) if applicable

3. EVIDENCE-BASED EVALUATION:
   - Quote specific passages from the paper when possible
   - Reference paragraph numbers or section titles
   - Explain WHY each strength/weakness matters for academic quality

4. ACTIONABLE RECOMMENDATIONS:
   - Provide specific, concrete steps the author can take
   - Prioritize improvements by impact (High/Medium/Low priority)
   - Suggest resources or strategies where appropriate

5. PROFESSIONAL TONE:
   - Use constructive, supportive language
   - Focus on helping the author improve
   - Be specific rather than general in feedback
   - Match the professional tone of academic peer review

Your analysis will be used to generate a comprehensive PACT report, so be thorough and precise.
"""

def format_dimension_criteria(dimension_data: Dict[str, Any]) -> str:
    """
    Format the evaluation criteria for a dimension.
    """
    criteria = []
    sections = dimension_data.get('sections', {})
    
    for section_id, section_data in sections.items():
        criteria.append(f"\n{section_id}: {section_data.get('name')}")
        
        # Add subsections if available
        subsections = section_data.get('subsections', {})
        for subsection_id, subsection_data in subsections.items():
            criteria.append(f"  - {subsection_id}: {subsection_data.get('name')}")
            
            # Add detection patterns if available
            patterns = subsection_data.get('detection_patterns', [])
            if patterns:
                criteria.append(f"    Detection patterns: {', '.join(patterns[:3])}")
    
    return '\n'.join(criteria)

async def critique_dimension_enhanced(state: PaperCritiqueState, dimension_id: str) -> Dict[str, Any]:
    """
    Enhanced critique for a PACT dimension with detailed subsection analysis.
    
    Args:
        state: The current critique state
        dimension_id: The PACT dimension to evaluate (e.g., "1.0.0")
    
    Returns:
        Dictionary with the detailed dimension critique
    """
    # Create enhanced critique prompt
    prompt = create_enhanced_dimension_critique_prompt(state['paper_content'], dimension_id)
    
    # Get detailed critique from model
    structured_model = critique_model.with_structured_output(DetailedDimensionCritique)
    critique = await structured_model.ainvoke([HumanMessage(content=prompt)])
    
    # Ensure dimension info is set
    critique.dimension_id = dimension_id
    if dimension_id in PACT_DIMENSIONS:
        critique.dimension_name = PACT_DIMENSIONS[dimension_id]['name']
    
    return critique.dict()

# Keep the original function for backward compatibility
async def critique_dimension(state: PaperCritiqueState, dimension_id: str) -> Dict[str, Any]:
    """
    Original critique function - delegates to enhanced version.
    """
    return await critique_dimension_enhanced(state, dimension_id)

# Create specific agent functions for each dimension
async def critique_research_foundations(state: PaperCritiqueState) -> Dict[str, Any]:
    """Agent 1: Critique Research Foundations (1.0.0)"""
    critique = await critique_dimension(state, "1.0.0")
    return {"dimension_critiques": {"1.0.0": critique}}

async def critique_methodological_rigor(state: PaperCritiqueState) -> Dict[str, Any]:
    """Agent 2: Critique Methodological Rigor (2.0.0)"""
    critique = await critique_dimension(state, "2.0.0")
    return {"dimension_critiques": {"2.0.0": critique}}

async def critique_structure_coherence(state: PaperCritiqueState) -> Dict[str, Any]:
    """Agent 3: Critique Structure & Coherence (3.0.0)"""
    critique = await critique_dimension(state, "3.0.0")
    return {"dimension_critiques": {"3.0.0": critique}}

async def critique_academic_precision(state: PaperCritiqueState) -> Dict[str, Any]:
    """Agent 4: Critique Academic Precision (4.0.0)"""
    critique = await critique_dimension(state, "4.0.0")
    return {"dimension_critiques": {"4.0.0": critique}}

async def critique_critical_sophistication(state: PaperCritiqueState) -> Dict[str, Any]:
    """Agent 5: Critique Critical Sophistication (5.0.0)"""
    critique = await critique_dimension(state, "5.0.0")
    return {"dimension_critiques": {"5.0.0": critique}}