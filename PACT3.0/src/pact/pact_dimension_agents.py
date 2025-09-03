
"""
Enhanced PACT Dimension Critique Agents with Deep Analysis

This module implements individual agents for each PACT dimension,
providing comprehensive, detailed analysis matching professional assessment quality.
"""

from typing import Dict, Any, List
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

from pact.state_pact_critique import PaperCritiqueState
from pact.enhanced_schemas import DetailedDimensionCritique, Issue
from pact.pact_taxonomy import load_pact_taxonomy, get_dimension_details

# Initialize model for critique agents with higher quality
critique_model = init_chat_model(model="openai:gpt-4o", temperature=0.2)

def create_enhanced_dimension_prompt(paper_content: str, dimension_data: Dict[str, Any]) -> str:
    """
    Create a comprehensive critique prompt for detailed PACT dimension analysis.
    """
    return f"""
You are an expert academic reviewer specializing in the '{dimension_data.get('name')}' dimension of academic writing.

Provide a COMPREHENSIVE, PROFESSIONAL-GRADE critique matching the depth of analysis in the PACT Analysis Report standard.

DIMENSION: {dimension_data.get('name')} ({dimension_data.get('id', 'N/A')})
Description: {dimension_data.get('description')}

EVALUATION FRAMEWORK:
{format_detailed_criteria(dimension_data)}

PAPER TO ANALYZE:
---
{paper_content[:10000]}  # Extended context for deeper analysis
---

REQUIRED ANALYSIS STRUCTURE:

1. **ELEMENT-BY-ELEMENT ANALYSIS**: 
   Analyze EACH specific PACT element within this dimension (e.g., 1.1.1, 1.1.2):
   - Element ID and Title (e.g., "1.1.1: Problem Identification and Significance")
   - Assessment Level: Inadequate/Developing/Competent/Strong/Exemplary  
   - 150-200 word detailed evaluation explaining the assessment
   - Specific strengths with evidence from the paper
   - Areas for improvement with actionable suggestions
   - Direct quotes or examples as supporting evidence

2. **COMPREHENSIVE ASSESSMENT** (200-300 words):
   - Overall evaluation of this dimension's execution in the paper
   - Patterns, recurring issues, and systemic strengths/weaknesses across elements
   - Context within academic standards for this document type
   - Submission readiness level assessment

2. **DETAILED STRENGTHS** (minimum 3-5, each 50-100 words):
   For each strength provide:
   - Clear identification of what was done well
   - Specific textual evidence (quotes or paraphrases with location)
   - Explanation of why this matters for academic quality
   - How it contributes to the paper's overall effectiveness

3. **CRITICAL ISSUES** (minimum 4-6 structured issues):
   For each issue provide:
   - Title: Clear, specific issue name
   - Location: Precise paragraph/section reference
   - Evidence: Direct quotes showing the problem
   - Why it matters: Academic principle or standard being violated
   - Suggested rewrite: Concrete example of improvement
   - Priority: Critical/High/Medium/Low

4. **SUBSECTION EVALUATION**:
   Analyze each subsection within this dimension:
   - Score (1-5) with detailed justification
   - Specific examples of excellence or deficiency
   - Targeted recommendations for that subsection

5. **ACTIONABLE RECOMMENDATIONS** (minimum 5):
   Prioritized, specific steps including:
   - What exactly to do
   - How to implement it
   - Expected impact on paper quality
   - Examples or templates where applicable

6. **SCORING**:
   - Dimension Score: 0-100 with justification
   - Qualitative Assessment: Inadequate/Developing/Competent/Strong/Exemplary
   - Detailed rationale referencing specific rubric criteria

Focus on academic rigor while remaining constructive. Provide the depth expected in professional peer review.
"""

def format_detailed_criteria(dimension_data: Dict[str, Any]) -> str:
    """
    Format comprehensive evaluation criteria with detection patterns.
    """
    criteria = []
    sections = dimension_data.get('sections', {})
    
    for section_id, section_data in sections.items():
        criteria.append(f"\n{section_id}: {section_data.get('name')}")
        criteria.append(f"   Purpose: {section_data.get('description', 'N/A')}")
        
        subsections = section_data.get('subsections', {})
        for sub_id, sub_data in subsections.items():
            criteria.append(f"\n   {sub_id}: {sub_data.get('name')}")
            
            # Include evaluation criteria
            if 'evaluation_criteria' in sub_data:
                criteria.append(f"      Criteria: {sub_data['evaluation_criteria']}")
            
            # Include detection patterns
            patterns = sub_data.get('detection_patterns', [])
            if patterns:
                criteria.append(f"      Look for: {'; '.join(patterns)}")
            
            # Include quality indicators
            if 'quality_indicators' in sub_data:
                criteria.append(f"      Excellence indicators: {sub_data['quality_indicators']}")
    
    return '\n'.join(criteria)

async def critique_dimension_enhanced(state: PaperCritiqueState, dimension_id: str) -> Dict[str, Any]:
    """
    Perform enhanced critique for a PACT dimension with detailed analysis.
    """
    # Load PACT taxonomy
    pact_data = load_pact_taxonomy()
    dimension_data = get_dimension_details(pact_data, dimension_id)
    
    if not dimension_data:
        raise ValueError(f"Dimension {dimension_id} not found in PACT taxonomy")
    
    # Add ID to dimension data
    dimension_data['id'] = dimension_id
    
    # Create enhanced prompt
    prompt = create_enhanced_dimension_prompt(state['paper_content'], dimension_data)
    
    # Get structured critique from model
    structured_model = critique_model.with_structured_output(DetailedDimensionCritique)
    critique = await structured_model.ainvoke([HumanMessage(content=prompt)])
    
    # Ensure dimension info is properly set
    critique.dimension_id = dimension_id
    critique.dimension_name = dimension_data.get('name', '')
    
    return critique.dict()

# Enhanced agent functions for each dimension
async def critique_research_foundations(state: PaperCritiqueState) -> Dict[str, Any]:
    """
    Agent 1: Deep analysis of Research Foundations (1.0.0)
    
    Evaluates: Problem identification, theoretical framework, literature review,
    research questions, hypotheses, and scholarly positioning.
    """
    critique = await critique_dimension_enhanced(state, "1.0.0")
    return {"dimension_critiques": {"1.0.0": critique}}

async def critique_methodological_rigor(state: PaperCritiqueState) -> Dict[str, Any]:
    """
    Agent 2: Deep analysis of Methodological Rigor (2.0.0)
    
    Evaluates: Research design, data collection, analysis methods,
    validity, reliability, ethics, and limitations.
    """
    critique = await critique_dimension_enhanced(state, "2.0.0")
    return {"dimension_critiques": {"2.0.0": critique}}

async def critique_structure_coherence(state: PaperCritiqueState) -> Dict[str, Any]:
    """
    Agent 3: Deep analysis of Structure & Coherence (3.0.0)
    
    Evaluates: Organization, logical flow, transitions, paragraph structure,
    introduction/conclusion quality, and overall coherence.
    """
    critique = await critique_dimension_enhanced(state, "3.0.0")
    return {"dimension_critiques": {"3.0.0": critique}}

async def critique_academic_precision(state: PaperCritiqueState) -> Dict[str, Any]:
    """
    Agent 4: Deep analysis of Academic Precision (4.0.0)
    
    Evaluates: Technical terminology, citation accuracy, grammar,
    formatting, academic style, and professional presentation.
    """
    critique = await critique_dimension_enhanced(state, "4.0.0")
    return {"dimension_critiques": {"4.0.0": critique}}

async def critique_critical_sophistication(state: PaperCritiqueState) -> Dict[str, Any]:
    """
    Agent 5: Deep analysis of Critical Sophistication (5.0.0)
    
    Evaluates: Critical thinking, theoretical depth, originality,
    reflexivity, nuanced argumentation, and scholarly maturity.
    """
    critique = await critique_dimension_enhanced(state, "5.0.0")
    return {"dimension_critiques": {"5.0.0": critique}}