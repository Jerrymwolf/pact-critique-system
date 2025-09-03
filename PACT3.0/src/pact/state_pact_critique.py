"""
State Definitions and Schemas for PACT Critique System

This module defines the state objects and structured schemas used for
the multi-agent paper critique workflow.
"""

import operator
from typing import Optional, List, Dict, Any
from typing_extensions import Annotated

from pydantic import BaseModel, Field
from langgraph.graph import MessagesState

def merge_dicts(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two dictionaries, with b taking precedence over a."""
    result = dict(a)
    result.update(b)
    return result

# ===== STATE DEFINITIONS =====

class PaperCritiqueState(MessagesState):
    """
    Main state for the PACT critique system.
    
    Tracks the paper being critiqued, individual agent feedback,
    and the final synthesized critique.
    """
    # The student paper to critique
    paper_content: str
    
    # Analysis mode (APA7, STANDARD, COMPREHENSIVE)
    mode: Optional[str] = "STANDARD"
    
    # Paper metadata
    paper_title: Optional[str] = None
    paper_type: Optional[str] = None  # thesis, dissertation, article, etc.
    
    # Individual dimension critiques from specialized agents
    dimension_critiques: Annotated[Dict[str, Any], merge_dicts] = {}
    
    # Supervisor's analysis plan
    critique_plan: Optional[str] = None
    
    # Final synthesized critique
    final_critique: Optional[str] = None
    
    # Overall paper score (0-100)
    overall_score: Optional[float] = None
    
    # Priority areas for improvement
    priority_improvements: List[str] = []

# ===== STRUCTURED OUTPUT SCHEMAS =====

class DimensionCritique(BaseModel):
    """
    Schema for individual dimension critiques from specialized agents.
    """
    dimension_id: str = Field(
        description="The PACT dimension ID (e.g., '1.0.0')"
    )
    dimension_name: str = Field(
        description="The dimension name (e.g., 'Research Foundations')"
    )
    dimension_label: Optional[str] = Field(
        description="Alternative dimension label for compatibility",
        default=None
    )
    overall_assessment: str = Field(
        description="Qualitative assessment (e.g., 'Developing', 'Proficient')",
        default="Developing"
    )
    issues: List[Dict[str, Any]] = Field(
        description="Structured issues with rubric mapping and evidence",
        default_factory=list
    )
    key_strengths: List[str] = Field(
        description="Key strengths identified in this dimension",
        default_factory=list
    )
    priority_improvements: List[str] = Field(
        description="Priority areas for improvement",
        default_factory=list
    )
    strengths: List[str] = Field(
        description="Specific strengths identified in this dimension",
        default_factory=list
    )
    weaknesses: List[str] = Field(
        description="Specific weaknesses or areas for improvement",
        default_factory=list
    )
    specific_issues: List[Dict[str, str]] = Field(
        description="Specific issues with location and severity",
        default_factory=list
    )
    recommendations: List[str] = Field(
        description="Actionable recommendations for improvement",
        default_factory=list
    )
    rubric_scores: Dict[str, int] = Field(
        description="Rubric scores for subsections (1-5 scale)",
        default_factory=dict
    )
    dimension_score: float = Field(
        description="Overall score for this dimension (0-100)",
        ge=0, le=100
    )
    severity: str = Field(
        description="Overall severity level: Critical, Major, Moderate, Minor",
        default="Moderate"
    )

class CritiquePlan(BaseModel):
    """
    Schema for the supervisor's critique plan.
    """
    paper_summary: str = Field(
        description="Brief summary of the paper's content and purpose"
    )
    initial_assessment: str = Field(
        description="Initial high-level assessment of paper quality"
    )
    dimensions_to_evaluate: List[str] = Field(
        description="List of PACT dimensions to evaluate",
        default_factory=lambda: ['1.0.0', '2.0.0', '3.0.0', '4.0.0', '5.0.0']
    )
    special_considerations: List[str] = Field(
        description="Any special considerations for this particular paper",
        default_factory=list
    )

class FinalCritique(BaseModel):
    """
    Schema for the final synthesized critique.
    """
    executive_summary: str = Field(
        description="Executive summary of the critique"
    )
    overall_assessment: str = Field(
        description="Overall assessment of the paper's quality"
    )
    dimension_summaries: Dict[str, str] = Field(
        description="Summary for each PACT dimension evaluated",
        default_factory=dict
    )
    key_strengths: List[str] = Field(
        description="Top 3-5 key strengths across all dimensions",
        default_factory=list
    )
    priority_improvements: List[str] = Field(
        description="Top 3-5 priority areas for improvement",
        default_factory=list
    )
    actionable_next_steps: List[str] = Field(
        description="Specific, actionable next steps for the author",
        default_factory=list
    )
    overall_score: float = Field(
        description="Overall paper score (0-100)",
        ge=0, le=100
    )
    recommendation: str = Field(
        description="Final recommendation: Accept, Revise, Major Revision, Reject"
    )