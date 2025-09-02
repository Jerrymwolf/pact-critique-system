"""
Enhanced Schemas for Detailed PACT Critique System

This module defines comprehensive schemas for detailed subsection-level analysis
matching the professional PACT Analysis Report structure.
"""

from typing import Dict, List, Optional, Literal
from pydantic import BaseModel, Field

# Assessment levels matching the PACT rubric
AssessmentLevel = Literal["Inadequate", "Developing", "Competent", "Strong", "Exemplary"]
RecommendationLevel = Literal["Accept", "Minor Revision", "Revise", "Major Revision", "Reject"]

class SpecificEvidence(BaseModel):
    """Specific evidence or example from the paper."""
    text_excerpt: str = Field(description="Direct quote or paraphrase from the paper")
    location: str = Field(description="Section or paragraph reference")
    analysis: str = Field(description="Analysis of this evidence")

class DetailedStrength(BaseModel):
    """Detailed strength with evidence and explanation."""
    strength: str = Field(description="The strength identified")
    evidence: List[SpecificEvidence] = Field(description="Evidence supporting this strength")
    impact: str = Field(description="Why this strength is important")

class DetailedImprovement(BaseModel):
    """Detailed improvement area with specific recommendations."""
    issue: str = Field(description="The issue or weakness identified")
    evidence: List[SpecificEvidence] = Field(description="Evidence of this issue")
    recommendation: str = Field(description="Specific actionable recommendation")
    priority: Literal["High", "Medium", "Low"] = Field(description="Priority level")

class SubsectionAnalysis(BaseModel):
    """Detailed analysis of a PACT subsection (e.g., 1.1.1)."""
    subsection_id: str = Field(description="PACT subsection code (e.g., '1.1.1')")
    subsection_label: str = Field(description="Subsection name")
    score: float = Field(description="Subsection score (0-100)", ge=0, le=100)
    strengths: List[str] = Field(description="Specific strengths in this subsection")
    improvements: List[str] = Field(description="Areas needing improvement")

class SectionAnalysis(BaseModel):
    """Analysis of a PACT section (e.g., 1.1)."""
    section_id: str = Field(description="PACT section code (e.g., '1.1')")
    section_label: str = Field(description="Section name")
    score: float = Field(description="Section score (0-100)", ge=0, le=100)
    rationale: str = Field(description="Rationale for section assessment")
    subsections: Dict[str, SubsectionAnalysis] = Field(
        description="Analysis of subsections within this section",
        default_factory=dict
    )

class DetailedDimensionCritique(BaseModel):
    """Comprehensive dimension critique following proper hierarchy."""
    dimension_id: str = Field(description="PACT dimension ID (e.g., '1.0.0')")
    dimension_label: str = Field(description="Full dimension name")
    overall_score: float = Field(description="Overall dimension score (0-100)", ge=0, le=100)
    rationale: str = Field(description="Overall rationale for this dimension")

    # Proper hierarchy: dimensions have sections, sections have subsections
    sections: Dict[str, SectionAnalysis] = Field(
        description="Analysis of sections within this dimension",
        default_factory=dict
    )

class PACTChecklistItem(BaseModel):
    """Individual checklist item for revision tracking."""
    code: str = Field(description="PACT code (e.g., '1.1.2')")
    name: str = Field(description="Element name")
    description: str = Field(description="What needs to be addressed")
    priority: Literal["Critical", "Important", "Recommended"] = Field(description="Priority level")
    completed: bool = Field(default=False, description="Whether item has been addressed")

class SubmissionReadiness(BaseModel):
    """Assessment of submission readiness."""
    overall_readiness: Literal["Ready", "Minor Revisions Needed", "Major Revisions Needed", "Not Ready"] = Field(
        description="Overall readiness assessment"
    )
    readiness_score: float = Field(description="Readiness score (1-5)", ge=1, le=5)
    recommendation: RecommendationLevel = Field(description="Publication recommendation")
    justification: str = Field(description="Explanation of the recommendation")
    estimated_revision_time: str = Field(description="Estimated time to address issues")

class ComprehensiveCritique(BaseModel):
    """Complete comprehensive critique matching the PDF report structure."""

    # Document metadata
    document_title: Optional[str] = Field(description="Paper title")
    analysis_date: str = Field(description="Analysis completion date")
    analysis_type: str = Field(default="Comprehensive", description="Type of analysis")

    # Overall assessment
    overall_assessment: AssessmentLevel = Field(description="Overall paper assessment")
    overall_score: float = Field(description="Overall score (0-100)", ge=0, le=100)
    submission_readiness: SubmissionReadiness = Field(description="Submission readiness assessment")

    # Dimension analyses
    dimension_analyses: Dict[str, DetailedDimensionCritique] = Field(
        description="Detailed analysis for each PACT dimension",
        default_factory=dict
    )

    # Summary sections
    executive_summary: str = Field(description="Executive summary of the entire critique")
    key_findings: List[str] = Field(description="Key findings across all dimensions")

    # Actionable elements
    checklist_items: List[PACTChecklistItem] = Field(description="Revision checklist", default_factory=list)
    next_steps: List[str] = Field(description="Prioritized next steps", default_factory=list)

    # Statistics
    total_strengths_identified: int = Field(description="Total number of strengths found")
    total_improvements_needed: int = Field(description="Total number of improvements needed")
    critical_issues_count: int = Field(description="Number of critical issues")

# PACT Dimension Definitions with subsections
PACT_DIMENSIONS = {
    "1.0.0": {
        "name": "Research Foundations",
        "subsections": {
            "1.1.1": "Problem Identification and Significance",
            "1.1.2": "Research Question Formulation", 
            "1.3.1": "Literature Synthesis as Argument",
            "1.3.2": "Critical Source Engagement",
            "1.3.3": "Gap Identification and Significance",
            "1.4.1": "Claim-Evidence Alignment",
            "1.4.2": "Logical Flow and Coherence"
        }
    },
    "2.0.0": {
        "name": "Methodological Rigor",
        "subsections": {
            "2.1.1": "Method-Question Alignment",
            "2.1.2": "Methodological Transparency",
            "2.2.1": "Data Collection Rigor",
            "2.2.2": "Sample Appropriateness",
            "2.3.1": "Analysis Validity",
            "2.3.2": "Limitation Recognition"
        }
    },
    "3.0.0": {
        "name": "Structure & Coherence", 
        "subsections": {
            "3.1.2": "Introduction and Conclusion Excellence",
            "3.2.1": "Paragraph Focus and Unity",
            "3.3.3": "Rhythm and Pacing",
            "3.4.1": "Introduction-Body-Conclusion Alignment",
            "3.4.2": "Theoretical Framework Integration"
        }
    },
    "4.0.0": {
        "name": "Academic Precision",
        "subsections": {
            "4.1.1": "Term Definition and Consistency",
            "4.1.2": "Sentence Construction and Clarity",
            "4.2.1": "Citation Accuracy and Integrity", 
            "4.3.1": "Grammar, Syntax, and Spelling"
        }
    },
    "5.0.0": {
        "name": "Critical Sophistication",
        "subsections": {
            "5.1.1": "Reflexivity and Positionality",
            "5.1.2": "Cultural Sensitivity and Inclusivity",
            "5.2.1": "Original Contribution",
            "5.2.2": "Theoretical Advancement",
            "5.3.1": "Scholarly Maturity"
        }
    }
}

def get_dimension_subsections(dimension_id: str) -> Dict[str, str]:
    """Get subsections for a specific PACT dimension."""
    return PACT_DIMENSIONS.get(dimension_id, {}).get("subsections", {})

def get_all_subsection_codes() -> List[str]:
    """Get all PACT subsection codes."""
    codes = []
    for dim_data in PACT_DIMENSIONS.values():
        codes.extend(dim_data.get("subsections", {}).keys())
    return sorted(codes)