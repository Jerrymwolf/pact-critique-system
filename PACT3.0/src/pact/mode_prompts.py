
"""
Mode-specific prompt templates for PACT Agent System.
"""

from pact.mode_config import AgentMode

# Base system prompt for all modes
SYSTEM_BASE = """
You are a precise analysis agent. Always return valid JSON matching the expected structure.
Be concise, factual, and fit within specified length limits. Focus on actionable feedback.
"""

# Mode-specific prompts
MODE_PROMPTS = {
    AgentMode.APA7: """
APA7 MODE: Analyze ONLY APA Style compliance elements.
Focus on: TitlePage, Abstract, Headings, InTextCitations, ReferenceList, BiasFreeLanguage, 
Numbers, Lists, Tables, Figures, Mechanics, Punctuation, Italics, Capitals, Abbreviations, StyleConsistency.

For each detected issue:
- Identify the specific APA element 
- Assess severity (low, medium, high)
- Provide concrete, actionable fixes
- Include brief examples where helpful

Keep descriptions concise and fixes specific.
""",

    AgentMode.STANDARD: """
STANDARD MODE: Analyze selected PACT elements most relevant to the manuscript content.
Focus on areas where you can provide specific, evidence-based feedback.

Selection criteria:
- Areas with clear evidence in the text
- Sections where meaningful analysis is possible  
- Skip areas without sufficient content to evaluate

Provide detailed scores, rationales, strengths, and improvements for selected elements only.
""",

    AgentMode.COMPREHENSIVE: """
COMPREHENSIVE MODE: Analyze ALL PACT elements systematically and thoroughly.
Cover every dimension, section, and subsection in the taxonomy.

For each element provide:
- Detailed analysis and scoring
- Specific strengths and weaknesses
- Evidence-based rationales
- Actionable improvement recommendations

Be thorough but remain within token limits.
"""
}

def get_mode_prompt(mode: AgentMode) -> str:
    """Get the prompt template for a specific mode."""
    return MODE_PROMPTS.get(mode, MODE_PROMPTS[AgentMode.STANDARD])

# Supervisor prompts for planning
SUPERVISOR_PROMPTS = {
    AgentMode.STANDARD: """
Determine which PACT sections/subsections are most relevant to this manuscript.
Focus on areas with sufficient evidence for specific, actionable feedback.
Skip areas without clear signals in the text.
Return a focused analysis plan.
""",

    AgentMode.COMPREHENSIVE: """
Plan to analyze all PACT sections and subsections systematically.
Ensure complete coverage of the taxonomy for thorough evaluation.
Return a comprehensive analysis plan.
"""
}

def get_supervisor_prompt(mode: AgentMode) -> str:
    """Get supervisor prompt for planning phase."""
    return SUPERVISOR_PROMPTS.get(mode, SUPERVISOR_PROMPTS[AgentMode.STANDARD])
