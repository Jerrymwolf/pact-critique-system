
"""
Mode Configuration for PACT Agent System

Defines the three analysis modes: APA7, STANDARD, and COMPREHENSIVE
with their respective token limits and processing caps.
"""

from enum import Enum
from typing import Dict, Any
from dataclasses import dataclass

class AgentMode(str, Enum):
    APA7 = "APA7"
    STANDARD = "STANDARD"  
    COMPREHENSIVE = "COMPREHENSIVE"

@dataclass
class TokenLimits:
    outline: int = 600
    section: int = 900
    subsection: int = 800
    apa7: int = 900
    synthesis: int = 900

@dataclass
class ProcessingCaps:
    rationale_len_standard: int = 600
    rationale_len_deep: int = 1200
    rationale_len_apa7: int = 700
    bullet_len: int = 160
    bullets_essentials: int = 3
    bullets_standard: int = 5
    bullets_apa7: int = 5
    evidence_snippets: int = 2
    evidence_len: int = 240

class ModeConfig:
    """Configuration manager for different analysis modes."""
    
    def __init__(self):
        self.token_limits = TokenLimits()
        self.caps = ProcessingCaps()
    
    def get_mode_description(self, mode: AgentMode) -> str:
        """Get description of what each mode does."""
        descriptions = {
            AgentMode.APA7: "Analyze ONLY APA Style elements with targeted fixes",
            AgentMode.STANDARD: "Auto-select relevant PACT elements based on paper content", 
            AgentMode.COMPREHENSIVE: "Analyze ALL PACT elements exhaustively"
        }
        return descriptions[mode]
    
    def get_token_limit(self, node_type: str) -> int:
        """Get token limit for specific node type."""
        return getattr(self.token_limits, node_type, 800)
    
    def apply_skinny_caps(self, mode: AgentMode) -> Dict[str, Any]:
        """Apply reduced caps for retry scenarios."""
        base_caps = {
            "rationale_len": self.caps.rationale_len_standard,
            "bullet_len": self.caps.bullet_len,
            "bullets": self.caps.bullets_standard,
            "evidence_snippets": self.caps.evidence_snippets,
            "evidence_len": self.caps.evidence_len
        }
        
        if mode == AgentMode.APA7:
            base_caps["rationale_len"] = self.caps.rationale_len_apa7
            base_caps["bullets"] = self.caps.bullets_apa7
        
        # Apply 60% reduction for skinny retry
        return {
            "rationale_len": int(base_caps["rationale_len"] * 0.6),
            "bullet_len": base_caps["bullet_len"],
            "bullets": max(2, base_caps["bullets"] - 2),
            "evidence_snippets": max(1, base_caps["evidence_snippets"] - 1),
            "evidence_len": int(base_caps["evidence_len"] * 0.6)
        }

# Global instance
mode_config = ModeConfig()
