
"""
PACT Utilities Package

Provides utility functions and safe enum handling for the PACT system.
Maintains backward compatibility with the original utils.py module.
"""

# Import everything from common module for backward compatibility
from .common import *

# Import enum safety utilities
from .enum_safety import enum_value, safe_status_value, safe_mode_value

# Make specific functions available at package level
__all__ = [
    # From common.py (original utils.py)
    'get_today_str',
    'get_current_dir',
    'summarization_model',
    'tavily_client',
    'tavily_search_multiple',
    'summarize_webpage_content',
    'deduplicate_search_results',
    'process_search_results',
    'format_search_output',
    'tavily_search',
    'think_tool',
    
    # From enum_safety.py
    'enum_value',
    'safe_status_value',
    'safe_mode_value',
]
