# src/pact/utils/__init__.py
from .common import *
from .enum_safety import enum_value, safe_status_value

# src/pact/utils/common.py
from enum import Enum

from langchain_openai import ChatOpenAI

# alias for clarity where you use status
safe_status_value = enum_value

# Define a default summarization model
summarization_model = ChatOpenAI(model="gpt-4o-mini")

# src/pact/utils/enum_safety.py
from enum import Enum

def enum_value(x):
    """Return Enum.value if present; otherwise x."""
    return getattr(x, "value", x)

# alias for clarity where you use status
safe_status_value = enum_value