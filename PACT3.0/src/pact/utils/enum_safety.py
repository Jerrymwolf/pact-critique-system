
"""
Safe Enum Value Accessor Utilities

Provides safe methods to extract values from enums or strings without errors.
"""

def enum_value(x):
    """
    Safely extract value from enum or return the object if it's already a string/value.
    
    Args:
        x: Enum instance, string, or any other value
        
    Returns:
        The enum's value if it has one, otherwise the original object
    """
    return getattr(x, "value", x)

def safe_status_value(status):
    """
    Safely extract status value with backward compatibility.
    
    Args:
        status: Status enum or string
        
    Returns:
        String representation of the status
    """
    if hasattr(status, 'value'):
        return status.value
    return str(status)

def safe_mode_value(mode):
    """
    Safely extract mode value.
    
    Args:
        mode: Mode enum or string
        
    Returns:
        String representation of the mode
    """
    if hasattr(mode, 'value'):
        return mode.value
    return str(mode) if mode else "STANDARD"
