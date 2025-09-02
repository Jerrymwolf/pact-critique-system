"""
PACT Taxonomy Loader and Parser

This module loads and structures the PACT taxonomy for use by the critique agents.
"""

import json
import zipfile
import xml.etree.ElementTree as ET
from typing import Dict, Any, List
from pathlib import Path

def load_pact_taxonomy(file_path: str = "pact_taxonomy.txt") -> Dict[str, Any]:
    """
    Load the PACT taxonomy from the txt file.
    
    Returns a structured dictionary with the taxonomy dimensions.
    """
    # Check if we already have a parsed JSON version
    json_path = Path("pact_taxonomy.json")
    if json_path.exists():
        with open(json_path, 'r') as f:
            return json.load(f)
    
    # Load from the txt file (which is actually JSON)
    try:
        with open(file_path, 'r') as f:
            pact_data = json.load(f)
            
        # Save for future use
        with open(json_path, 'w') as f:
            json.dump(pact_data, f, indent=2)
        
        return pact_data
    except Exception as e:
        print(f"Error loading PACT taxonomy: {e}")
        # Return basic structure if file not found
        return {
            "dimensions": {
                "1.0.0": {"name": "Research Foundations"},
                "2.0.0": {"name": "Methodological Rigor"},
                "3.0.0": {"name": "Structure & Coherence"},
                "4.0.0": {"name": "Academic Precision"},
                "5.0.0": {"name": "Critical Sophistication"}
            }
        }

def get_dimension_details(pact_data: Dict[str, Any], dimension_id: str) -> Dict[str, Any]:
    """
    Extract details for a specific PACT dimension.
    
    Args:
        pact_data: The full PACT taxonomy data
        dimension_id: The dimension ID (e.g., "1.0.0")
    
    Returns:
        Dictionary with dimension details including subsections
    """
    dimensions = pact_data.get('dimensions', {})
    return dimensions.get(dimension_id, {})

def get_all_dimensions(pact_data: Dict[str, Any]) -> List[tuple]:
    """
    Get all main dimensions from the PACT taxonomy.
    
    Returns:
        List of (dimension_id, dimension_name, dimension_data) tuples
    """
    dimensions = pact_data.get('dimensions', {})
    main_dimensions = []
    
    for dim_id in ['1.0.0', '2.0.0', '3.0.0', '4.0.0', '5.0.0']:
        if dim_id in dimensions:
            dim_data = dimensions[dim_id]
            main_dimensions.append((dim_id, dim_data.get('name'), dim_data))
    
    return main_dimensions