
"""
PACT Taxonomy Loader and Parser

This module loads and structures the PACT taxonomy for use by the critique agents.
"""

import json
import zipfile
import xml.etree.ElementTree as ET
from typing import Dict, Any, List
from pathlib import Path

def load_pact_taxonomy(file_path: str = "../PACT_JSON.docx") -> Dict[str, Any]:
    """
    Load the PACT taxonomy from the docx file.
    
    Returns a structured dictionary with the taxonomy dimensions.
    """
    # Check if we already have a parsed JSON version
    json_path = Path("../pact_taxonomy.json")
    if json_path.exists():
        with open(json_path, 'r') as f:
            return json.load(f)
    
    # Otherwise, parse from docx
    with zipfile.ZipFile(file_path, 'r') as docx:
        xml_content = docx.read('word/document.xml')
        tree = ET.fromstring(xml_content)
        
        namespace = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
        paragraphs = []
        for para in tree.iter(namespace + 'p'):
            texts = [node.text for node in para.iter(namespace + 't') if node.text]
            if texts:
                paragraphs.append(''.join(texts))
        
        # Parse JSON from text
        full_text = '\n'.join(paragraphs)
        json_start = full_text.find('{')
        if json_start != -1:
            json_text = full_text[json_start:]
            # Find matching closing brace
            brace_count = 0
            end_pos = 0
            for i, char in enumerate(json_text):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_pos = i + 1
                        break
            
            json_text = json_text[:end_pos]
            pact_data = json.loads(json_text)
            
            # Save for future use
            with open(json_path, 'w') as f:
                json.dump(pact_data, f, indent=2)
            
            return pact_data
    
    raise ValueError("Could not parse PACT taxonomy from file")

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