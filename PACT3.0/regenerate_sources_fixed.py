#!/usr/bin/env python3
"""
Regenerate source files from notebook cells with %%writefile directives
"""

import os
import re
from pathlib import Path
import nbformat

def extract_writefile_cells(notebook_path):
    """Extract and execute %%writefile cells from notebook."""
    
    print(f"Reading notebook: {notebook_path}")
    
    # Read notebook
    with open(notebook_path, 'r') as f:
        nb = nbformat.read(f, as_version=4)
    
    writefile_count = 0
    
    for cell_idx, cell in enumerate(nb.cells):
        if cell.cell_type == 'code':
            source = cell.source
            
            # Check if this is a %%writefile cell
            if source.strip().startswith('%%writefile'):
                lines = source.strip().split('\n')
                first_line = lines[0]
                
                # Extract file path
                match = re.match(r'%%writefile\s+(.+)', first_line)
                if match:
                    file_path = match.group(1).strip()
                    
                    print(f"Cell {cell_idx}: %%writefile {file_path}")
                    
                    # Get the rest of the content (skip the %%writefile line)
                    content = '\n'.join(lines[1:])
                    
                    # Resolve relative path from notebooks directory  
                    if file_path.startswith('../'):
                        # We're running from PACT3.0 directory, notebook is in notebooks/
                        # So ../src/pact/file.py should become src/pact/file.py
                        file_path = file_path[3:]  # Remove ../
                    
                    full_path = Path(file_path)
                    
                    # Create directory if needed
                    full_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Write file
                    print(f"Writing: {file_path} ({len(content)} chars)")
                    with open(full_path, 'w') as f:
                        f.write(content)
                    
                    writefile_count += 1
                    
                    # Show first few lines for verification
                    content_lines = content.split('\n')
                    if len(content_lines) > 3:
                        print(f"  First lines: {content_lines[1][:50]}...")
    
    print(f"Generated {writefile_count} source files")

if __name__ == "__main__":
    notebook_path = "notebooks/pact_critique_agent.ipynb"
    if Path(notebook_path).exists():
        extract_writefile_cells(notebook_path)
        print("✓ Source files regenerated successfully!")
    else:
        print(f"❌ Notebook not found: {notebook_path}")
        print("Make sure you're running from the PACT3.0 directory")