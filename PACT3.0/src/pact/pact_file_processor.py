
"""
File Processing Utilities for PACT Critique System

Handles various file formats for paper submission.
"""

import os
from pathlib import Path
from typing import Optional
import PyPDF2
import docx

def read_paper_from_file(file_path: str) -> str:
    """
    Read paper content from various file formats.
    
    Supports: .txt, .pdf, .docx, .md
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    extension = path.suffix.lower()
    
    if extension == '.txt' or extension == '.md':
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    
    elif extension == '.pdf':
        text = ""
        with open(path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text
    
    elif extension == '.docx':
        doc = docx.Document(path)
        return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
    
    else:
        raise ValueError(f"Unsupported file format: {extension}")

def save_critique_to_file(critique: str, output_path: str, format: str = 'md') -> str:
    """
    Save critique to file in specified format.
    """
    path = Path(output_path)
    
    if format == 'md':
        path = path.with_suffix('.md')
        with open(path, 'w', encoding='utf-8') as f:
            f.write(critique)
    
    elif format == 'txt':
        path = path.with_suffix('.txt')
        # Convert markdown to plain text (basic conversion)
        plain_text = critique.replace('#', '').replace('*', '').replace('_', '')
        with open(path, 'w', encoding='utf-8') as f:
            f.write(plain_text)
    
    elif format == 'html':
        path = path.with_suffix('.html')
        import markdown
        html_content = markdown.markdown(critique)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>PACT Critique Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; }}
        h3 {{ color: #888; }}
    </style>
</head>
<body>
{html_content}
</body>
</html>""")
    
    return str(path)

async def critique_paper_file(file_path: str, output_dir: Optional[str] = None) -> str:
    """
    Critique a paper from a file and save results.
    """
    from pact.pact_critique_agent import pact_critique_agent
    from langchain_core.messages import HumanMessage
    
    # Read paper content
    paper_content = read_paper_from_file(file_path)
    
    # Run critique
    result = await pact_critique_agent.ainvoke(
        {"messages": [HumanMessage(content=paper_content)]}
    )
    
    # Save critique if output directory specified
    if output_dir and 'final_critique' in result:
        output_path = Path(output_dir) / f"{Path(file_path).stem}_critique"
        saved_path = save_critique_to_file(
            result['final_critique'], 
            str(output_path), 
            format='md'
        )
        print(f"Critique saved to: {saved_path}")
    
    return result.get('final_critique', 'Critique generation failed')