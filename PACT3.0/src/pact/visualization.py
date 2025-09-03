
"""
Visualization Components for PACT Critique System

This module provides spider chart and other visualizations for PACT analysis.
"""

import json
import base64
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO

def create_spider_chart(dimension_scores: Dict[str, float], 
                        dimension_names: Dict[str, str] = None) -> str:
    """
    Create a spider/radar chart for PACT dimension scores.
    
    Args:
        dimension_scores: Dictionary mapping dimension IDs to scores (0-100)
        dimension_names: Optional custom names for dimensions
        
    Returns:
        Base64 encoded image string for embedding in HTML
    """
    # Default dimension names
    if dimension_names is None:
        dimension_names = {
            "1.0.0": "Research\nFoundations",
            "2.0.0": "Methodological\nRigor",
            "3.0.0": "Structure &\nCoherence",
            "4.0.0": "Academic\nPrecision",
            "5.0.0": "Critical\nSophistication"
        }
    
    # Prepare data
    categories = []
    scores = []
    
    for dim_id in ["1.0.0", "2.0.0", "3.0.0", "4.0.0", "5.0.0"]:
        categories.append(dimension_names.get(dim_id, f"Dimension {dim_id}"))
        scores.append(dimension_scores.get(dim_id, 0))
    
    # Number of variables
    N = len(categories)
    
    # Compute angle for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    scores += scores[:1]  # Complete the circle
    angles += angles[:1]
    
    # Initialize the spider plot
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    # Draw the plot
    ax.plot(angles, scores, 'o-', linewidth=2, color='#667eea', label='Score')
    ax.fill(angles, scores, alpha=0.25, color='#667eea')
    
    # Fix axis to go in the right order and start at 12 o'clock
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=10)
    
    # Set y-axis limits and labels
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], size=8)
    
    # Add grid
    ax.grid(True)
    
    # Add title
    plt.title('PACT Dimension Analysis', size=14, weight='bold', pad=20)
    
    # Add legend with average score
    avg_score = np.mean(scores[:-1])
    ax.legend([f'Overall: {avg_score:.1f}/100'], loc='upper right', bbox_to_anchor=(1.2, 1.1))
    
    # Save to base64 string
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight', transparent=True)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return f"data:image/png;base64,{image_base64}"

def create_subsection_heatmap(subsection_scores: Dict[str, Dict[str, float]]) -> str:
    """
    Create a heatmap visualization for detailed subsection scores.
    
    Args:
        subsection_scores: Nested dict of dimension -> subsection -> score
        
    Returns:
        Base64 encoded heatmap image
    """
    import seaborn as sns
    
    # Prepare data matrix
    dimensions = sorted(subsection_scores.keys())
    subsections = set()
    for subs in subsection_scores.values():
        subsections.update(subs.keys())
    subsections = sorted(subsections)
    
    # Create matrix
    matrix = []
    for dim in dimensions:
        row = []
        for sub in subsections:
            row.append(subsection_scores[dim].get(sub, 0))
        matrix.append(row)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(matrix, 
                xticklabels=subsections, 
                yticklabels=dimensions,
                annot=True, 
                fmt='.1f',
                cmap='RdYlGn',
                vmin=0, 
                vmax=100,
                cbar_kws={'label': 'Score'})
    
    plt.title('Detailed Subsection Analysis', size=14, weight='bold')
    plt.xlabel('Subsections', size=12)
    plt.ylabel('Dimensions', size=12)
    plt.tight_layout()
    
    # Save to base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return f"data:image/png;base64,{image_base64}"

def generate_html_with_charts(critique_data: Dict[str, Any]) -> str:
    """
    Generate an HTML report with embedded visualizations.
    
    Args:
        critique_data: Complete critique data including scores
        
    Returns:
        HTML string with embedded charts
    """
    # Extract dimension scores
    dimension_scores = {}
    dimension_names = {}
    
    for dim_id, critique in critique_data.get('dimension_critiques', {}).items():
        dimension_scores[dim_id] = critique.get('dimension_score', 0)
        dimension_names[dim_id] = critique.get('dimension_name', f'Dimension {dim_id}')
    
    # Generate spider chart
    spider_chart = create_spider_chart(dimension_scores, dimension_names)
    
    # Generate HTML
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PACT Analysis Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .spider-chart {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
            margin-bottom: 30px;
        }}
        .dimension-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .score-badge {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            margin-left: 10px;
        }}
        .score-high {{ background: #4caf50; color: white; }}
        .score-medium {{ background: #ff9800; color: white; }}
        .score-low {{ background: #f44336; color: white; }}
        .issues-list {{
            list-style-type: none;
            padding-left: 0;
        }}
        .issue-item {{
            padding: 10px;
            margin: 10px 0;
            border-left: 4px solid #667eea;
            background: #f8f9fa;
        }}
        .priority-critical {{ border-left-color: #f44336; }}
        .priority-high {{ border-left-color: #ff9800; }}
        .priority-medium {{ border-left-color: #ffc107; }}
        .priority-low {{ border-left-color: #4caf50; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>PACT Comprehensive Analysis Report</h1>
        <p>Generated: {critique_data.get('timestamp', 'N/A')}</p>
        <p>Overall Score: <strong>{critique_data.get('overall_score', 0):.1f}/100</strong></p>
    </div>
    
    <div class="spider-chart">
        <h2>PACT Dimensions Overview</h2>
        <img src="{spider_chart}" alt="PACT Spider Chart" style="max-width: 600px;">
    </div>
    
    <div class="content">
        {generate_dimension_cards(critique_data)}
    </div>
</body>
</html>
    """
    
    return html

def generate_dimension_cards(critique_data: Dict[str, Any]) -> str:
    """Generate HTML cards for each dimension's detailed analysis."""
    cards_html = ""
    
    for dim_id, critique in critique_data.get('dimension_critiques', {}).items():
        score = critique.get('dimension_score', 0)
        score_class = 'score-high' if score >= 70 else 'score-medium' if score >= 50 else 'score-low'
        
        # Format issues
        issues_html = ""
        for issue in critique.get('issues', []):
            priority_class = f"priority-{issue.get('priority', 'medium').lower()}"
            issues_html += f"""
            <li class="issue-item {priority_class}">
                <strong>{issue.get('title', 'Issue')}</strong><br>
                <small>Location: {issue.get('rubric_id', 'N/A')}</small><br>
                {issue.get('why_it_matters', '')}
                {f"<br><em>Suggestion: {issue.get('rewrite', '')}</em>" if issue.get('rewrite') else ''}
            </li>
            """
        
        cards_html += f"""
        <div class="dimension-card">
            <h3>{critique.get('dimension_name', dim_id)} 
                <span class="score-badge {score_class}">{score:.0f}/100</span>
            </h3>
            <p><strong>Assessment:</strong> {critique.get('overall_assessment', 'N/A')}</p>
            
            <h4>Key Strengths</h4>
            <ul>
                {''.join(f"<li>{s}</li>" for s in critique.get('key_strengths', []))}
            </ul>
            
            <h4>Priority Improvements</h4>
            <ul class="issues-list">
                {issues_html}
            </ul>
        </div>
        """
    
    return cards_html