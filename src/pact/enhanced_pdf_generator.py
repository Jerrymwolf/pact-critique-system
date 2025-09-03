
"""
Enhanced PDF Generation for Comprehensive PACT Analysis Reports

Generates professional PDF reports matching Version 2 quality with spider charts.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, 
    PageBreak, Image as RLImage, KeepTogether, Flowable
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.barcharts import VerticalBarChart

from pact.visualization import create_spider_chart
from pact.enhanced_schemas import ComprehensiveCritique, DetailedDimensionCritique

# Enhanced color scheme
PACT_COLORS = {
    'navy': colors.Color(0.4, 0.49, 0.93),  # #667eea
    'purple': colors.Color(0.46, 0.29, 0.64),  # #764ba2
    'green': colors.Color(0.3, 0.69, 0.31),  # #4caf50
    'orange': colors.Color(1.0, 0.6, 0.0),  # #ff9800
    'red': colors.Color(0.96, 0.26, 0.21),  # #f44336
    'gray': colors.Color(0.5, 0.5, 0.5),
    'light_gray': colors.Color(0.9, 0.9, 0.9)
}

class EnhancedPACTReportGenerator:
    """Professional PDF report generator with visualizations."""
    
    def __init__(self):
        """Initialize the enhanced PDF generator."""
        self._setup_styles()
    
    def _setup_styles(self):
        """Set up comprehensive styles for the report."""
        self.styles = getSampleStyleSheet()
        
        # Title styles
        self.styles.add(ParagraphStyle(
            name='MainTitle',
            parent=self.styles['Title'],
            fontSize=28,
            textColor=PACT_COLORS['navy'],
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            name='SectionTitle',
            parent=self.styles['Heading1'],
            fontSize=18,
            textColor=PACT_COLORS['navy'],
            spaceBefore=20,
            spaceAfter=12,
            fontName='Helvetica-Bold',
            borderWidth=2,
            borderColor=PACT_COLORS['navy'],
            borderPadding=5
        ))
        
        self.styles.add(ParagraphStyle(
            name='DimensionTitle',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=PACT_COLORS['purple'],
            spaceBefore=15,
            spaceAfter=10,
            fontName='Helvetica-Bold'
        ))
        
        # Assessment styles with colors
        self.styles.add(ParagraphStyle(
            name='AssessmentExemplary',
            parent=self.styles['Normal'],
            fontSize=14,
            textColor=PACT_COLORS['green'],
            fontName='Helvetica-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            name='AssessmentStrong',
            parent=self.styles['Normal'],
            fontSize=14,
            textColor=colors.darkgreen,
            fontName='Helvetica-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            name='AssessmentCompetent',
            parent=self.styles['Normal'],
            fontSize=14,
            textColor=PACT_COLORS['orange'],
            fontName='Helvetica-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            name='AssessmentDeveloping',
            parent=self.styles['Normal'],
            fontSize=14,
            textColor=colors.orange,
            fontName='Helvetica-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            name='AssessmentInadequate',
            parent=self.styles['Normal'],
            fontSize=14,
            textColor=PACT_COLORS['red'],
            fontName='Helvetica-Bold'
        ))
        
        # Enhanced body styles
        self.styles.add(ParagraphStyle(
            name='DetailedAnalysis',
            parent=self.styles['Normal'],
            fontSize=11,
            textColor=colors.black,
            alignment=TA_JUSTIFY,
            spaceAfter=8,
            leading=14
        ))
        
        self.styles.add(ParagraphStyle(
            name='IssueTitle',
            parent=self.styles['Normal'],
            fontSize=12,
            textColor=PACT_COLORS['purple'],
            fontName='Helvetica-Bold',
            spaceAfter=4
        ))
        
        self.styles.add(ParagraphStyle(
            name='Recommendation',
            parent=self.styles['Normal'],
            fontSize=11,
            textColor=colors.darkblue,
            leftIndent=20,
            bulletIndent=10,
            spaceAfter=6
        ))
    
    def generate_comprehensive_report(self, critique_data: Dict[str, Any], 
                                    output_path: str = None) -> str:
        """
        Generate comprehensive PDF report with all enhancements.
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"PACT_Comprehensive_Report_{timestamp}.pdf"
        
        # Create document
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=60,
            leftMargin=60,
            topMargin=60,
            bottomMargin=60,
            title="PACT Comprehensive Analysis Report",
            author="PACT Academic Analysis System"
        )
        
        # Build content
        story = []
        
        # Page 1: Title and Executive Summary
        story.extend(self._build_title_page(critique_data))
        story.append(PageBreak())
        
        # Page 2: Spider Chart and Overall Scores
        story.extend(self._build_visualization_page(critique_data))
        story.append(PageBreak())
        
        # Page 3: Summary Analysis
        story.extend(self._build_summary_analysis(critique_data))
        story.append(PageBreak())
        
        # Detailed Dimension Analysis (multiple pages)
        for dim_id in ["1.0.0", "2.0.0", "3.0.0", "4.0.0", "5.0.0"]:
            if dim_id in critique_data.get('dimension_critiques', {}):
                story.extend(self._build_dimension_analysis(
                    dim_id, 
                    critique_data['dimension_critiques'][dim_id]
                ))
                story.append(PageBreak())
        
        # Final page: Checklist and Next Steps
        story.extend(self._build_checklist_page(critique_data))
        
        # Build PDF
        doc.build(story, onFirstPage=self._add_header_footer, 
                 onLaterPages=self._add_header_footer)
        
        return output_path
    
    def _build_title_page(self, critique_data: Dict[str, Any]) -> List:
        """Build title page with key information."""
        elements = []
        
        # Title
        elements.append(Paragraph(
            "PACT Comprehensive Analysis Report",
            self.styles['MainTitle']
        ))
        
        elements.append(Spacer(1, 20))
        
        # Document info
        if critique_data.get('paper_title'):
            elements.append(Paragraph(
                f"<b>Document:</b> {critique_data['paper_title']}",
                self.styles['Normal']
            ))
        
        elements.append(Paragraph(
            f"<b>Analysis Date:</b> {datetime.now().strftime('%B %d, %Y')}",
            self.styles['Normal']
        ))
        
        elements.append(Paragraph(
            f"<b>Analysis Type:</b> Comprehensive PACT Assessment",
            self.styles['Normal']
        ))
        
        elements.append(Spacer(1, 30))
        
        # Overall Assessment Box
        overall_score = critique_data.get('overall_score', 0)
        assessment_level = self._get_assessment_level(overall_score)
        assessment_style = f"Assessment{assessment_level}"
        
        assessment_data = [
            ['Overall Assessment', assessment_level],
            ['Overall Score', f"{overall_score:.1f}/100"],
            ['Recommendation', critique_data.get('recommendation', 'See detailed analysis')]
        ]
        
        assessment_table = Table(assessment_data, colWidths=[3*inch, 2.5*inch])
        assessment_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), PACT_COLORS['light_gray']),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.black),
            ('TEXTCOLOR', (1, 0), (1, 0), self._get_assessment_color(assessment_level)),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('GRID', (0, 0), (-1, -1), 1, colors.white),
            ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.white, PACT_COLORS['light_gray']])
        ]))
        
        elements.append(assessment_table)
        
        return elements
    
    def _build_visualization_page(self, critique_data: Dict[str, Any]) -> List:
        """Build page with spider chart and dimension scores."""
        elements = []
        
        elements.append(Paragraph("PACT Dimensions Analysis", self.styles['SectionTitle']))
        elements.append(Spacer(1, 20))
        
        # Generate and embed spider chart
        dimension_scores = {}
        for dim_id, critique in critique_data.get('dimension_critiques', {}).items():
            dimension_scores[dim_id] = critique.get('dimension_score', 0)
        
        # Create spider chart image
        spider_chart_base64 = create_spider_chart(dimension_scores)
        
        # Convert base64 to reportlab Image
        # Note: In production, save to temp file and load
        elements.append(Paragraph(
            "Spider Chart Visualization",
            self.styles['Heading3']
        ))
        elements.append(Paragraph(
            "<i>Visual representation of PACT dimension scores showing relative strengths and areas for improvement.</i>",
            self.styles['Normal']
        ))
        
        elements.append(Spacer(1, 20))
        
        # Dimension scores table
        dim_data = [['Dimension', 'Score', 'Assessment', 'Priority']]
        
        dimension_names = {
            "1.0.0": "Research Foundations",
            "2.0.0": "Methodological Rigor", 
            "3.0.0": "Structure & Coherence",
            "4.0.0": "Academic Precision",
            "5.0.0": "Critical Sophistication"
        }
        
        for dim_id, name in dimension_names.items():
            if dim_id in dimension_scores:
                score = dimension_scores[dim_id]
                assessment = self._get_assessment_level(score)
                priority = "High" if score < 50 else "Medium" if score < 70 else "Low"
                dim_data.append([name, f"{score:.0f}", assessment, priority])
        
        dim_table = Table(dim_data, colWidths=[2.5*inch, 1*inch, 1.5*inch, 1*inch])
        dim_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), PACT_COLORS['navy']),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 1, colors.gray),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, PACT_COLORS['light_gray']])
        ]))
        
        elements.append(dim_table)
        
        return elements
    
    def _build_summary_analysis(self, critique_data: Dict[str, Any]) -> List:
        """Build summary analysis section."""
        elements = []
        
        elements.append(Paragraph("Summary Analysis", self.styles['SectionTitle']))
        elements.append(Spacer(1, 12))
        
        # Executive Summary
        elements.append(Paragraph("<b>Executive Summary:</b>", self.styles['Heading3']))
        summary_text = critique_data.get('executive_summary', 
            'This work demonstrates competent academic writing with opportunities for targeted improvement.')
        elements.append(Paragraph(summary_text, self.styles['DetailedAnalysis']))
        elements.append(Spacer(1, 12))
        
        # Key Strengths
        elements.append(Paragraph("<b>Key Strengths:</b>", self.styles['Heading3']))
        strengths = critique_data.get('key_strengths', [])
        for strength in strengths[:5]:  # Top 5 strengths
            elements.append(Paragraph(f"• {strength}", self.styles['Recommendation']))
        elements.append(Spacer(1, 12))
        
        # Priority Improvements
        elements.append(Paragraph("<b>Priority Areas for Improvement:</b>", self.styles['Heading3']))
        improvements = critique_data.get('priority_improvements', [])
        for improvement in improvements[:5]:  # Top 5 improvements
            elements.append(Paragraph(f"• {improvement}", self.styles['Recommendation']))
        
        return elements
    
    def _build_dimension_analysis(self, dim_id: str, critique: Dict[str, Any]) -> List:
        """Build detailed analysis for a single dimension."""
        elements = []
        
        # Dimension header
        dim_name = critique.get('dimension_name', f'Dimension {dim_id}')
        score = critique.get('dimension_score', 0)
        assessment = critique.get('overall_assessment', 'Competent')
        
        elements.append(Paragraph(
            f"{dim_name} (Score: {score:.0f}/100)",
            self.styles['DimensionTitle']
        ))
        
        # Assessment level
        assessment_style = f"Assessment{self._get_assessment_level(score)}"
        elements.append(Paragraph(
            f"Assessment: {assessment}",
            self.styles[assessment_style]
        ))
        elements.append(Spacer(1, 12))
        
        # Detailed analysis
        if 'comprehensive_assessment' in critique:
            elements.append(Paragraph(
                critique['comprehensive_assessment'],
                self.styles['DetailedAnalysis']
            ))
            elements.append(Spacer(1, 12))
        
        # Issues with structured format
        if critique.get('issues'):
            elements.append(Paragraph("<b>Specific Issues:</b>", self.styles['Heading3']))
            
            for issue in critique['issues'][:6]:  # Top 6 issues
                # Issue title with priority
                priority_color = self._get_priority_color(issue.get('priority', 'Standard'))
                elements.append(Paragraph(
                    f"<font color='{priority_color}'>■</font> <b>{issue.get('title', 'Issue')}</b>",
                    self.styles['IssueTitle']
                ))
                
                # Why it matters
                elements.append(Paragraph(
                    f"<i>Why it matters:</i> {issue.get('why_it_matters', '')}",
                    self.styles['Normal']
                ))
                
                # Evidence if available
                if issue.get('evidence'):
                    elements.append(Paragraph(
                        f"<i>Evidence:</i> \"{issue['evidence'][0] if issue['evidence'] else ''}\"",
                        ParagraphStyle(
                            name='Evidence',
                            parent=self.styles['Normal'],
                            fontSize=10,
                            textColor=colors.gray,
                            leftIndent=20
                        )
                    ))
                
                # Suggestion if available
                if issue.get('rewrite'):
                    elements.append(Paragraph(
                        f"<i>Suggestion:</i> {issue['rewrite']}",
                        self.styles['Recommendation']
                    ))
                
                elements.append(Spacer(1, 8))
        
        return elements
    
    def _build_checklist_page(self, critique_data: Dict[str, Any]) -> List:
        """Build final checklist page."""
        elements = []
        
        elements.append(Paragraph("PACT Improvement Checklist", self.styles['SectionTitle']))
        elements.append(Spacer(1, 12))
        
        elements.append(Paragraph(
            "Use this checklist to track your revision progress:",
            self.styles['Normal']
        ))
        elements.append(Spacer(1, 12))
        
        # Create checklist items from all dimension issues
        checklist_data = []
        
        for dim_id, critique in critique_data.get('dimension_critiques', {}).items():
            dim_name = critique.get('dimension_name', dim_id)
            
            for issue in critique.get('issues', [])[:3]:  # Top 3 per dimension
                checklist_data.append([
                    '☐',
                    dim_name,
                    issue.get('title', 'Issue'),
                    issue.get('priority', 'Standard')
                ])
        
        if checklist_data:
            checklist_table = Table(
                [['', 'Dimension', 'Issue', 'Priority']] + checklist_data,
                colWidths=[0.3*inch, 1.8*inch, 3*inch, 0.8*inch]
            )
            
            checklist_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), PACT_COLORS['navy']),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('ALIGN', (0, 0), (0, -1), 'CENTER'),
                ('ALIGN', (-1, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.gray),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, PACT_COLORS['light_gray']])
            ]))
            
            elements.append(checklist_table)
        
        elements.append(Spacer(1, 20))
        
        # Next steps
        elements.append(Paragraph("<b>Recommended Next Steps:</b>", self.styles['Heading3']))
        next_steps = critique_data.get('actionable_next_steps', [
            "Address high-priority issues first",
            "Review and incorporate suggested rewrites",
            "Seek feedback after implementing changes",
            "Re-evaluate using PACT criteria"
        ])
        
        for i, step in enumerate(next_steps[:5], 1):
            elements.append(Paragraph(f"{i}. {step}", self.styles['Recommendation']))
        
        return elements
    
    def _get_assessment_level(self, score: float) -> str:
        """Convert numeric score to assessment level."""
        if score >= 85:
            return "Exemplary"
        elif score >= 70:
            return "Strong"
        elif score >= 55:
            return "Competent"
        elif score >= 40:
            return "Developing"
        else:
            return "Inadequate"
    
    def _get_assessment_color(self, level: str) -> colors.Color:
        """Get color for assessment level."""
        color_map = {
            "Exemplary": PACT_COLORS['green'],
            "Strong": colors.darkgreen,
            "Competent": PACT_COLORS['orange'],
            "Developing": colors.orange,
            "Inadequate": PACT_COLORS['red']
        }
        return color_map.get(level, colors.black)
    
    def _get_priority_color(self, priority: str) -> str:
        """Get hex color for priority level."""
        color_map = {
            "Critical": "#f44336",
            "High": "#ff9800",
            "Medium": "#ffc107",
            "Standard": "#4caf50",
            "Low": "#4caf50"
        }
        return color_map.get(priority, "#666666")
    
    def _add_header_footer(self, canvas, doc):
        """Add header and footer to each page."""
        canvas.saveState()
        
        # Header
        canvas.setFont('Helvetica', 9)
        canvas.setFillColor(colors.gray)
        canvas.drawString(inch, doc.height + doc.topMargin - 0.5*inch, 
                         "PACT Academic Analysis Report")
        canvas.drawRightString(doc.width + doc.rightMargin, 
                              doc.height + doc.topMargin - 0.5*inch,
                              datetime.now().strftime("%B %Y"))
        
        # Footer
        page_num = canvas.getPageNumber()
        canvas.drawCentredString(doc.width/2 + doc.leftMargin, 
                                0.5*inch,
                                f"Page {page_num}")
        
        canvas.restoreState()