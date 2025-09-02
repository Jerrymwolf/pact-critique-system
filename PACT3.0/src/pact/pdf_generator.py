"""
Professional PDF Generation for PACT Critique Reports

This module generates professional PDF reports matching the quality and structure
of the reference PACT Analysis Report using ReportLab.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, 
    PageBreak, Image, KeepTogether, Flowable
)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY

from .enhanced_schemas import ComprehensiveCritique, DetailedDimensionCritique

# PennCLO Brand Colors (based on the logo)
PENN_NAVY = colors.Color(0.2, 0.3, 0.5)  # Navy blue
PENN_BURGUNDY = colors.Color(0.6, 0.2, 0.2)  # Burgundy/Maroon
PENN_GRAY = colors.Color(0.5, 0.5, 0.5)  # Gray
PENN_LIGHT_GRAY = colors.Color(0.9, 0.9, 0.9)  # Light gray

class PACTReportGenerator:
    """Professional PDF report generator for PACT critiques."""
    
    def __init__(self, assets_dir: str = None):
        """Initialize the PDF generator."""
        if assets_dir is None:
            assets_dir = Path(__file__).parent / "assets"
        self.assets_dir = Path(assets_dir)
        self.logo_path = self.assets_dir / "pennCLO_logo.png"
        
        # Initialize styles
        self._setup_styles()
    
    def _setup_styles(self):
        """Set up custom styles for the report."""
        self.styles = getSampleStyleSheet()
        
        # Title style
        self.styles.add(ParagraphStyle(
            name='ReportTitle',
            parent=self.styles['Title'],
            fontSize=24,
            textColor=PENN_NAVY,
            spaceAfter=20,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        # Header styles
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading1'],
            fontSize=16,
            textColor=PENN_NAVY,
            spaceBefore=20,
            spaceAfter=10,
            fontName='Helvetica-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            name='SubsectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=PENN_BURGUNDY,
            spaceBefore=15,
            spaceAfter=8,
            fontName='Helvetica-Bold'
        ))
        
        # Body text styles
        self.styles.add(ParagraphStyle(
            name='BodyText',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.black,
            spaceAfter=6,
            alignment=TA_JUSTIFY
        ))
        
        self.styles.add(ParagraphStyle(
            name='BulletPoint',
            parent=self.styles['Normal'],
            fontSize=10,
            leftIndent=20,
            bulletIndent=10,
            spaceAfter=4
        ))
        
        # Assessment styles
        self.styles.add(ParagraphStyle(
            name='AssessmentStrong',
            parent=self.styles['Normal'],
            fontSize=12,
            textColor=colors.darkgreen,
            fontName='Helvetica-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            name='AssessmentDeveloping',
            parent=self.styles['Normal'],
            fontSize=12,
            textColor=colors.orange,
            fontName='Helvetica-Bold'
        ))
    
    def generate_comprehensive_report(self, critique: ComprehensiveCritique, 
                                    session_id: str, output_path: str = None) -> str:
        """
        Generate a comprehensive PDF report matching the reference structure.
        
        Args:
            critique: The comprehensive critique data
            session_id: Session identifier
            output_path: Optional output path, defaults to generated filename
            
        Returns:
            Path to the generated PDF file
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            output_path = f"PACT_Analysis_Report_{timestamp}.pdf"
        
        # Create the document
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # Build the story (content)
        story = []
        
        # Page 1: Header and Overall Assessment
        story.extend(self._build_header_section(critique))
        story.extend(self._build_overall_assessment(critique))
        story.extend(self._build_dimension_scores(critique))
        story.extend(self._build_summary_analysis(critique))
        
        # Page 2: PACT Findings Checklist
        story.append(PageBreak())
        story.extend(self._build_checklist_section(critique))
        
        # Pages 3+: Detailed Element Analysis
        story.append(PageBreak())
        story.extend(self._build_detailed_analysis(critique))
        
        # Build the PDF
        doc.build(story)
        return output_path
    
    def _build_header_section(self, critique: ComprehensiveCritique) -> list:
        """Build the header section with logo and title."""
        elements = []
        
        # Logo and title in a table
        if self.logo_path.exists():
            logo = Image(str(self.logo_path), width=120, height=120)
            
            # Create header table
            header_data = [
                [logo, Paragraph("PACT Comprehensive Report", self.styles['ReportTitle'])]
            ]
            
            header_table = Table(header_data, colWidths=[140, 400])
            header_table.setStyle(TableStyle([
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('BACKGROUND', (0, 0), (-1, -1), PENN_NAVY),
                ('TEXTCOLOR', (1, 0), (1, 0), colors.white),
            ]))
            
            elements.append(header_table)
        else:
            # Fallback without logo
            elements.append(Paragraph("PACT Comprehensive Report", self.styles['ReportTitle']))
        
        elements.append(Spacer(1, 20))
        
        # Document info
        generated_date = datetime.now().strftime("%m/%d/%Y, %I:%M:%S %p")
        elements.append(Paragraph(f"Generated: {generated_date}", self.styles['BodyText']))
        
        if critique.document_title:
            elements.append(Paragraph(f"Document: {critique.document_title}", self.styles['BodyText']))
        
        elements.append(Spacer(1, 20))
        
        return elements
    
    def _build_overall_assessment(self, critique: ComprehensiveCritique) -> list:
        """Build the overall assessment section."""
        elements = []
        
        elements.append(Paragraph("Overall Assessment", self.styles['SectionHeader']))
        
        # Assessment level
        assessment_style = self.styles['AssessmentStrong'] if critique.overall_assessment == 'Strong' else self.styles['AssessmentDeveloping']
        elements.append(Paragraph(f"Overall Assessment: {critique.overall_assessment}", assessment_style))
        
        # Analysis details
        elements.append(Paragraph(f"Analysis completed: {critique.analysis_date}", self.styles['BodyText']))
        elements.append(Paragraph(f"Analysis type: {critique.analysis_type}", self.styles['BodyText']))
        
        elements.append(Spacer(1, 15))
        return elements
    
    def _build_dimension_scores(self, critique: ComprehensiveCritique) -> list:
        """Build the dimension scores section."""
        elements = []
        
        elements.append(Paragraph("Dimension Scores", self.styles['SectionHeader']))
        
        for dim_id, dimension in critique.dimension_analyses.items():
            elements.append(Paragraph(
                f"{dimension.dimension_name.upper()}: {dimension.overall_assessment}",
                self.styles['BodyText']
            ))
        
        elements.append(Spacer(1, 15))
        return elements
    
    def _build_summary_analysis(self, critique: ComprehensiveCritique) -> list:
        """Build the summary analysis section."""
        elements = []
        
        elements.append(Paragraph("Summary Analysis", self.styles['SectionHeader']))
        
        # Overall Assessment
        elements.append(Paragraph("Overall Assessment:", self.styles['SubsectionHeader']))
        elements.append(Paragraph(critique.executive_summary, self.styles['BodyText']))
        
        # Revision Recommendation
        elements.append(Paragraph("Revision Recommendation:", self.styles['SubsectionHeader']))
        readiness = critique.submission_readiness
        elements.append(Paragraph(f"Level: {readiness.overall_readiness.lower()} (Score: {readiness.readiness_score}/5)", self.styles['BodyText']))
        elements.append(Paragraph(readiness.justification, self.styles['BodyText']))
        
        # Key Strengths
        if critique.key_findings:
            elements.append(Paragraph("Key Strengths:", self.styles['SubsectionHeader']))
            for strength in critique.key_findings[:4]:  # Limit to top 4
                elements.append(Paragraph(f"• {strength}", self.styles['BulletPoint']))
        
        # Priority Focus Areas
        if critique.next_steps:
            elements.append(Paragraph("Priority Focus Areas:", self.styles['SubsectionHeader']))
            for area in critique.next_steps[:3]:  # Top 3 priorities
                elements.append(Paragraph(f"• {area}", self.styles['BulletPoint']))
        
        return elements
    
    def _build_checklist_section(self, critique: ComprehensiveCritique) -> list:
        """Build the PACT findings checklist section."""
        elements = []
        
        elements.append(Paragraph("PACT Analysis Report", self.styles['ReportTitle']))
        elements.append(Paragraph(f"Generated: {datetime.now().strftime('%m/%d/%Y, %I:%M:%S %p')}", self.styles['BodyText']))
        elements.append(Spacer(1, 20))
        
        # Submission Readiness
        elements.append(Paragraph("Submission Readiness:", self.styles['SectionHeader']))
        elements.append(Paragraph(critique.submission_readiness.justification, self.styles['BodyText']))
        elements.append(Spacer(1, 15))
        
        # Checklist
        elements.append(Paragraph("PACT Findings Checklist", self.styles['SectionHeader']))
        elements.append(Paragraph("Use this checklist to track your revision progress. Check off items as you address them.", self.styles['BodyText']))
        elements.append(Spacer(1, 10))
        
        # Areas for Improvement
        improvement_items = [item for item in critique.checklist_items if not item.completed]
        if improvement_items:
            elements.append(Paragraph(f"Areas for Improvement ({len(improvement_items)}) - Should Be Addressed", 
                                    self.styles['SubsectionHeader']))
            
            for item in improvement_items:
                checkbox = "☐"  # Empty checkbox
                elements.append(Paragraph(f"{checkbox} <b>{item.code}</b> {item.name}", self.styles['BodyText']))
                elements.append(Paragraph(item.description, self.styles['BulletPoint']))
                elements.append(Spacer(1, 8))
        
        # Strengths
        strength_items = [item for item in critique.checklist_items if item.completed]
        if strength_items:
            elements.append(Paragraph(f"Strengths ({len(strength_items)}) - Well-Executed Elements", 
                                    self.styles['SubsectionHeader']))
            
            for item in strength_items:
                checkbox = "☑"  # Checked checkbox
                elements.append(Paragraph(f"{checkbox} <b>{item.code}</b> {item.name}", self.styles['BodyText']))
                elements.append(Paragraph(item.description, self.styles['BulletPoint']))
                elements.append(Spacer(1, 8))
        
        return elements
    
    def _build_detailed_analysis(self, critique: ComprehensiveCritique) -> list:
        """Build the detailed element analysis section."""
        elements = []
        
        elements.append(Paragraph("Detailed Element Analysis", self.styles['SectionHeader']))
        elements.append(Spacer(1, 15))
        
        for dim_id, dimension in critique.dimension_analyses.items():
            elements.extend(self._build_dimension_detail(dimension))
            elements.append(Spacer(1, 20))
        
        return elements
    
    def _build_dimension_detail(self, dimension: DetailedDimensionCritique) -> list:
        """Build detailed analysis for a single dimension."""
        elements = []
        
        # Dimension header
        elements.append(Paragraph(dimension.dimension_name.upper(), self.styles['SectionHeader']))
        elements.append(Spacer(1, 10))
        
        # Subsection analyses
        for subsection_code, subsection in dimension.subsections.items():
            elements.append(KeepTogether([
                Paragraph(f"{subsection_code}: {subsection.name}", self.styles['SubsectionHeader']),
                Paragraph(f"Assessment: {subsection.assessment}", self.styles['AssessmentStrong' if subsection.assessment == 'Strong' else 'BodyText']),
                Paragraph(subsection.detailed_feedback, self.styles['BodyText']),
                Spacer(1, 8)
            ]))
            
            # Strengths
            if subsection.strengths:
                elements.append(Paragraph("Strengths:", self.styles['BodyText']))
                for strength in subsection.strengths:
                    elements.append(Paragraph(f"• {strength}", self.styles['BulletPoint']))
            
            # Areas for Improvement
            if subsection.areas_for_improvement:
                elements.append(Paragraph("Areas for Improvement:", self.styles['BodyText']))
                for improvement in subsection.areas_for_improvement:
                    elements.append(Paragraph(f"• {improvement}", self.styles['BulletPoint']))
            
            elements.append(Spacer(1, 12))
        
        return elements

def generate_pact_pdf_report(critique_data: Dict[str, Any], session_id: str, 
                           output_dir: str = "reports") -> str:
    """
    Generate a PDF report from critique data.
    
    Args:
        critique_data: Dictionary containing the comprehensive critique
        session_id: Session identifier
        output_dir: Output directory for the PDF
        
    Returns:
        Path to the generated PDF file
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"PACT_Analysis_Report_{timestamp}.pdf"
    output_path = os.path.join(output_dir, filename)
    
    # Create the generator
    generator = PACTReportGenerator()
    
    # Convert dict to Pydantic model if needed
    if isinstance(critique_data, dict):
        critique = ComprehensiveCritique(**critique_data)
    else:
        critique = critique_data
    
    # Generate the report
    return generator.generate_comprehensive_report(critique, session_id, output_path)