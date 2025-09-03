#!/usr/bin/env python3
"""
Test script for enhanced PACT Critique System
"""

import asyncio
import json
from pathlib import Path
from datetime import datetime

# Test paper content
TEST_PAPER = """
Title: The Impact of Social Media on Academic Performance: A Mixed Methods Study

Abstract:
This study examines the relationship between social media usage and academic performance among 
undergraduate students. Using surveys and interviews with 200 students, we found that excessive 
social media use correlates with lower GPA scores. However, educational use of social media 
platforms showed positive effects on collaborative learning.

Introduction:
Social media has become ubiquitous in student life. Many educators worry about its impact on 
academic performance. This study investigates whether these concerns are justified. Previous 
research has shown mixed results, with some studies finding negative correlations and others 
finding no significant relationship. Our research aims to clarify these conflicting findings.

Literature Review:
Smith (2020) found that students who spend more than 3 hours daily on social media have lower 
grades. Jones et al. (2019) argued that the type of social media use matters more than duration. 
Educational platforms like LinkedIn showed different patterns than entertainment-focused platforms 
like TikTok. However, these studies used different methodologies, making comparisons difficult.

Methodology:
We surveyed 200 undergraduate students about their social media habits and collected their GPA 
data. Additionally, we conducted 20 in-depth interviews to understand usage patterns. The survey 
included questions about daily usage time, platform preferences, and purposes of use.

Results:
Students using social media for more than 4 hours daily had an average GPA of 2.8, compared to 
3.2 for those using it less than 2 hours. However, students who used educational features had 
higher engagement scores in group projects.

Discussion:
Our findings suggest a nuanced relationship between social media and academic performance. While 
excessive recreational use appears detrimental, educational applications show promise. Universities 
should consider developing guidelines that encourage productive social media use.

Conclusion:
Social media's impact on academic performance depends on how it's used. Future research should 
explore interventions that promote beneficial usage patterns while minimizing distractions.
"""

async def test_enhanced_critique():
    """Test the enhanced PACT critique system."""
    
    print("=" * 60)
    print("TESTING ENHANCED PACT CRITIQUE SYSTEM")
    print("=" * 60)
    
    try:
        # Import the enhanced components
        from src.pact.pact_critique_agent import pact_critique_agent
        from src.pact.visualization import create_spider_chart, generate_html_with_charts
        from src.pact.enhanced_pdf_generator import EnhancedPACTReportGenerator
        from langchain_core.messages import HumanMessage
        
        print("\n1. Running PACT critique analysis...")
        
        # Run the critique
        result = await pact_critique_agent.ainvoke(
            {"messages": [HumanMessage(content=TEST_PAPER)]}
        )
        
        # Extract scores
        print("\n2. Dimension Scores:")
        print("-" * 40)
        
        dimension_scores = {}
        for dim_id, critique in result.get('dimension_critiques', {}).items():
            score = critique.get('dimension_score', 0)
            name = critique.get('dimension_name', dim_id)
            assessment = critique.get('overall_assessment', 'N/A')
            dimension_scores[dim_id] = score
            print(f"   {name}: {score:.1f}/100 - {assessment}")
        
        overall_score = result.get('overall_score', 0)
        print(f"\n   OVERALL SCORE: {overall_score:.1f}/100")
        
        # Test spider chart generation
        print("\n3. Generating spider chart visualization...")
        try:
            spider_chart = create_spider_chart(dimension_scores)
            print("   ✓ Spider chart generated successfully")
        except ImportError as e:
            print(f"   ⚠ Spider chart generation skipped (missing dependency: {e})")
        
        # Test HTML report generation
        print("\n4. Generating HTML report with visualizations...")
        try:
            html_report = generate_html_with_charts(result)
            html_path = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            with open(html_path, 'w') as f:
                f.write(html_report)
            print(f"   ✓ HTML report saved to: {html_path}")
        except Exception as e:
            print(f"   ⚠ HTML generation failed: {e}")
        
        # Test PDF generation
        print("\n5. Generating comprehensive PDF report...")
        try:
            pdf_gen = EnhancedPACTReportGenerator()
            pdf_path = pdf_gen.generate_comprehensive_report(
                result,
                output_path=f"test_pact_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            )
            print(f"   ✓ PDF report saved to: {pdf_path}")
        except Exception as e:
            print(f"   ⚠ PDF generation failed: {e}")
        
        # Show sample of detailed analysis
        print("\n6. Sample Detailed Analysis:")
        print("-" * 40)
        
        # Get first dimension's analysis
        first_dim = list(result.get('dimension_critiques', {}).values())[0]
        if first_dim:
            print(f"\nDimension: {first_dim.get('dimension_name', 'Unknown')}")
            
            # Show issues if available
            issues = first_dim.get('issues', [])
            if issues:
                print("\nTop Issues Found:")
                for i, issue in enumerate(issues[:3], 1):
                    print(f"\n{i}. {issue.get('title', 'Issue')}")
                    print(f"   Priority: {issue.get('priority', 'Standard')}")
                    print(f"   Why it matters: {issue.get('why_it_matters', 'N/A')[:100]}...")
                    if issue.get('rewrite'):
                        print(f"   Suggestion: {issue.get('rewrite', '')[:100]}...")
            
            # Show strengths
            strengths = first_dim.get('key_strengths', [])
            if strengths:
                print("\nKey Strengths:")
                for strength in strengths[:3]:
                    print(f"   • {strength}")
        
        # Save full result for inspection
        result_path = f"test_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(result_path, 'w') as f:
            # Convert to JSON-serializable format
            json_result = {
                'overall_score': result.get('overall_score'),
                'dimension_critiques': result.get('dimension_critiques', {}),
                'final_critique': result.get('final_critique', ''),
                'priority_improvements': result.get('priority_improvements', [])
            }
            json.dump(json_result, f, indent=2, default=str)
        print(f"\n7. Full results saved to: {result_path}")
        
        print("\n" + "=" * 60)
        print("TEST COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("\nMake sure to run the notebook cells to generate source files:")
        print("  1. Open PACT3.0/notebooks/pact_critique_agent.ipynb")
        print("  2. Run all cells to generate the source files")
        print("  3. Try this test again")
        
    except Exception as e:
        print(f"\n❌ Test Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_enhanced_critique())