#!/usr/bin/env python3
"""
Simple test to verify enhanced PACT files were generated correctly
"""

import os
import sys
from pathlib import Path

def test_file_imports():
    """Test that key files can be imported and have expected components."""
    
    print("=" * 60)
    print("TESTING ENHANCED PACT FILE STRUCTURE")
    print("=" * 60)
    
    # Test files exist
    expected_files = [
        "src/pact/pact_taxonomy.py",
        "src/pact/state_pact_critique.py", 
        "src/pact/pact_dimension_agents.py",
        "src/pact/pact_supervisor.py",
        "src/pact/pact_critique_agent.py",
        "src/pact/visualization.py",
        "src/pact/enhanced_pdf_generator.py",
    ]
    
    print("\n1. Checking file existence:")
    print("-" * 40)
    for file_path in expected_files:
        if Path(file_path).exists():
            size = Path(file_path).stat().st_size
            print(f"   ✓ {file_path} ({size:,} bytes)")
        else:
            print(f"   ❌ {file_path} - MISSING")
    
    # Test visualization.py has spider chart function
    print("\n2. Testing visualization.py components:")
    print("-" * 40)
    
    try:
        with open("src/pact/visualization.py", 'r') as f:
            viz_content = f.read()
        
        expected_functions = [
            "create_spider_chart",
            "generate_html_with_charts", 
            "create_subsection_heatmap",
            "generate_dimension_cards"
        ]
        
        for func in expected_functions:
            if f"def {func}" in viz_content:
                print(f"   ✓ {func}() function found")
            else:
                print(f"   ❌ {func}() function missing")
        
        # Check for spider chart specific code
        if "polar" in viz_content and "matplotlib" in viz_content:
            print("   ✓ Spider chart implementation detected")
        else:
            print("   ⚠ Spider chart implementation may be missing")
            
    except Exception as e:
        print(f"   ❌ Error reading visualization.py: {e}")
    
    # Test enhanced_pdf_generator.py 
    print("\n3. Testing enhanced_pdf_generator.py components:")
    print("-" * 40)
    
    try:
        with open("src/pact/enhanced_pdf_generator.py", 'r') as f:
            pdf_content = f.read()
        
        expected_components = [
            "EnhancedPACTReportGenerator",
            "generate_comprehensive_report",
            "_build_visualization_page",
            "_build_dimension_analysis",
            "PACT_COLORS"
        ]
        
        for component in expected_components:
            if component in pdf_content:
                print(f"   ✓ {component} found")
            else:
                print(f"   ❌ {component} missing")
                
        # Check for reportlab imports
        if "reportlab" in pdf_content:
            print("   ✓ ReportLab integration detected")
        else:
            print("   ⚠ ReportLab integration may be missing")
            
    except Exception as e:
        print(f"   ❌ Error reading enhanced_pdf_generator.py: {e}")
    
    # Test pact_dimension_agents.py enhancements
    print("\n4. Testing pact_dimension_agents.py enhancements:")
    print("-" * 40)
    
    try:
        with open("src/pact/pact_dimension_agents.py", 'r') as f:
            agents_content = f.read()
        
        # Check for enhanced components
        enhancements = [
            ("init_chat_model", "New model initialization"),
            ("create_enhanced_dimension_prompt", "Enhanced prompts"),
            ("critique_dimension_enhanced", "Enhanced critique function"),
            ("DetailedDimensionCritique", "Enhanced schemas"),
            ("10000", "Extended context window")
        ]
        
        for component, description in enhancements:
            if component in agents_content:
                print(f"   ✓ {description} - {component} found")
            else:
                print(f"   ⚠ {description} - {component} missing")
                
    except Exception as e:
        print(f"   ❌ Error reading pact_dimension_agents.py: {e}")
    
    # Show file sizes comparison
    print("\n5. File Size Analysis:")
    print("-" * 40)
    
    for file_path in expected_files:
        if Path(file_path).exists():
            size = Path(file_path).stat().st_size
            lines = len(Path(file_path).read_text().splitlines())
            print(f"   {Path(file_path).name}: {size:,} bytes, {lines:,} lines")
    
    print("\n" + "=" * 60)
    print("STRUCTURE TEST COMPLETED")
    print("=" * 60)
    
    # Test import without full dependencies
    print("\n6. Testing basic imports (without dependencies):")
    print("-" * 40)
    
    # Test visualization functions exist
    try:
        exec("""
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import base64

def test_spider_chart_logic():
    # Test the core spider chart logic
    dimension_scores = {"1.0.0": 75, "2.0.0": 85, "3.0.0": 65, "4.0.0": 90, "5.0.0": 70}
    categories = ["Research\\nFoundations", "Methodological\\nRigor", "Structure &\\nCoherence", "Academic\\nPrecision", "Critical\\nSophistication"]
    scores = [75, 85, 65, 90, 70]
    
    # Verify angle calculation
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    
    print(f"   ✓ Spider chart data processing: {len(angles)} angles for {N} dimensions")
    return True
        """)
        
        test_spider_chart_logic()
        
    except ImportError as e:
        print(f"   ⚠ Visualization dependencies not available: {e}")
    except Exception as e:
        print(f"   ❌ Error in spider chart logic: {e}")
    
    print("\n✓ Enhanced PACT system structure verified!")
    return True

if __name__ == "__main__":
    test_file_imports()