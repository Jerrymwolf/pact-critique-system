#!/usr/bin/env python3
"""
Integration test for PACT Critique System

This script tests the complete system end-to-end without requiring
the full server setup.
"""

import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Load environment variables
load_dotenv()

async def test_pact_critique():
    """Test the PACT critique system with a sample paper."""
    
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set. Please set it in .env file.")
        return False
    
    print("🔧 Testing PACT Critique System with ChatGPT 5...")
    
    try:
        # Import the critique agent
        from pact.pact_critique_agent import pact_critique_agent
        from langchain_core.messages import HumanMessage
        
        # Sample paper for testing
        sample_paper = """
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
        
        print("📝 Running critique on sample paper...")
        
        # Create messages
        messages = [HumanMessage(content=sample_paper)]
        
        # Configure the session
        config = {
            "configurable": {"thread_id": "test_session"},
            "recursion_limit": 20
        }
        
        # Run the critique
        print("⏳ Processing (this may take 1-2 minutes)...")
        result = await pact_critique_agent.ainvoke(
            {"messages": messages},
            config=config
        )
        
        # Check results
        if result.get('final_critique'):
            print("✅ Critique generated successfully!")
            print(f"📊 Overall Score: {result.get('overall_score', 'N/A')}")
            print("\n📋 Sample of generated critique:")
            print("-" * 50)
            critique_lines = result['final_critique'].split('\n')
            print('\n'.join(critique_lines[:10]) + '\n...')
            print("-" * 50)
            return True
        else:
            print("❌ No critique generated")
            return False
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        print(traceback.format_exc())
        return False

async def test_api_components():
    """Test individual API components."""
    
    print("\n🔧 Testing individual components...")
    
    try:
        # Test session manager
        from pact.session_manager import SessionManager
        session_manager = SessionManager("test_sessions")
        
        session_id = session_manager.create_session(
            paper_content="Test paper content",
            paper_title="Test Paper"
        )
        
        session = session_manager.get_session(session_id)
        if session and session.paper_content == "Test paper content":
            print("✅ Session manager working")
        else:
            print("❌ Session manager failed")
            return False
            
        # Test PACT taxonomy loading
        from pact.pact_taxonomy import load_pact_taxonomy
        pact_data = load_pact_taxonomy()
        
        if 'dimensions' in pact_data:
            print("✅ PACT taxonomy loaded")
        else:
            print("❌ PACT taxonomy loading failed")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Component test error: {e}")
        return False

if __name__ == "__main__":
    async def main():
        print("🚀 PACT Critique System Integration Test")
        print("=" * 50)
        
        # Test components
        components_ok = await test_api_components()
        
        if components_ok:
            # Test full system
            system_ok = await test_pact_critique()
            
            if system_ok:
                print("\n🎉 All tests passed! System is ready for use.")
                print("\n📝 To start the server:")
                print("   python start_server.py")
                print("\n🌐 Then access: http://localhost:8000")
            else:
                print("\n❌ System test failed")
        else:
            print("\n❌ Component tests failed")
    
    asyncio.run(main())