#!/usr/bin/env python3
"""
ChatGPT 5 Model Validation

This script validates that the ChatGPT 5 model configuration is correct
and tests a simple call to ensure the API works.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Load environment variables
load_dotenv()

async def validate_chatgpt5():
    """Validate ChatGPT 5 configuration and connectivity."""
    
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set. Please set it in .env file.")
        return False
    
    print("🔧 Validating ChatGPT 5 Configuration...")
    print(f"📝 Model: {os.getenv('OPENAI_MODEL', 'chatgpt-5')}")
    
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage
        
        # Initialize ChatGPT 5 model (no temperature setting)
        model = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "chatgpt-5"),
            max_tokens=100,  # Small test
            timeout=30
        )
        
        print("🤖 Testing ChatGPT 5 API call...")
        
        # Simple test message
        test_message = "Please respond with exactly 'ChatGPT 5 is working' to confirm you're operational."
        
        response = await model.ainvoke([HumanMessage(content=test_message)])
        
        if response and hasattr(response, 'content'):
            print(f"✅ ChatGPT 5 Response: {response.content}")
            print("✅ ChatGPT 5 is properly configured and responsive!")
            return True
        else:
            print("❌ Unexpected response format from ChatGPT 5")
            return False
            
    except Exception as e:
        print(f"❌ Error testing ChatGPT 5: {e}")
        
        # Check for common issues
        error_str = str(e).lower()
        if "model" in error_str and "not found" in error_str:
            print("\n💡 Suggestion: The model identifier might be incorrect.")
            print("   Try these alternatives:")
            print("   - gpt-5")
            print("   - gpt-5-turbo") 
            print("   - chatgpt-5-turbo")
            print("\n   Update OPENAI_MODEL in your .env file with the correct identifier.")
        elif "api_key" in error_str or "authentication" in error_str:
            print("\n💡 Suggestion: Check your OpenAI API key.")
            print("   Make sure you have access to ChatGPT 5 models.")
        elif "rate limit" in error_str:
            print("\n💡 Suggestion: Rate limit reached. Wait a moment and try again.")
        else:
            print(f"\n💡 Full error details: {e}")
        
        return False

if __name__ == "__main__":
    import asyncio
    
    async def main():
        print("🚀 ChatGPT 5 Validation Script")
        print("=" * 40)
        
        is_valid = await validate_chatgpt5()
        
        if is_valid:
            print("\n🎉 ChatGPT 5 is ready for PACT critique system!")
            print("\n📝 Next steps:")
            print("   python start_server.py")
        else:
            print("\n❌ ChatGPT 5 validation failed.")
            print("\n📝 Please check:")
            print("   1. Your OpenAI API key is valid")
            print("   2. You have access to ChatGPT 5 models") 
            print("   3. The model identifier is correct")
    
    asyncio.run(main())