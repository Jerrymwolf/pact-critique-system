#!/usr/bin/env python3
"""
Startup script for PACT Critique API Server

This script loads environment variables and starts the FastAPI server
with proper configuration for the PACT critique system.
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

# Check for required environment variables
required_vars = ["OPENAI_API_KEY"]
optional_vars = ["TAVILY_API_KEY"]
missing_vars = []

for var in required_vars:
    if not os.getenv(var):
        missing_vars.append(var)

if missing_vars:
    print("❌ Missing required environment variables:")
    for var in missing_vars:
        print(f"   - {var}")
    print("\nPlease set these in your .env file or environment.")
    print("See .env.example for reference.")
    sys.exit(1)

print("✅ Environment configured successfully")
print(f"📝 OpenAI Model: {os.getenv('OPENAI_MODEL', 'gpt-5')}")
print("🤖 Using GPT-5 (no custom temperature - uses model defaults)")

# Check optional services
tavily_key = os.getenv("TAVILY_API_KEY")
if tavily_key:
    print("🔍 Tavily search enabled for research agents")
else:
    print("⚠️  Tavily API key not found - web search functionality will be limited")

# Start the server
if __name__ == "__main__":
    import uvicorn
    from pact.api_server import app
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("DEBUG", "true").lower() == "true"
    
    print(f"🚀 Starting PACT Critique API Server on {host}:{port}")
    print(f"🔗 Access the app in the Replit webview or at your repl's URL")
    print("📊 WebSocket endpoint: /api/critique/live/{session_id}")
    print("📖 API documentation: /docs")
    
    uvicorn.run(
        "pact.api_server:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info"
    )