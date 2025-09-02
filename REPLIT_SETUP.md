
# PACT 3.0 - Replit Setup Guide

## Quick Start

1. **Install Dependencies**: Click on the "Install Dependencies" workflow in the dropdown next to the Run button
2. **Set Environment Variables**: Go to the Secrets tab and add your `OPENAI_API_KEY`
3. **Run the Application**: Click the Run button to start the PACT server

## Required Environment Variables

Add these in the Secrets tab:
- `OPENAI_API_KEY`: Your OpenAI API key for ChatGPT-5

## Optional Environment Variables

- `OPENAI_MODEL`: Default is "chatgpt-5"
- `DEBUG`: Set to "false" for production
- `HOST`: Default is "0.0.0.0"
- `PORT`: Default is "5000"

## Access Points

Once running, you can access:
- **Main API**: Your repl's webview URL
- **API Documentation**: `/docs` endpoint
- **WebSocket**: `/api/critique/live/{session_id}` for real-time updates

## Features

- Academic paper analysis using Penn CLO Academic Critique Taxonomy (PACT)
- Multi-agent research system with LangGraph
- Professional PDF report generation
- Real-time WebSocket updates
- Comprehensive API with FastAPI

## Troubleshooting

If you encounter issues:
1. Make sure all dependencies are installed
2. Verify your OpenAI API key is set in Secrets
3. Check the console for error messages
4. Ensure you're using Python 3.12+

For more details, see the main README.md in the PACT3.0 directory.
