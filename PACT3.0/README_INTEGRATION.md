# PACT 3.0 - ChatGPT 5 Integration Setup Guide

This document provides setup instructions for the ChatGPT 5 integrated PACT critique system.

## 🚀 Quick Start

### 1. Environment Setup

1. **Copy the environment template:**
   ```bash
   cp .env.example .env
   ```

2. **Set your OpenAI API key in `.env`:**
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   OPENAI_MODEL=chatgpt-5
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### 2. Test the System

Run the integration test to verify everything is working:

```bash
python test_integration.py
```

This will:
- Test individual components
- Run a full critique on a sample paper
- Verify GPT-5 integration

### 3. Start the Server

```bash
python start_server.py
```

The server will start at `http://localhost:8000`

## 🏗️ System Architecture

```
Frontend (HTML/JS) ←→ FastAPI Backend ←→ LangGraph Agents ←→ ChatGPT 5
     ↓                      ↓                    ↓              ↓
WebSocket Updates    Session Management    Agent Coordination  AI Analysis
Progress Tracking    File Processing      PACT Taxonomy      Structured Output
Results Display      Error Handling       Parallel Execution  Rate Limiting
```

## 🤖 Agent System

### 5 Specialized PACT Agents:
1. **Research Foundations Agent** - Problem definition, frameworks, literature
2. **Methodological Rigor Agent** - Methods, data, analysis, limitations  
3. **Structure & Coherence Agent** - Organization, flow, transitions
4. **Academic Precision Agent** - Terms, citations, grammar, formatting
5. **Critical Sophistication Agent** - Reflexivity, originality, theoretical depth

### Supervisor Agent:
- Plans critique approach based on paper content
- Coordinates parallel agent execution
- Synthesizes final comprehensive critique

## 📡 API Endpoints

- `POST /api/critique/start` - Submit paper for analysis
- `GET /api/critique/status/{session_id}` - Get current progress
- `GET /api/critique/results/{session_id}` - Get final results
- `WebSocket /api/critique/live/{session_id}` - Real-time updates

## 🔧 Configuration Options

Environment variables in `.env`:

```bash
# Core Configuration
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=chatgpt-5

# Server Configuration  
HOST=0.0.0.0
PORT=8000
DEBUG=true

# Session Management
MAX_SESSION_AGE_HOURS=24

# Note: ChatGPT 5 doesn't support custom temperature settings
```

## 📊 Real-Time Progress Tracking

The system provides real-time updates via:
- **WebSocket connections** for instant updates
- **Progress polling** as fallback
- **Agent status tracking** with individual progress
- **Error handling** with retry logic

## 🎯 Features

### ✅ Completed Integration:
- [x] ChatGPT 5 integration with all agents
- [x] Real-time WebSocket progress tracking
- [x] Structured critique output with PACT taxonomy
- [x] Session management with persistence
- [x] Parallel agent execution
- [x] Error handling and retry logic
- [x] File upload support (planned)
- [x] Multiple output formats (planned)

### 🔄 Real-Time UI Updates:
- Agent status indicators (Waiting → Active → Complete)
- Progress bar with percentage
- Live agent messages
- Animated transitions
- Error state handling

## 🚨 Troubleshooting

### Common Issues:

1. **"Missing OPENAI_API_KEY"**
   - Make sure you've set the API key in your `.env` file
   - Verify the `.env` file is in the root directory

2. **"Import errors"**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version (3.8+ required)

3. **"WebSocket connection failed"**
   - This is normal - the system falls back to polling
   - Check browser console for specific errors

4. **"Analysis failed"**
   - Check API key is valid and has sufficient credits
   - Verify internet connection
   - Check server logs for detailed error messages

### Development Mode:

Start with debug logging:
```bash
DEBUG=true LOG_LEVEL=debug python start_server.py
```

## 🔒 Security Notes

- API keys are loaded from environment variables only
- Sessions are isolated with unique IDs
- File uploads are validated and limited
- Rate limiting prevents abuse
- Sessions auto-expire after 24 hours

## 📈 Performance

- **Parallel agent execution** reduces total analysis time
- **Streaming responses** via WebSocket for immediate feedback  
- **Session caching** avoids redundant processing
- **Background processing** keeps UI responsive

## 🤝 Contributing

The system is modular and extensible:
- Add new PACT dimensions by creating additional agents
- Modify critique prompts in the agent files
- Extend output formats in the API responses
- Add new file types in the upload handler

## 📖 Usage Examples

### Basic Paper Submission:
1. Paste your paper content in the text area
2. Select paper type (research, thesis, etc.)
3. Click "Start PACT Analysis"
4. Watch real-time progress as agents work
5. Review detailed critique results

### Advanced Features:
- Download critique reports in multiple formats
- Session persistence allows returning to results
- WebSocket connections provide instant updates
- Detailed PACT dimension scoring with explanations