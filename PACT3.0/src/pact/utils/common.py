
"""
PACT Utilities Module

This module provides utility functions for the PACT system including:
- Date/directory utilities
- Tavily search integration  
- Web content summarization
- LLM model initialization
"""

import os
from datetime import datetime
from pathlib import Path

# LangChain imports with fallback
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    # fallback for older setups
    from langchain_community.chat_models import ChatOpenAI  # noqa: F401

# Tavily search client
try:
    from tavily import TavilyClient
    tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
except ImportError:
    tavily_client = None

def init_chat_model(model: str, temperature: float | None = None, **kwargs):
    """Compatibility wrapper so the rest of your code doesn't change."""
    model_kwargs = {}
    if temperature is not None:
        model_kwargs['temperature'] = temperature
    model_kwargs.update(kwargs)
    return ChatOpenAI(model=model, **model_kwargs)

# Define a default summarization model
summarization_model = init_chat_model(model="gpt-4o-mini")

def get_today_str() -> str:
    """Get today's date as a formatted string."""
    return datetime.now().strftime("%Y-%m-%d")

def get_current_dir() -> str:
    """Get the current directory path."""
    return str(Path.cwd())

def tavily_search_multiple(queries: list[str], max_results: int = 3) -> list[dict]:
    """Search multiple queries using Tavily."""
    if not tavily_client:
        return []
    
    all_results = []
    for query in queries:
        try:
            results = tavily_client.search(query=query, max_results=max_results)
            all_results.extend(results.get('results', []))
        except Exception as e:
            print(f"Error searching for '{query}': {e}")
    
    return all_results

def summarize_webpage_content(content: str, max_length: int = 500) -> str:
    """Summarize webpage content using the summarization model."""
    if not content or len(content) < max_length:
        return content
    
    try:
        from langchain_core.messages import HumanMessage
        prompt = f"Summarize this content in {max_length} characters or less:\n\n{content}"
        response = summarization_model.invoke([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        print(f"Error summarizing content: {e}")
        return content[:max_length] + "..."

def deduplicate_search_results(results: list[dict]) -> list[dict]:
    """Remove duplicate search results based on URL."""
    seen_urls = set()
    unique_results = []
    
    for result in results:
        url = result.get('url', '')
        if url not in seen_urls:
            seen_urls.add(url)
            unique_results.append(result)
    
    return unique_results

def process_search_results(results: list[dict]) -> list[dict]:
    """Process and clean search results."""
    processed = []
    
    for result in results:
        processed_result = {
            'title': result.get('title', 'No title'),
            'url': result.get('url', ''),
            'content': summarize_webpage_content(result.get('content', '')),
            'score': result.get('score', 0)
        }
        processed.append(processed_result)
    
    return processed

def format_search_output(results: list[dict]) -> str:
    """Format search results for display."""
    if not results:
        return "No search results found."
    
    formatted = []
    for i, result in enumerate(results, 1):
        formatted.append(f"{i}. **{result['title']}**")
        formatted.append(f"   URL: {result['url']}")
        formatted.append(f"   Content: {result['content']}")
        formatted.append("")
    
    return "\n".join(formatted)

def tavily_search(query: str, max_results: int = 5) -> str:
    """Perform a Tavily search and return formatted results."""
    if not tavily_client:
        return "Tavily search not available - API key not configured."
    
    try:
        results = tavily_client.search(query=query, max_results=max_results)
        search_results = results.get('results', [])
        
        # Process and deduplicate results
        processed_results = process_search_results(search_results)
        unique_results = deduplicate_search_results(processed_results)
        
        return format_search_output(unique_results)
    
    except Exception as e:
        return f"Error performing search: {e}"

def think_tool(thinking: str) -> str:
    """A tool for the agent to think through problems."""
    return f"Thinking: {thinking}"
