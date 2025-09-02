"""
PACT-Based Multi-Agent Paper Critique System

This module integrates all components of the PACT critique system:
- Input processing and paper parsing
- Supervisor planning and coordination
- Parallel evaluation by specialized dimension agents
- Synthesis of feedback into comprehensive critique
"""

from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage

from .state_pact_critique import PaperCritiqueState
from .pact_supervisor import plan_critique, synthesize_critique
from .pact_dimension_agents import (
    critique_research_foundations,
    critique_methodological_rigor,
    critique_structure_coherence,
    critique_academic_precision,
    critique_critical_sophistication
)

# ===== INPUT PROCESSING =====

def process_paper_input(state: MessagesState) -> dict:
    """
    Process the input paper from user messages.
    
    Extracts the paper content and any metadata provided.
    """
    # Get the last user message which should contain the paper
    messages = state.get('messages', [])
    if not messages:
        raise ValueError("No paper provided for critique")
    
    # Extract paper content from the last message
    last_message = messages[-1]
    if isinstance(last_message, HumanMessage):
        paper_content = last_message.content
    else:
        paper_content = str(last_message)
    
    # Extract title if provided in a specific format
    paper_title = None
    if paper_content.startswith("Title:"):
        lines = paper_content.split('\n')
        paper_title = lines[0].replace("Title:", "").strip()
    
    return {
        "paper_content": paper_content,
        "paper_title": paper_title
    }

# ===== PARALLEL DIMENSION EVALUATION =====

async def evaluate_dimensions(state: PaperCritiqueState) -> dict:
    """
    Coordinate parallel evaluation of all PACT dimensions.
    
    This node spawns all dimension agents to work in parallel.
    """
    # Run all dimension critiques in parallel
    import asyncio
    
    tasks = [
        critique_research_foundations(state),
        critique_methodological_rigor(state),
        critique_structure_coherence(state),
        critique_academic_precision(state),
        critique_critical_sophistication(state)
    ]
    
    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks)
    
    # Combine all dimension critiques
    combined_critiques = {}
    for result in results:
        if 'dimension_critiques' in result:
            combined_critiques.update(result['dimension_critiques'])
    
    return {"dimension_critiques": combined_critiques}

# ===== GRAPH CONSTRUCTION =====

def build_pact_critique_graph():
    """
    Build the complete PACT critique workflow graph.
    """
    # Create the workflow
    workflow = StateGraph(PaperCritiqueState)
    
    # Add nodes
    workflow.add_node("process_input", process_paper_input)
    workflow.add_node("plan_critique", plan_critique)
    workflow.add_node("evaluate_dimensions", evaluate_dimensions)
    workflow.add_node("synthesize_critique", synthesize_critique)
    
    # Add edges
    workflow.add_edge(START, "process_input")
    workflow.add_edge("process_input", "plan_critique")
    # plan_critique uses Command to go to evaluate_dimensions
    workflow.add_edge("evaluate_dimensions", "synthesize_critique")
    workflow.add_edge("synthesize_critique", END)
    
    return workflow

# Compile the workflow
from langgraph.checkpoint.memory import MemorySaver

pact_critique_builder = build_pact_critique_graph()
memory = MemorySaver()
pact_critique_agent = pact_critique_builder.compile(checkpointer=memory)