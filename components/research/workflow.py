import sys
import os
from typing import Dict, List, Annotated, TypedDict
from langgraph.graph import Graph, StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import json

# Add parent directory to Python path
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_dir)

from config import LLM_CONFIG, WORKFLOW_CONFIG
from state import PlanningState
from stages.validation import validate_question, should_retry
from stages.planning import plan_sections
from stages.search import generate_search_queries, perform_web_search
from stages.summarization import generate_summaries
from stages.citations import process_citations, update_citation_numbers
from stages.reflection import reflect_on_research, reflect_on_all_sections

def create_workflow_graph() -> Graph:
    """Create the workflow graph for the research process."""
    workflow = StateGraph(PlanningState)
    
    # Add nodes for each stage
    workflow.add_node("validate", validate_question)
    workflow.add_node("plan", plan_sections)
    workflow.add_node("generate_queries", generate_search_queries)
    workflow.add_node("search", perform_web_search)
    workflow.add_node("summarize", generate_summaries)
    workflow.add_node("section_reflect", reflect_on_all_sections)  # Section-specific reflection
    workflow.add_node("general_reflect", reflect_on_research)  # General research reflection
    
    # Add edges to define the workflow
    # workflow.add_edge("validate", "plan")
    workflow.add_edge("plan", "generate_queries")
    workflow.add_edge("generate_queries", "search")
    workflow.add_edge("search", "summarize")
    workflow.add_edge("summarize", "section_reflect")
    
    # Add conditional edges
    # Validation conditional edge
    workflow.add_conditional_edges(
        "validate",
        should_retry,
        {
            "validate_question": "validate",
            "plan_sections": "plan",
            "end": END
        }
    )
    
    # Section reflection conditional edge
    def section_reflect_condition(state: PlanningState) -> str:
        if state.get("needs_enhancement"):
            return "search"
        return "general_reflect"
    
    workflow.add_conditional_edges(
        "section_reflect",
        section_reflect_condition,
        {
            "search": "search",
            "general_reflect": "general_reflect"
        }
    )
    
    # General reflection conditional edge
    def general_reflect_condition(state: PlanningState) -> str:
        if state.get("general_reflection") and state["general_reflection"].needs_enhancement:
            return "search"
        return "end"
    
    workflow.add_conditional_edges(
        "general_reflect",
        general_reflect_condition,
        {
            "search": "search",
            "end": END
        }
    )
    
    # Set entry point
    workflow.set_entry_point("validate")
    
    return workflow.compile()

def main():
    """Main entry point for the research workflow"""
    # Get question from user
    print("\nWelcome to the Car Accident Research Assistant!")
    print("Please enter your question about car accidents:")
    # question = input("> ").strip()
    question = "what are the main causes for car accidents in Haifa?"
    
    # Create initial state
    initial_state = PlanningState(
        query=question,
        sections=[],
        search_queries=[],
        search_results=[],
        summaries=[],
        citations={},
        needs_enhancement=False,
        reflection_count=0,
        is_query_valid=False,
        query_retry_count=0
    )
    
    # Create and run the workflow
    graph = create_workflow_graph()
    final_state = graph.invoke(initial_state)
    
    # Print results
    print("\nResearch Results:")
    print("=" * 50)
    print(f"Main Question: {final_state['query']}")
    print("\nSections:")
    for section in final_state["sections"]:
        print(f"- {section.title}: {section.description}")
    
    print("\nSummaries:")
    for summary in final_state["summaries"]:
        print(f"\n{summary.section_title}:")
        print(summary.summary)
    
    print("\nCitations:")
    for number, citation in final_state["citations"].items():
        print(f"[{number}] {citation.text} - {citation.url}")

if __name__ == "__main__":
    main()
