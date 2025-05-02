from typing import Dict, List
import json
import sys
import os
from research.config import WORKFLOW_CONFIG

# Add parent directory to Python path
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_dir)

from agents import create_section_reflector_agent, create_general_reflector_agent
from state import PlanningState, Section, ReflectionResult, Evaluation, EnhancementQuery, Summary

def reflect_on_section(section: Section, state: PlanningState) -> ReflectionResult:
    """Reflect on a specific section and suggest improvements."""
    print(f"\n=== Starting Reflection for Section: {section.title} ===")
    print(f"Section description: {section.description}")
    
    # Get section citations
    section_citations = [
        citation for citation in state["citations"].values()
        if citation.section_title == section.title
    ]
    print(f"Found {len(section_citations)} citations for {section.title}")
    
    # Get section summary
    section_summary = next(
        (summary for summary in state["summaries"] if summary.section_title == section.title),
        Summary(section_title=section.title, summary="", citation_numbers=[])
    )
    print(f"Summary length: {len(section_summary.summary)} characters")
    
    # Create section reflector agent
    reflector = create_section_reflector_agent()
    
    # Prepare input for the agent
    input_data = {
        "main_question": state["query"],
        "section_title": section.title,
        "section_description": section.description,
        "section_content": section_summary.summary,
        "citations": json.dumps([{
            "text": citation.text,
            "source": citation.source,
            "url": citation.url,
            "section_title": citation.section_title
        } for citation in section_citations])
    }
    
    # Get reflection results
    print(f"Analyzing {section.title} for improvements...")
    result = reflector.invoke(input_data)
    print(f"Reflection complete for {section.title}")
    
    # Convert to ReflectionResult object
    reflection_result = ReflectionResult(
        section_title=section.title,
        evaluation=Evaluation(
            completeness=result["evaluation"]["completeness"],
            relevance=result["evaluation"]["relevance"],
            structure=result["evaluation"]["structure"],
            correctness=result["evaluation"]["correctness"]
        ),
        needs_enhancement=result["needs_enhancement"],
        enhancement_queries=[
            EnhancementQuery(
                section=section.title,
                query=query["query"],
                reason=query["reason"]
            ) for query in result["enhancement_queries"]
        ],
        suggestions=result["suggestions"]
    )
    
    print(f"Evaluation results for {section.title}:")
    print(f"- Completeness: {reflection_result.evaluation.completeness}")
    print(f"- Relevance: {reflection_result.evaluation.relevance}")
    print(f"- Structure: {reflection_result.evaluation.structure}")
    print(f"- Correctness: {reflection_result.evaluation.correctness}")
    print(f"Needs enhancement: {reflection_result.needs_enhancement}")
    print(f"Number of enhancement queries: {len(reflection_result.enhancement_queries)}")
    print(f"Number of suggestions: {len(reflection_result.suggestions)}")
    print(f"=== Completed Reflection for Section: {section.title} ===\n")
    
    return reflection_result

def reflect_on_all_sections(state: PlanningState) -> PlanningState:
    """Reflect on all sections in the research."""
    # Initialize reflection_results if not present
    if "reflection_results" not in state:
        state["reflection_results"] = []
    
    # Check if we've exceeded max iterations
    if state.get("reflection_count", 0) >= WORKFLOW_CONFIG["max_reflection_iterations"]:
        print(f"Maximum reflection iterations reached")
        state["needs_enhancement"] = False
        for reflection in state["reflection_results"]:
            reflection.needs_enhancement = False
        return state

    # Increment reflection iterations
    state["reflection_count"] = state.get("reflection_count", 0) + 1
    print(f"reflection_count iteration {state['reflection_count']}/{WORKFLOW_CONFIG['max_reflection_iterations']}")
    # Reflect on each section
    for section in state["sections"]:
        reflection_result = reflect_on_section(section, state)
        state["reflection_results"].append(reflection_result)
    
    # Update needs_enhancement based on reflection results
    state["needs_enhancement"] = any(r.needs_enhancement for r in state["reflection_results"])
    
    return state

def reflect_on_research(state: PlanningState) -> PlanningState:
    """Reflect on the entire research and suggest improvements."""
    print("\n=== Starting General Research Reflection ===")
    
    # Check if general reflection has already been done
    if "general_reflection" in state:
        print("General reflection already completed")
        state["general_reflection"].needs_enhancement = False
        return state
    
    # Do a general reflection on the entire research
    print("Creating general reflector agent...")
    general_reflector = create_general_reflector_agent()
    
    # Prepare input for the general reflector
    print("Preparing research data for reflection...")
    input_data = {
        "main_question": state["query"],
        "sections": json.dumps([
            {
                "title": section.title,
                "description": section.description,
                "summary": next(
                    (summary.summary for summary in state["summaries"] 
                     if summary.section_title == section.title),
                    ""
                )
            }
            for section in state["sections"]
        ]),
        "citations": json.dumps([{
            "text": citation.text,
            "source": citation.source,
            "url": citation.url,
            "section_title": citation.section_title
        } for citation in state["citations"].values()])
    }
    
    # Get general reflection results
    print("Analyzing overall research for improvements...")
    result = general_reflector.invoke(input_data)
    print("General reflection complete")
    
    # Convert to ReflectionResult object
    state["general_reflection"] = ReflectionResult(
        section_title="General Research",
        evaluation=Evaluation(
            completeness=result["evaluation"]["completeness"],
            relevance=result["evaluation"]["relevance"],
            structure=result["evaluation"]["structure"],
            correctness=result["evaluation"]["correctness"]
        ),
        needs_enhancement=result["needs_enhancement"],
        enhancement_queries=[
            EnhancementQuery(
                section="General Research",
                query=query["query"],
                reason=query["reason"]
            ) for query in result["enhancement_queries"]
        ],
        suggestions=result["suggestions"]
    )
    
    print("\nGeneral Research Evaluation Results:")
    print(f"- Completeness: {state['general_reflection'].evaluation.completeness}")
    print(f"- Relevance: {state['general_reflection'].evaluation.relevance}")
    print(f"- Structure: {state['general_reflection'].evaluation.structure}")
    print(f"- Correctness: {state['general_reflection'].evaluation.correctness}")
    print(f"Needs enhancement: {state['general_reflection'].needs_enhancement}")
    print(f"Number of enhancement queries: {len(state['general_reflection'].enhancement_queries)}")
    print(f"Number of suggestions: {len(state['general_reflection'].suggestions)}")
    print("=== Completed General Research Reflection ===\n")
    
    return state 