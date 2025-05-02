from typing import List, Dict, Tuple
import json
from agents import create_summary_agent, create_conclusion_agent, create_enhanced_summary_agent
from research.config import SECTION_CONFIG, SEARCH_CONFIG
from state import PlanningState, Section, Summary, Citation
from .citations import process_citations, update_citation_numbers
from .search import search_single_query

def summarize_section(section: Section, state: PlanningState, citation_counter: int) -> Tuple[str, List[Dict], int]:
    """Generate summary for a single section and process its citations."""
    print(f"\nGenerating summary for section: {section.title}")
    
    # Get search results for this section
    matching_results = next(
        result for result in state["search_results"]
        if result["section_title"] == section.title
    )
    
    # Generate summary
    print(f"Processing search results for {section.title}...")
    summary_agent = create_summary_agent()
    result = summary_agent.invoke({
        "section_title": section.title,
        "section_description": section.description,
        "search_results": json.dumps(matching_results["results"])
    })
    
    # Process citations
    print(f"Processing citations for {section.title}...")
    processed_citations, citation_objects, citation_counter = process_citations(
        result["citations"],
        matching_results["results"],
        citation_counter,
        section.title
    )
    
    # Update summary text with citation numbers
    summary_text = update_citation_numbers(result["summary"], processed_citations)
    print(f"Summary generated for {section.title}")
    
    return summary_text, citation_objects, citation_counter

def summarize_conclusion_section(state: PlanningState, citation_counter: int) -> Tuple[str, List[Dict], int]:
    """Generate summary for the conclusion section."""
    print("\nGenerating conclusion section...")
    
    # Get all section summaries
    section_summaries = [
        section.summary for section in state["sections"]
        if section.title.lower() != SECTION_CONFIG["conclusion_section"].lower()
    ]
    
    # Generate conclusion
    print("Processing section summaries for conclusion...")
    conclusion_agent = create_conclusion_agent()
    result = conclusion_agent.invoke({
        "main_question": state["query"],
        "section_summaries": json.dumps(section_summaries)
    })
    
    # Process citations
    print("Processing citations for conclusion...")
    processed_citations, citation_objects, citation_counter = process_citations(
        result["citations"],
        [],  # No search results needed for conclusion
        citation_counter,
        SECTION_CONFIG["conclusion_section"]
    )
    
    # Update conclusion text with citation numbers
    conclusion_text = update_citation_numbers(result["summary"], processed_citations)
    print("Conclusion generated")
    
    return conclusion_text, citation_objects, citation_counter

def generate_web_summaries(state: PlanningState) -> PlanningState:
    """Generate summaries from web search results."""
    citation_counter = 1
    
    # Process each section (except conclusion)
    for section in state["sections"]:
        if section.title.lower() != SECTION_CONFIG["conclusion_section"].lower():
            summary, new_citations, citation_counter = summarize_section(
                section, state, citation_counter
            )
            section.summary = summary
            state["citations"].update(new_citations)
    
    return state

def generate_conclusion(state: PlanningState) -> PlanningState:
    """Generate conclusion based on previous summaries."""
    citation_counter = len(state["citations"]) + 1
    
    # Find conclusion section
    conclusion_section = next(
        section for section in state["sections"]
        if section.title.lower() == SECTION_CONFIG["conclusion_section"].lower()
    )
    
    # Generate conclusion
    conclusion_summary, new_citations, _ = summarize_conclusion_section(state, citation_counter)
    conclusion_section.summary = conclusion_summary
    state["citations"].update(new_citations)
    
    return state

def enhance_section_summary(section: Section, state: PlanningState, citation_counter: int) -> Tuple[str, List[Dict], int]:
    """Enhance an existing section summary with new information and reflection insights."""
    # Get reflection result for this section
    reflection_result = next(
        result for result in state["reflection_results"]
        if result.section_title == section.title
    )
    
    # Get new search results for this section
    matching_results = next(
        result for result in state["search_results"]
        if result["section_title"] == section.title
    )
    
    # Generate enhanced summary
    enhanced_summary_agent = create_enhanced_summary_agent()
    result = enhanced_summary_agent.invoke({
        "section_title": section.title,
        "section_description": section.description,
        "current_summary": section.summary,
        "evaluation": json.dumps({
            "completeness": reflection_result.evaluation.completeness,
            "relevance": reflection_result.evaluation.relevance,
            "structure": reflection_result.evaluation.structure,
            "correctness": reflection_result.evaluation.correctness
        }),
        "suggestions": json.dumps(reflection_result.suggestions),
        "new_search_results": json.dumps(matching_results["results"])
    })
    
    # Process citations
    processed_citations, citation_objects, citation_counter = process_citations(
        result["citations"],
        matching_results["results"],
        citation_counter,
        section.title
    )
    
    # Update summary text with citation numbers
    summary_text = update_citation_numbers(result["summary"], processed_citations)
    
    return summary_text, citation_objects, citation_counter

def enhance_summaries(state: PlanningState) -> PlanningState:
    """Enhance existing summaries based on reflection results and new search results."""
    citation_counter = len(state["citations"]) + 1
    
    # Enhance sections that need improvement
    for section in state["sections"]:
        if section.title.lower() != SECTION_CONFIG["conclusion_section"].lower():
            reflection_result = next(
                (result for result in state["reflection_results"] 
                 if result.section_title == section.title),
                None
            )
            if reflection_result and reflection_result.needs_enhancement:
                summary, new_citations, citation_counter = enhance_section_summary(
                    section, state, citation_counter
                )
                section.summary = summary
                state["citations"].update(new_citations)
       
    return state

def generate_initial_summaries(state: PlanningState) -> PlanningState:
    """Generate initial summaries for all sections."""
    print("\n=== Starting Initial Summary Generation ===")
    for section in state["sections"]:       
        print(f"\nProcessing section: {section.title}")
        print(f"Description: {section.description}")
        
        # Get search results for this section
        matching_results = {
            "section_title": section.title,
            "results": [
                result for result in state["search_results"]
                if result["section_title"] == section.title
            ]
        }
        print(f"Found {len(matching_results['results'])} search results for {section.title}")
        
        # Generate summary
        print(f"Generating summary for {section.title}...")
        summary_agent = create_summary_agent()
        result = summary_agent.invoke({
            "section_title": section.title,
            "section_description": section.description,
            "search_results": json.dumps(matching_results["results"])
        })
        
        # Create new summary and add citations
        new_summary = Summary(
            section_title=section.title,
            summary=result["summary"],
            citation_numbers=[]
        )
        print(f"Summary generated for {section.title}")
        
        # Add citations
        print(f"Processing citations for {section.title}...")
        for citation in result["citations"]:
            citation["section_title"] = section.title
            state["citations"][len(state["citations"]) + 1] = Citation(**citation)
            new_summary.citation_numbers.append(len(state["citations"]))
            print(f"Added citation [{len(state['citations'])}] for {section.title}")
        
        # Add summary to state
        state["summaries"].append(new_summary)
        print(f"Completed processing for {section.title}")
    
    print("\n=== Initial Summary Generation Complete ===")
    return state

def enhance_section_summaries(state: PlanningState) -> PlanningState:
    """Enhance summaries for sections that need improvement."""
    print("\n=== Starting Section Summary Enhancement ===")
    for reflection in state["reflection_results"]:
        if not reflection.needs_enhancement:
            print(f"\nSkipping enhancement for {reflection.section_title} - no enhancement needed")
            continue
            
        print(f"\nEnhancing section: {reflection.section_title}")
        print(f"Evaluation: {reflection.evaluation}")
        print(f"Suggestions: {reflection.suggestions}")
        
        # Find the section to enhance
        section = next(s for s in state["sections"] if s.title == reflection.section_title)
        
        # Get current summary
        current_summary = next(
            summary for summary in state["summaries"]
            if summary.section_title == section.title
        )
        print(f"Current summary length: {len(current_summary.summary)} characters")
        
        # Get new search results for this section
        matching_results = {
            "section_title": section.title,
            "results": [
                result for result in state["search_results"]
                if result.get("section_title") == section.title
            ]
        }
        print(f"Found {len(matching_results['results'])} new search results for enhancement")
        
        # Generate enhanced summary
        print(f"Generating enhanced summary for {section.title}...")
        enhanced_summary_agent = create_enhanced_summary_agent()
        result = enhanced_summary_agent.invoke({
            "section_title": section.title,
            "section_description": section.description,
            "current_summary": current_summary.summary,
            "evaluation": reflection.evaluation,
            "suggestions": reflection.suggestions,
            "new_search_results": json.dumps(matching_results["results"])
        })
        
        # Update summary and add new citations
        current_summary.summary = result["summary"]
        print(f"Enhanced summary length: {len(result['summary'])} characters")
        
        print(f"Processing new citations for {section.title}...")
        for citation in result["citations"]:
            citation["section_title"] = section.title
            state["citations"][len(state["citations"]) + 1] = Citation(**citation)
            current_summary.citation_numbers.append(len(state["citations"]))
            print(f"Added new citation [{len(state['citations'])}] for {section.title}")
        
        print(f"Completed enhancement for {section.title}")
    
    print("\n=== Section Summary Enhancement Complete ===")
    return state

def enhance_research(state: PlanningState) -> PlanningState:
    """Enhance the research based on general reflection results."""
    print("\n=== Starting General Research Enhancement ===")
    
    # Get general reflection results
    general_reflection = state.get("general_reflection")
    if not general_reflection or not general_reflection.needs_enhancement:
        print("No general enhancement needed")
        return state
    
    print("Processing enhancement queries from general reflection...")
    for query in general_reflection.enhancement_queries:
        print(f"\nEnhancing research with query: {query.query}")
        print(f"Reason: {query.reason}")
            
        # Search for new information
        print("Searching for new information...")
        search_results = search_single_query(query.query, SEARCH_CONFIG["max_results"])
        if not search_results:
            print("No new search results found")
            continue
            
        print(f"Found {len(search_results)} new search results")
        
        # Get current summaries
        current_summaries = "\n\n".join(
            f"Section: {summary.section_title}\n{summary.summary}"
            for summary in state["summaries"]
        )
        
        # Create enhanced summary
        print("Creating enhanced summary...")
        enhanced_summary = create_enhanced_summary_agent().invoke({
            "section_title": "General Research",
            "section_description": "Enhancing the entire research",
            "current_summary": current_summaries,
            "evaluation": json.dumps(general_reflection.evaluation.__dict__),
            "suggestions": json.dumps(general_reflection.suggestions),
            "new_search_results": json.dumps([{
                "title": result["title"],
                "snippet": result["body"],
                "url": result["href"]
            } for result in search_results])
        })
        
        # Update citations
        print("Adding new citations...")
        for citation in enhanced_summary["citations"]:
            state["citations"][len(state["citations"]) + 1] = Citation(
                text=citation["text"],
                source=citation["source"],
                url=citation["url"],
                section_title="General Research"
            )
        
        print(f"Added {len(enhanced_summary['citations'])} new citations")
    
    print("=== Completed General Research Enhancement ===\n")
    return state

def generate_summaries(state: PlanningState) -> PlanningState:
    """Route to appropriate summary generation function based on state."""
    if state.get("reflection_results") and state.get("needs_enhancement", False):
        return enhance_section_summaries(state)
    elif "general_reflection" in state and state.get("general_reflection") and state.get("general_reflection").needs_enhancement:
        return enhance_research(state)
    else:
        return generate_initial_summaries(state) 