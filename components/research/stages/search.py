from typing import List, Dict
import json
import time
from duckduckgo_search import DDGS
from agents import create_query_generator_agent
from config import SEARCH_CONFIG, SECTION_CONFIG, WORKFLOW_CONFIG
from state import PlanningState, SearchQuery, SearchResult

# Initialize the search tool
search = DDGS()

def search_single_query(query: str, max_results: int) -> List[Dict]:
    """Perform a single search query with rate limiting"""
    try:
        results = []
        for r in search.text(query, max_results=max_results):
            results.append(r)
            time.sleep(SEARCH_CONFIG["delay"])
        return results
    except Exception as e:
        return [f"Search failed: {str(e)}"]

def search_section_queries(section_title: str, queries: List[Dict]) -> List[Dict]:
    """Search all queries for a section and return results"""
    results = []
    for query in queries:
        search_results = search_single_query(query["query"], SEARCH_CONFIG["max_results"])
        results.append({
            "section_title": section_title,
            "query": query["query"],
            "results": search_results
        })
    return results

def generate_search_queries(state: PlanningState) -> PlanningState:
    """Generate search queries for each section."""
    state["search_queries"] = []
    
    # Filter out conclusion sections
    non_summary_sections = [section for section in state["sections"] 
                          if section.title.lower() != SECTION_CONFIG["conclusion_section"].lower()]
    
    if non_summary_sections:
        query_generator = create_query_generator_agent()
        
        # Generate queries for each section
        for section in non_summary_sections:
            # Convert section to dictionary format
            section_dict = {
                "title": section.title,
                "description": section.description
            }
            
            result = query_generator.invoke({
                "main_question": state["query"],
                "sections": json.dumps(section_dict),
                "queries_per_section": WORKFLOW_CONFIG["queries_per_section"]
            })
            
            if "search_queries" in result:
                state["search_queries"].extend(result["search_queries"])
    
    return state

def perform_initial_web_search(state: PlanningState) -> PlanningState:
    """Perform initial web searches for all section queries."""
    # Group queries by section
    section_queries = {}
    for query in state["search_queries"]:
        if query["section_title"] not in section_queries:
            section_queries[query["section_title"]] = []
        section_queries[query["section_title"]].append(query)
    
    # Process each section's queries
    for section_title, queries in section_queries.items():
        print(f"\nSearching for section: {section_title}")
        section_results = search_section_queries(section_title, queries)
        state["search_results"].extend(section_results)
    
    return state

def perform_section_enhancement_search(state: PlanningState) -> PlanningState:
    """Perform web searches for section enhancement queries."""
    # Get enhancement queries from reflection results
    enhancement_queries = []
    for result in state["reflection_results"]:
        if result.needs_enhancement:
            enhancement_queries.extend(result.enhancement_queries)
    
    # Process enhancement queries
    for query in enhancement_queries:
        print(f"\nSearching for enhancement query: {query.query}")
        search_results = search_single_query(query.query, SEARCH_CONFIG["max_results"])
        state["search_results"].append({
            "section_title": query.section,
            "query": query.query,
            "results": search_results
        })
    
    return state

def perform_general_enhancement_search(state: PlanningState) -> PlanningState:
    """Perform web searches for general enhancement queries."""
    # Get enhancement queries from general reflection
    enhancement_queries = state["general_reflection"].enhancement_queries
    
    # Process enhancement queries
    for query in enhancement_queries:
        print(f"\nSearching for general enhancement query: {query.query}")
        search_results = search_single_query(query.query, SEARCH_CONFIG["max_results"])
        state["search_results"].append({
            "section_title": query.section,
            "query": query.query,
            "results": search_results
        })
    
    return state

def perform_web_search(state: PlanningState) -> PlanningState:
    """Route to appropriate search function based on state."""
    if state.get("reflection_results") and state.get("needs_enhancement", False):
        return perform_section_enhancement_search(state)
    elif "general_reflection" in state and state.get("general_reflection") and state.get("general_reflection").needs_enhancement:
        return perform_general_enhancement_search(state)
    else:
        return perform_initial_web_search(state) 