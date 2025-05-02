from typing import List, Dict, Tuple, Union
from state import Citation

def find_citation_url(citation: Union[str, Dict], search_results: List[Dict]) -> str:
    """Find the URL for a citation in search results"""
    # Handle dictionary citations
    if isinstance(citation, dict):
        if "url" in citation and citation["url"]:
            return citation["url"]
        citation_text = citation.get("text", "")
    else:
        # Handle string citations
        citation_text = citation
        if citation_text.startswith("http"):
            return citation_text
    
    # Search for the citation in search results
    for result in search_results:
        if citation_text in result.get("snippet", ""):
            return result.get("link", "")
    
    return None

def process_citations(citations: List[Union[str, Dict]], search_results: List[Dict], start_number: int, section_title: str) -> Tuple[Dict[str, int], Dict[int, Citation], int]:
    """Process citations and assign numbers"""
    processed_citations = {}
    citation_objects = {}
    current_number = start_number
    
    for citation in citations:
        url = find_citation_url(citation, search_results)
        if url:
            # Get citation text from either dictionary or string
            citation_text = citation["text"] if isinstance(citation, dict) else citation
            source = citation.get("source", "Web Search") if isinstance(citation, dict) else "Web Search"
            processed_citations[citation_text] = current_number
            citation_objects[current_number] = Citation(
                text=citation_text,
                source=source,
                url=url,
                section_title=section_title
            )
            current_number += 1
    
    return processed_citations, citation_objects, current_number

def update_citation_numbers(text: str, citations: Dict[str, int]) -> str:
    """Update text with citation numbers"""
    for citation, number in citations.items():
        text = text.replace(f"[{citation}]", f"[{number}]")
    return text 