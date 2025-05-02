from typing import TypedDict, List, Dict
from dataclasses import dataclass
import sys
import os

# Add the current directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

@dataclass
class Section:
    title: str
    description: str

@dataclass
class SearchQuery:
    section_title: str
    query: str

@dataclass
class SearchResult:
    section_title: str
    query: str
    results: List[Dict]

@dataclass
class Citation:
    text: str
    source: str
    url: str
    section_title: str

@dataclass
class Summary:
    section_title: str
    summary: str
    citation_numbers: List[int]

@dataclass
class EnhancementQuery:
    section: str
    query: str
    reason: str

@dataclass
class Evaluation:
    completeness: str
    relevance: str
    structure: str
    correctness: str

@dataclass
class ReflectionResult:
    section_title: str
    evaluation: Evaluation
    needs_enhancement: bool
    enhancement_queries: List[EnhancementQuery]
    suggestions: List[str]

class PlanningState(TypedDict):
    query: str
    sections: List[Section]
    search_queries: List[SearchQuery]
    search_results: List[SearchResult]
    summaries: List[Summary]
    citations: Dict[int, Citation]
    reflection_results: List[ReflectionResult]
    general_reflection: ReflectionResult
    needs_enhancement: bool
    reflection_count: int
    is_query_valid: bool
    query_retry_count: int 