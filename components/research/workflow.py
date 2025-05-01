from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
import json
import time
from duckduckgo_search import DDGS

# Constants
MAX_RETRIES = 3
MAX_SEARCH_RESULTS = 1
SEARCH_DELAY = 2  # Delay in seconds between searches
MAX_REFLECTION_ITERATIONS = 2  # Maximum number of reflection and enhancement cycles

# Define the state for our workflow
class PlanningState(TypedDict):
    query: str  # The initial research question
    sections: List[dict]  # The planned sections
    search_queries: List[dict]  # Search queries for each section
    search_results: List[dict]  # Web search results for each section
    summaries: List[dict]  # Summaries for each section
    citations: Dict[int, dict]  # Global citation list
    reflection_results: List[dict]  # Results from reflection process
    needs_enhancement: bool  # Whether the research needs enhancement
    enhancement_queries: List[dict]  # Additional queries suggested by reflection
    reflection_count: int  # Number of reflection iterations
    is_valid: bool  # Whether the question is related to car accidents
    retry_count: int  # Number of validation attempts

# Initialize the LLM and search tool
llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
search = DDGS()

# Create the question validator agent
def create_validator_agent():
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert at analyzing questions. Your task is to determine if a question is related to car accidents.
        Return a JSON with the following structure:
        {{
            "is_valid": true/false,
            "reason": "brief explanation of why the question is or isn't related to car accidents"
        }}
        
        Guidelines:
        - Consider questions about car accidents, road safety, traffic incidents, vehicle collisions, etc. as valid
        - Questions should be directly or indirectly related to car accidents
        - If the question is not related, provide a clear explanation why"""),
        ("human", "{input}")
    ])
    return prompt | llm | JsonOutputParser()

# Create the planner agent
def create_planner_agent():
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert research planner. Given a research question, create a detailed plan with appropriate sections.
        Analyze the question and determine how many sections are needed to thoroughly cover the topic.
        Each section should focus on a specific aspect of the research question.
        
        Return a JSON with the following structure:
        {{
            "sections": [
                {{"title": "Introduction", "description": "What to cover in the introduction"}},
                {{"title": "Section Title", "description": "What to cover in this section"}},
                ... (add as many sections as needed)
                {{"title": "Conclusions", "description": "What to cover in the conclusions"}}
            ]
        }}
        
        Guidelines:
        - Always include Introduction and Conclusions sections
        - Add as many body sections as needed to cover the topic comprehensively
        - Each section should have a clear, descriptive title
        - The description should explain what specific aspects will be covered in that section
        - Ensure sections flow logically from one to the next"""),
        ("human", "{input}")
    ])
    return prompt | llm | JsonOutputParser()

# Create the search query generator agent
def create_query_generator_agent():
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert at creating effective web search queries. Given a research question and a section plan,
        create specific search queries that will help gather relevant information for each section.
        
        Return a JSON with the following structure:
        {{
            "search_queries": [
                {{"section_title": "Introduction", "query": "search query for introduction"}},
                {{"section_title": "Section Title", "query": "search query for this section"}},
                ... (one query per section)
            ]
        }}
        
        Guidelines:
        - Create specific, focused queries for each section
        - Include relevant keywords from both the main question and section description
        - Use appropriate search operators if needed
        - Ensure queries are concise but comprehensive
        - Each query should target the specific information needed for that section
        - DO NOT create queries for summary or conclusion sections as they will summarize existing content"""),
        ("human", "Main question: {main_question}\nSections: {sections}")
    ])
    return prompt | llm | JsonOutputParser()

# Create the summary generator agent
def create_summary_agent():
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert at summarizing research findings. Given search results for a section,
        create a comprehensive summary that incorporates information from multiple sources.
        
        Return a JSON with the following structure:
        {{
            "summary": "detailed summary of the findings with [citation numbers] inline",
            "citations": [
                {{
                    "text": "specific text being cited",
                    "source": "title or source of the information",
                    "url": "URL of the source (must be included)"
                }}
            ]
        }}
        
        Guidelines:
        - Create a coherent summary that combines information from all sources
        - Include specific citations for important facts or statistics
        - Place citation numbers in square brackets [1] immediately after the relevant sentence
        - Make sure citations are clear and traceable
        - Focus on the most relevant and reliable information
        - Maintain academic tone and accuracy
        - ALWAYS include the URL for each citation
        - Example: "According to recent studies, distracted driving is a major cause of accidents [1]."""),
        ("human", "Section: {section_title}\nDescription: {section_description}\nSearch Results: {search_results}")
    ])
    return prompt | llm | JsonOutputParser()

# Create the conclusion generator agent
def create_conclusion_agent():
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert at creating research conclusions. Given summaries from all sections,
        create a comprehensive conclusion that ties together all the findings.
        
        Return a JSON with the following structure:
        {{
            "summary": "comprehensive conclusion with [citation numbers] inline",
            "citations": [
                {{
                    "text": "specific text being cited",
                    "source": "title or source of the information",
                    "url": "URL of the source (must be included)"
                }}
            ]
        }}
        
        Guidelines:
        - Synthesize information from all section summaries
        - Highlight key findings and their implications
        - Draw meaningful conclusions from the research
        - Place citation numbers in square brackets [1] immediately after relevant sentences
        - Maintain academic tone and accuracy
        - Focus on the main research question
        - ALWAYS include the URL for each citation
        - Example: "The research demonstrates that urban car accidents are primarily caused by three factors [1]."""),
        ("human", "Main Question: {main_question}\nSection Summaries: {section_summaries}")
    ])
    return prompt | llm | JsonOutputParser()

# Create the reflector agent
def create_reflector_agent():
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert research reviewer. Your task is to evaluate a specific section of the research and suggest improvements.
        
        Return a JSON with the following structure:
        {{
            "evaluation": {{
                "completeness": "assessment of how well this section addresses its specific topic",
                "relevance": "assessment of how relevant the information is to the section's focus",
                "structure": "assessment of the logical flow and organization within this section",
                "correctness": "assessment of factual accuracy in this section"
            }},
            "needs_enhancement": true/false,
            "enhancement_queries": [
                {{
                    "query": "specific search query to find missing information for this section",
                    "reason": "explanation of why this enhancement is needed"
                }}
            ],
            "suggestions": [
                "specific suggestions for improving this section"
            ]
        }}
        
        Guidelines:
        - Focus only on the specific section being evaluated
        - Consider how well this section contributes to answering the main research question
        - Suggest enhancements only when necessary for this specific section
        - Focus on gaps in information or areas needing clarification within this section
        - Ensure suggested queries are specific and targeted to this section's needs"""),
        ("human", """Original Question: {main_question}
        
Section Title: {section_title}
Section Description: {section_description}
Section Content: {section_content}

Section Citations:
{citations}""")
    ])
    return prompt | llm | JsonOutputParser()

# Define the validation node
def validate_question(state: PlanningState) -> PlanningState:
    validator = create_validator_agent()
    result = validator.invoke({"input": state["query"]})
    state["is_valid"] = result["is_valid"]
    state["retry_count"] += 1
    
    if not state["is_valid"]:
        print(f"\nInvalid question (Attempt {state['retry_count']}/{MAX_RETRIES}): {result['reason']}")
        if state["retry_count"] < MAX_RETRIES:
            print("Please enter a new question related to car accidents:")
            new_question = input("> ").strip()
            state["query"] = new_question
    
    return state

# Define the retry decision node
def should_retry(state: PlanningState) -> str:
    if state["is_valid"]:
        return "plan_sections"
    elif state["retry_count"] < MAX_RETRIES:
        return "validate_question"
    else:
        print(f"\nMaximum retries ({MAX_RETRIES}) reached. Please try again with a question related to car accidents.")
        return "end"

# Define the planning node
def plan_sections(state: PlanningState) -> PlanningState:
    planner_agent = create_planner_agent()
    result = planner_agent.invoke({"input": state["query"]})
    state["sections"] = result["sections"]
    return state

# Define the search query generation node
def generate_search_queries(state: PlanningState) -> PlanningState:
    # Filter out only conclusion/summary sections
    non_summary_sections = [
        section for section in state["sections"]
        if section["title"].lower() not in ["conclusions", "conclusion", "summary"]
    ]
    
    if non_summary_sections:
        query_generator = create_query_generator_agent()
        result = query_generator.invoke({
            "main_question": state["query"],
            "sections": json.dumps(non_summary_sections)
        })
        state["search_queries"] = result["search_queries"]
    else:
        state["search_queries"] = []
    
    return state

# Define the web search node
def perform_web_search(state: PlanningState) -> PlanningState:
    state["search_results"] = []
    
    for query in state["search_queries"]:
        print(f"\nSearching for: {query['query']}")
        try:
            # Use DDGS directly with rate limiting
            results = []
            for r in search.text(query["query"], max_results=MAX_SEARCH_RESULTS):
                results.append(r)
                time.sleep(SEARCH_DELAY)  # Add delay between results
            
            # Store results with section context
            state["search_results"].append({
                "section_title": query["section_title"],
                "query": query["query"],
                "results": results
            })
            
        except Exception as e:
            print(f"Error searching for {query['query']}: {str(e)}")
            state["search_results"].append({
                "section_title": query["section_title"],
                "query": query["query"],
                "results": f"Search failed: {str(e)}"
            })
    
    return state

# Define the summary generation node
def generate_summaries(state: PlanningState) -> PlanningState:
    state["summaries"] = []
    state["citations"] = {}
    citation_counter = 1
    
    # First, generate summaries for all sections except conclusion
    for section in state["sections"]:
        if section["title"].lower() not in ["conclusions", "conclusion", "summary"]:
            # Find matching search results
            matching_results = next(
                (result for result in state["search_results"] 
                 if result["section_title"] == section["title"]),
                None
            )
            
            if matching_results:
                summary_agent = create_summary_agent()
                result = summary_agent.invoke({
                    "section_title": section["title"],
                    "section_description": section["description"],
                    "search_results": json.dumps(matching_results["results"])
                })
                
                # Process citations and assign numbers
                processed_citations = []
                for citation in result["citations"]:
                    if "url" not in citation or not citation["url"]:
                        # If URL is missing, try to find it in the search results
                        matching_result = next(
                            (r for r in matching_results["results"] 
                             if citation["text"] in r.get("body", "")),
                            None
                        )
                        if matching_result and "link" in matching_result:
                            citation["url"] = matching_result["link"]
                    
                    if "url" in citation and citation["url"]:
                        state["citations"][citation_counter] = {
                            "text": citation["text"],
                            "source": citation["source"],
                            "url": citation["url"]
                        }
                        processed_citations.append(citation_counter)
                        citation_counter += 1
                
                # Update summary text with correct citation numbers
                summary_text = result["summary"]
                for i, citation_num in enumerate(processed_citations):
                    summary_text = summary_text.replace(f"[{i+1}]", f"[{citation_num}]")
                
                # Store summary with citation numbers already in the text
                state["summaries"].append({
                    "section_title": section["title"],
                    "summary": summary_text,
                    "citation_numbers": processed_citations
                })
    
    # Then, generate the conclusion based on all previous summaries
    conclusion_section = next(
        (section for section in state["sections"] 
         if section["title"].lower() in ["conclusions", "conclusion", "summary"]),
        None
    )
    
    if conclusion_section:
        # Prepare section summaries for the conclusion
        section_summaries = [
            f"{summary['section_title']}:\n{summary['summary']}"
            for summary in state["summaries"]
        ]
        
        conclusion_agent = create_conclusion_agent()
        result = conclusion_agent.invoke({
            "main_question": state["query"],
            "section_summaries": "\n\n".join(section_summaries)
        })
        
        # Process citations for conclusion
        processed_citations = []
        for citation in result["citations"]:
            if "url" not in citation or not citation["url"]:
                # Try to find the URL in previous citations
                matching_citation = next(
                    (c for c in state["citations"].values() 
                     if citation["text"] in c["text"]),
                    None
                )
                if matching_citation:
                    citation["url"] = matching_citation["url"]
            
            if "url" in citation and citation["url"]:
                state["citations"][citation_counter] = {
                    "text": citation["text"],
                    "source": citation["source"],
                    "url": citation["url"]
                }
                processed_citations.append(citation_counter)
                citation_counter += 1
        
        # Update conclusion text with correct citation numbers
        conclusion_text = result["summary"]
        for i, citation_num in enumerate(processed_citations):
            conclusion_text = conclusion_text.replace(f"[{i+1}]", f"[{citation_num}]")
        
        # Add conclusion to summaries
        state["summaries"].append({
            "section_title": conclusion_section["title"],
            "summary": conclusion_text,
            "citation_numbers": processed_citations
        })
    
    return state

# Define the reflection node
def reflect_on_research(state: PlanningState) -> PlanningState:
    if state["reflection_count"] >= MAX_REFLECTION_ITERATIONS:
        return state
    
    # Initialize reflection results
    state["reflection_results"] = []
    state["needs_enhancement"] = False
    state["enhancement_queries"] = []
    
    # Evaluate each section separately
    for summary in state["summaries"]:
        # Get citations relevant to this section
        section_citations = "\n".join([
            f"[{num}] {citation['text']}\n   Source: {citation['source']}\n   URL: {citation['url']}"
            for num, citation in state["citations"].items()
            if num in summary.get("citation_numbers", [])
        ])
        
        # Run reflection for this section
        reflector = create_reflector_agent()
        result = reflector.invoke({
            "main_question": state["query"],
            "section_title": summary["section_title"],
            "section_description": summary.get("description", ""),
            "section_content": summary["summary"],
            "citations": section_citations
        })
        
        # Store reflection results
        state["reflection_results"].append({
            "section_title": summary["section_title"],
            "evaluation": result
        })
        
        # If any section needs enhancement, mark the whole research for enhancement
        if result["needs_enhancement"]:
            state["needs_enhancement"] = True
            # Add enhancement queries with section context
            for query in result["enhancement_queries"]:
                state["enhancement_queries"].append({
                    "section": summary["section_title"],
                    "query": query["query"],
                    "reason": query["reason"]
                })
    
    state["reflection_count"] += 1
    return state

# Define the enhancement node
def enhance_research(state: PlanningState) -> PlanningState:
    if not state["needs_enhancement"]:
        return state
    
    # Process each enhancement query
    for enhancement in state["enhancement_queries"]:
        print(f"\nEnhancing section '{enhancement['section']}':")
        print(f"Reason: {enhancement['reason']}")
        print(f"Searching for: {enhancement['query']}")
        
        try:
            # Perform additional search
            results = []
            for r in search.text(enhancement["query"], max_results=MAX_SEARCH_RESULTS):
                results.append(r)
                time.sleep(SEARCH_DELAY)
            
            # Generate enhanced summary
            summary_agent = create_summary_agent()
            result = summary_agent.invoke({
                "section_title": enhancement["section"],
                "section_description": "Additional information to enhance the section",
                "search_results": json.dumps(results)
            })
            
            # Process new citations
            for citation in result["citations"]:
                if "url" not in citation or not citation["url"]:
                    matching_result = next(
                        (r for r in results 
                         if citation["text"] in r.get("body", "")),
                        None
                    )
                    if matching_result and "link" in matching_result:
                        citation["url"] = matching_result["link"]
                
                if "url" in citation and citation["url"]:
                    state["citations"][len(state["citations"]) + 1] = {
                        "text": citation["text"],
                        "source": citation["source"],
                        "url": citation["url"]
                    }
            
            # Update the section summary
            for summary in state["summaries"]:
                if summary["section_title"] == enhancement["section"]:
                    summary["summary"] = result["summary"]
                    break
        
        except Exception as e:
            print(f"Error enhancing section: {str(e)}")
    
    return state

# Create and compile the workflow graph
def create_planning_graph() -> StateGraph:
    graph = StateGraph(PlanningState)
    
    # Add nodes
    graph.add_node("validate_question", validate_question)
    graph.add_node("plan_sections", plan_sections)
    graph.add_node("generate_queries", generate_search_queries)
    graph.add_node("web_search", perform_web_search)
    graph.add_node("generate_summaries", generate_summaries)
    graph.add_node("reflect", reflect_on_research)
    graph.add_node("enhance", enhance_research)
    
    # Add edges
    graph.add_conditional_edges(
        "validate_question",
        should_retry,
        {
            "plan_sections": "plan_sections",
            "validate_question": "validate_question",
            "end": END
        }
    )
    graph.add_edge("plan_sections", "generate_queries")
    graph.add_edge("generate_queries", "web_search")
    graph.add_edge("web_search", "generate_summaries")
    graph.add_edge("generate_summaries", "reflect")
    
    # Add conditional edge for enhancement
    def should_enhance(state: PlanningState) -> str:
        if state["needs_enhancement"] and state["reflection_count"] < MAX_REFLECTION_ITERATIONS:
            return "enhance"
        return "end"
    
    graph.add_conditional_edges(
        "reflect",
        should_enhance,
        {
            "enhance": "enhance",
            "end": END
        }
    )
    graph.add_edge("enhance", "generate_summaries")
    
    graph.set_entry_point("validate_question")
    return graph.compile()

# Example usage
if __name__ == "__main__":
    # Create the graph
    graph = create_planning_graph()
    
    # Get question from user
    print("\nWelcome to the Car Accident Research Assistant!")
    print("Please enter your question about car accidents:")
    question = input("> ").strip()
    
    # Run the workflow
    initial_state = PlanningState(
        query=question,
        sections=[],
        search_queries=[],
        search_results=[],
        summaries=[],
        citations={},
        reflection_results=[],
        needs_enhancement=False,
        enhancement_queries=[],
        reflection_count=0,
        is_valid=False,
        retry_count=0
    )

    final_state = graph.invoke(initial_state)
    
    if final_state["is_valid"]:
        print("\nResearch Summary:")
        print("=" * 50)
        print(f"Main Question: {question}\n")
        
        # Print each section's summary
        for summary in final_state["summaries"]:
            print(f"\n{summary['section_title']}:")
            print(summary["summary"])
            print("-" * 50)
        
        # Print reflection results
        if final_state["reflection_results"]:
            print("\nResearch Evaluation:")
            print("=" * 50)
            for reflection in final_state["reflection_results"]:
                print(f"\nSection: {reflection['section_title']}")
                eval_result = reflection["evaluation"]
                print(f"Completeness: {eval_result['evaluation']['completeness']}")
                print(f"Relevance: {eval_result['evaluation']['relevance']}")
                print(f"Structure: {eval_result['evaluation']['structure']}")
                print(f"Correctness: {eval_result['evaluation']['correctness']}")
                print("\nSuggestions:")
                for suggestion in eval_result["suggestions"]:
                    print(f"- {suggestion}")
                print("-" * 50)
        
        # Print all citations at the end
        print("\nReferences:")
        print("=" * 50)
        for num, citation in final_state["citations"].items():
            print(f"[{num}] {citation['text']}")
            print(f"   Source: {citation['source']}")
            print(f"   URL: {citation['url']}\n")