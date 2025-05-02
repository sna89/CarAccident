from typing import List, Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
import sys
import os

# Add the current directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Now we can import directly from the same directory
from config import LLM_CONFIG, WORKFLOW_CONFIG

# Initialize the LLM
llm = ChatOpenAI(**LLM_CONFIG)

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

def create_planner_agent():
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert research planner. Given a research question, create a detailed plan with appropriate sections.
        Analyze the question and determine how many sections are needed to thoroughly cover the topic.
        Each section should focus on a specific aspect of the research question.
        
        Important: You must create no more than {max_sections} main sections (excluding the conclusion section).
        If the topic requires more coverage, combine related aspects into broader sections.
        
        Return a JSON with the following structure:
        {{
            "sections": [
                {{"title": "Introduction", "description": "What to cover in the introduction"}},
                {{"title": "Section Title", "description": "What to cover in this section"}},
                ... (add as many sections as needed, up to {max_sections})
                {{"title": "Conclusions", "description": "What to cover in the conclusions"}}
            ]
        }}
        
        Guidelines:
        - Create a clear, logical structure
        - Each section should have a distinct focus
        - Keep descriptions concise but informative
        - Ensure the conclusion section is always included
        - Do not exceed {max_sections} main sections"""),
        ("human", "{input}")
    ])
    return prompt | llm | JsonOutputParser()

def create_query_generator_agent():
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert at creating effective web search queries. Given a research question and a specific section,
        create {queries_per_section} focused search queries that will help gather relevant information for this section.
        
        Return a JSON with the following structure:
        {{
            "search_queries": [
                {{"section_title": "Section Title", "query": "search query for this section"}},
                ... (exactly {queries_per_section} queries for this section)
            ]
        }}
        
        Guidelines:
        - Create exactly {queries_per_section} specific, focused queries for this section
        - Each query should target a different aspect or perspective of the section's topic
        - Include relevant keywords from both the main question and section description
        - Use appropriate search operators if needed
        - Ensure queries are concise but comprehensive
        - Focus on finding authoritative sources and specific data relevant to this section
        - Make each query distinct and complementary to the others"""),
        ("human", "Main question: {main_question}\nSections: {sections}")
    ])
    return prompt | llm | JsonOutputParser()

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

def create_section_reflector_agent():
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are an expert research reviewer. Your task is to evaluate a specific section of the research and suggest improvements.
        
        Return a JSON with the following structure:
        {{{{ 
            "evaluation": {{{{ 
                "completeness": "assessment of how well this section addresses its specific topic",
                "relevance": "assessment of how relevant the information is to the section's focus",
                "structure": "assessment of the logical flow and organization within this section",
                "correctness": "assessment of factual accuracy in this section"
            }}}},
            "needs_enhancement": true/false,
            "enhancement_queries": [
                {{{{ 
                    "query": "specific search query to find missing information for this section",
                    "reason": "explanation of why this enhancement is needed"
                }}}}
            ],
            "suggestions": [
                "specific suggestions for improving this section"
            ]
        }}}}
        
        Guidelines:
        - Focus only on the specific section being evaluated
        - Consider how well this section contributes to answering the main research question
        - Suggest up to {WORKFLOW_CONFIG['max_enhancement_queries']} enhancement queries when necessary
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

def create_general_reflector_agent():
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are an expert research reviewer. Your task is to evaluate the entire research and suggest improvements.
        
        Return a JSON with the following structure:
        {{{{ 
            "evaluation": {{{{ 
                "completeness": "assessment of how well the research covers all aspects of the question",
                "relevance": "assessment of how well the research stays focused on the main question",
                "structure": "assessment of the logical flow and organization of the entire research",
                "correctness": "assessment of factual accuracy across all sections"
            }}}},
            "needs_enhancement": true/false,
            "enhancement_queries": [
                {{{{ 
                    "query": "specific search query to find missing information",
                    "reason": "explanation of why this enhancement is needed",
                    "section": "section title that needs this enhancement"
                }}}}
            ],
            "suggestions": [
                "specific suggestions for improving the overall research"
            ]
        }}}}
        
        Guidelines:
        - Evaluate the research as a whole, considering how all sections work together
        - Consider how well the research answers the main question
        - Suggest up to {WORKFLOW_CONFIG['max_enhancement_queries']} enhancement queries when necessary
        - Focus on gaps in information or areas needing clarification
        - Ensure suggested queries are specific and targeted to the research needs
        - Consider both content and structure in your evaluation"""),
        ("human", """Original Question: {main_question}
        
Research Sections:
{sections}

Citations:
{citations}""")
    ])
    return prompt | llm | JsonOutputParser()

def create_enhanced_summary_agent():
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert at enhancing research summaries. Given a current summary, new search results, and reflection insights,
        create an improved summary that addresses the evaluation feedback and incorporates new information.
        
        Return a JSON with the following structure:
        {{
            "summary": "enhanced summary with [citation numbers] inline",
            "citations": [
                {{
                    "text": "specific text being cited",
                    "source": "title or source of the information",
                    "url": "URL of the source (must be included)"
                }}
            ]
        }}
        
        Guidelines:
        - Start with the current summary and improve it based on the evaluation and suggestions
        - Incorporate new information from the search results to address gaps
        - Focus on improving areas identified in the evaluation (completeness, relevance, structure, correctness)
        - Maintain or improve the logical flow and organization
        - Include specific citations for important facts or statistics
        - Place citation numbers in square brackets [1] immediately after the relevant sentence
        - Make sure citations are clear and traceable
        - Focus on the most relevant and reliable information
        - Maintain academic tone and accuracy
        - ALWAYS include a valid URL for each citation
        - NEVER include comments or placeholders in the JSON output
        - Example: "While the original summary mentioned distracted driving [1], new research shows it accounts for 25% of accidents [2]."""),
        ("human", """Section: {section_title}
Description: {section_description}

Current Summary:
{current_summary}

Evaluation:
{evaluation}

Suggestions:
{suggestions}

New Search Results:
{new_search_results}""")
    ])
    return prompt | llm | JsonOutputParser()