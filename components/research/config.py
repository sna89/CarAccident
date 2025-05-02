# LLM Configuration
LLM_CONFIG = {
    "model": "gpt-4o-mini",
    "temperature": 0
}

# Search Configuration
SEARCH_CONFIG = {
    "max_results": 1,
    "delay": 2  # Delay in seconds between searches
}

# Workflow Configuration
WORKFLOW_CONFIG = {
    "max_validation_retries": 3,  # Maximum number of retries for question validation
    "max_reflection_iterations": 2,
    "queries_per_section": 3,
    "max_enhancement_queries": 2
}

# Section Configuration
SECTION_CONFIG = {
    "conclusion_section": "Conclusions",  # The standard name for the conclusion section
    "max_sections": 2  # Maximum number of sections (excluding conclusion)
} 