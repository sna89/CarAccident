import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from typing import List, Dict, Any, Tuple, Optional
import json
import streamlit as st
import pandas as pd
import plotly.express as px
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from utils.session.session_handler import initialize_session
from utils.agents.python_agent import run_chart_code
from components.map.prompts import (
    GENERAL_ANALYSIS_PROMPT,
    GENERAL_SUMMARY_PROMPT,
    CAUSE_ANALYSIS_PROMPT,
    CAUSE_SUMMARY_PROMPT,
    OUTCOME_ANALYSIS_PROMPT,
    OUTCOME_SUMMARY_PROMPT,
    OUTCOME_CATEGORIES,
    CAUSE_COLUMN_CATEGORIES as COLUMN_CATEGORIES,
    BASE_JOIN_PROMPT,
    GENERAL_INSTRUCTIONS,
    FINAL_OUTPUT_FORMAT,
    CHART_PROMPT
)


# Data Models
class AccidentData(BaseModel):
    """Model for accident data statistics."""
    year: int = Field(description="The year of the accident")
    severity_level: str = Field(description="The severity level (fatal, serious, minor)")
    accident_count: int = Field(description="Number of accidents")
    people_involved: int = Field(description="Total number of people involved")
    vehicles_involved: int = Field(description="Total number of vehicles involved")
    avg_people_per_accident: float = Field(description="Average people per accident")
    avg_vehicles_per_accident: float = Field(description="Average vehicles per accident")


class AccidentResponse(BaseModel):
    """Model for accident analysis response."""
    accidents: List[AccidentData] = Field(description="List of accident data over time")

    def model_dump_json(self) -> str:
        """Convert to a flat list of accident data."""
        return json.dumps([accident.model_dump() for accident in self.accidents])


class CauseData(BaseModel):
    """Model for cause analysis data."""
    year: int = Field(description="The year of the accident")
    severity_level: str = Field(description="The severity level (fatal, serious, minor)")
    cause_column: str = Field(description="The name of the cause column")
    category_name: str = Field(description="The specific category name")
    count: int = Field(description="Number of occurrences")
    percentage: float = Field(description="Percentage of total for that severity level")


class CauseResponse(BaseModel):
    """Model for cause analysis response."""
    causes: List[CauseData] = Field(description="List of cause analysis data")

    def model_dump_json(self) -> str:
        """Convert to a flat list of cause data."""
        return json.dumps([cause.model_dump() for cause in self.causes])


class OutcomeData(BaseModel):
    """Model for outcome analysis data."""
    year: int = Field(description="The year of the accident")
    severity_level: str = Field(description="The severity level (fatal, serious, minor)")
    outcome_column: str = Field(description="The name of the outcome column")
    category_name: str = Field(description="The specific category name")
    count: int = Field(description="Number of occurrences")
    percentage: float = Field(description="Percentage of total for that severity level")
    avg_casualties: float = Field(description="Average number of casualties")
    avg_vehicles: float = Field(description="Average number of vehicles involved")


class OutcomeResponse(BaseModel):
    """Model for outcome analysis response."""
    outcomes: List[OutcomeData] = Field(description="List of outcome analysis data")

    def model_dump_json(self) -> str:
        """Convert to a flat list of outcome data."""
        return json.dumps([outcome.model_dump() for outcome in self.outcomes])


# Helper Functions
def load_column_mappings() -> Dict[str, Dict[str, str]]:
    """Load column mappings from JSON file."""
    try:
        with open('data/accident_data_mapping.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading column mappings: {str(e)}")
        return {}


def format_column_categories(categories: Dict[str, str], mappings: Dict[str, Dict[str, str]]) -> Tuple[str, str]:
    """Format column categories for prompt parameters."""
    column_categories = {
        col: mappings[mapping_key]
        for col, mapping_key in categories.items()
    }
    
    columns = "\n".join([f"- {col}" for col in column_categories.keys()])
    category_details = "\n".join([
        f"* {col} categories: {', '.join(cats.values())}"
        for col, cats in column_categories.items()
    ])
    
    return columns, category_details


# Analysis Functions
@st.cache_data()
def analyze_data(
    filtered_locations: List[str],
    filter_level: str,
    filtered_df: pd.DataFrame,
    analysis_type: str = "general"
) -> Tuple[Optional[str], Optional[Any], Optional[pd.DataFrame]]:
    """Analyze accident data based on the specified type.
    
    Args:
        filtered_locations: List of locations to analyze
        filter_level: Level of location filtering
        filtered_df: Filtered DataFrame containing the data
        analysis_type: Type of analysis to perform ("general", "cause", or "outcome")
        
    Returns:
        Tuple containing analysis output, figure, and DataFrame
    """
    location_name_dict = filtered_df[filtered_df[filter_level].isin(filtered_locations)].to_dict()
    
    analysis_functions = {
        "general": analyze_general,
        "cause": analyze_cause,
        "outcome": analyze_outcome
    }
    
    if analysis_type not in analysis_functions:
        return None, None, None
        
    return analysis_functions[analysis_type](filter_level, location_name_dict)


def analyze_general(filter_level: str, location_name_dict: dict) -> Tuple[Optional[str], Optional[Any], Optional[pd.DataFrame]]:
    """Analyze general accident statistics."""
    return execute_analysis_pipeline(
        filter_level,
        location_name_dict,
        {},  # No categories needed for general analysis
        GENERAL_ANALYSIS_PROMPT,
        GENERAL_SUMMARY_PROMPT,
        AccidentResponse
    )


def analyze_cause(filter_level: str, location_name_dict: dict) -> Tuple[Optional[str], Optional[Any], Optional[pd.DataFrame]]:
    """Analyze accident causes by severity level."""
    return execute_analysis_pipeline(
        filter_level,
        location_name_dict,
        COLUMN_CATEGORIES,
        CAUSE_ANALYSIS_PROMPT,
        CAUSE_SUMMARY_PROMPT,
        CauseResponse
    )


def analyze_outcome(filter_level: str, location_name_dict: dict) -> Tuple[Optional[str], Optional[Any], Optional[pd.DataFrame]]:
    """Analyze accident outcomes."""
    return execute_analysis_pipeline(
        filter_level,
        location_name_dict,
        OUTCOME_CATEGORIES,
        OUTCOME_ANALYSIS_PROMPT,
        OUTCOME_SUMMARY_PROMPT,
        OutcomeResponse
    )


def generate_base_query(filter_level: str, location_name_dict: dict) -> str:
    """Generate the base SQL query for analysis.
    
    Args:
        filter_level: Level of location filtering
        location_name_dict: Dictionary of location names
        
    Returns:
        The generated base query
    """
    base_prompt = BASE_JOIN_PROMPT.format(
        filter_level=filter_level,
        location_dict=location_name_dict,
        general_instructions=GENERAL_INSTRUCTIONS,
        final_output_format=FINAL_OUTPUT_FORMAT
    )
    return st.session_state.sql_llm_agent.execute_sql_query(base_prompt)


def prepare_prompt_parameters(
    base_query: str,
    response_model: BaseModel,
    categories: Optional[Dict[str, str]] = None
) -> Dict[str, str]:
    """Prepare parameters for the analysis prompt.
    
    Args:
        base_query: The base SQL query
        response_model: Pydantic model for response parsing
        categories: Optional dictionary of categories for analysis
        
    Returns:
        Dictionary of prompt parameters
    """
    parser = PydanticOutputParser(pydantic_object=response_model)
    prompt_params = {
        "base_query": base_query,
        "format_instructions": parser.get_format_instructions(),
        "general_instructions": GENERAL_INSTRUCTIONS,
        "final_output_format": FINAL_OUTPUT_FORMAT
    }
    
    if categories:
        mappings = load_column_mappings()
        columns, category_details = format_column_categories(categories, mappings)
        prompt_params.update({
            "columns": columns,
            "category_details": category_details
        })
    
    return prompt_params


def process_analysis_results(
    analysis_results: str,
    response_model: BaseModel
) -> Tuple[pd.DataFrame, str]:
    """Process and parse the analysis results.
    
    Args:
        analysis_results: Raw analysis results from SQL query
        response_model: Pydantic model for response parsing
        
    Returns:
        Tuple of (DataFrame, JSON string)
    """
    parser = PydanticOutputParser(pydantic_object=response_model)
    parsed_output = parser.parse(analysis_results)
    json_str = parsed_output.model_dump_json()
    df = pd.read_json(json_str)
    return df, json_str


def generate_visualization(df: pd.DataFrame, json_str: str) -> Optional[Any]:
    """Generate visualization from the analysis results.
    
    Args:
        df: DataFrame containing the analysis results
        json_str: JSON string representation of the data
        
    Returns:
        Plotly figure or None if generation fails
    """
    chart_prompt = CHART_PROMPT.format(data=json_str)
    fig_result = st.session_state.python_agent.execute(chart_prompt)
    return run_chart_code(fig_result, df)


def generate_summary(df: pd.DataFrame, summary_prompt: str) -> Any:
    """Generate summary of the analysis results.
    
    Args:
        df: DataFrame containing the analysis results
        summary_prompt: Prompt template for summary generation
        
    Returns:
        Generated summary
    """
    return st.session_state.llm_model.invoke(
        summary_prompt.format(data=df)
    )


def execute_analysis_pipeline(
    filter_level: str,
    location_name_dict: dict,
    categories: Dict[str, str],
    analysis_prompt: str,
    summary_prompt: str,
    response_model: BaseModel
) -> Tuple[Optional[str], Optional[Any], Optional[pd.DataFrame]]:
    """Execute the complete analysis pipeline including query generation, data processing, and visualization.
    
    Args:
        filter_level: Level of location filtering
        location_name_dict: Dictionary of location names
        categories: Dictionary of categories for analysis
        analysis_prompt: Prompt template for analysis
        summary_prompt: Prompt template for summary
        response_model: Pydantic model for response parsing
        
    Returns:
        Tuple containing analysis output, figure, and DataFrame
    """
    try:
        # Generate base query
        base_query = generate_base_query(filter_level, location_name_dict)
        
        # Prepare prompt parameters
        prompt_params = prepare_prompt_parameters(base_query, response_model, categories)
        
        # Execute analysis
        analysis_results = st.session_state.sql_llm_agent.execute_sql_query(
            analysis_prompt.format(**prompt_params)
        )
        
        # Process results
        df, json_str = process_analysis_results(analysis_results, response_model)
        
        # Generate visualization
        fig = generate_visualization(df, json_str)
        
        # Generate summary
        summary_output = generate_summary(df, summary_prompt)
        
        return summary_output, fig, df
        
    except Exception as e:
        st.error(f"Error executing or parsing query: {str(e)}")
        return None, None, None


def analyze_dataframe(
    filtered_locations: List[str],
    filter_level: str,
    filtered_df: pd.DataFrame,
    analysis_type: str = "general"
) -> None:
    """Run AI-based accident analysis and display results.
    
    Args:
        filtered_locations: List of locations to analyze
        filter_level: Level of location filtering
        filtered_df: Filtered DataFrame containing the data
        analysis_type: Type of analysis to perform
    """
    # Set up button name and columns
    button_names = {
        "general": "General Analysis",
        "cause": "Cause Analysis",
        "outcome": "Outcome Analysis"
    }
    
    button_name = button_names.get(analysis_type)
    if not button_name:
        return
        
    # Create layout
    button_col, popover_output_col, chart_output_col, data_output_col = st.columns([1.5, 1.5, 5, 5])
    
    with button_col:
        if st.button(
            f"📈 {button_name}",
            key=f"Accident {str.capitalize(analysis_type)} Analysis",
            help=f"Process {analysis_type} accident data"
        ):
            # Validate inputs
            if filtered_df.empty:
                st.error("Please choose point on the map and filter level to proceed")
            elif not filter_level:
                st.error("Please choose a filter level.")
            elif not filtered_locations:
                st.error("Please choose locations to analyze.")
            else:
                # Perform analysis
                with st.spinner(f"Performing {analysis_type} analysis..."):
                    output, fig, df = analyze_data(
                        filtered_locations,
                        filter_level,
                        filtered_df,
                        analysis_type
                    )
                    
                    # Display results
                    with popover_output_col:
                        with st.popover(f"Show {button_name}"):
                            st.markdown(output.content)
                    
                    if fig is not None:
                        with chart_output_col:
                            with st.popover(f"Show {button_name} chart", use_container_width=True):
                                st.plotly_chart(
                                    fig,
                                    use_container_width=True,
                                    height=500,
                                    width=1500,
                                    config={'displayModeBar': True}
                                )
                    else:
                        with chart_output_col:
                            st.info("No data available to create chart")
                    
                    # Display DataFrame
                    if df is not None:
                        with data_output_col:
                            with st.popover(f"Show {button_name} Data", use_container_width=True):
                                st.dataframe(
                                    df,
                                    use_container_width=True,
                                    height=500
                                )


if __name__ == "__main__":
    initialize_session()
    # Example usage
    analyze_cause, fig, df = analyze_general(
        "ROAD",
        {'ROAD': {9: '40'}, 'SUBURB': {9: None}, 'TOWN': {9: None}, 'CITY': {9: None}, 'CITY_DISTRICT': {9: None}}
    )
    print(analyze_cause)
