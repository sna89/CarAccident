import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from typing import List,  Any, Tuple, Optional, TypedDict
import json
import streamlit as st
import pandas as pd
import plotly.express as px
from langchain.output_parsers import StructuredOutputParser
from langchain.output_parsers import ResponseSchema
from pydantic import BaseModel, Field

from utils.session.session_handler import initialize_session
from components.map.prompts import (
    GENERAL_SUMMARY_PROMPT,
    CAUSE_SUMMARY_PROMPT,
    OUTCOME_SUMMARY_PROMPT,
    BASE_JOIN_PROMPT,
)

from components.map.static_queries import (
    GENERAL_ANALYSIS_QUERY, 
    CAUSE_ANALYSIS_QUERY, 
    OUTCOME_ANALYSIS_QUERY
)
from components.map.analysis_graph import create_analysis_graph


class SQLQueryResponse(BaseModel):
    """Model for SQL query response."""
    sql_query: str = Field(description="The SQL query to execute")

    def model_dump_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.model_dump())


def generate_base_query(filter_level: str, location_name_dict: dict, base_query_format_instructions: str) -> str:
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
        base_query_format_instructions=base_query_format_instructions
    )
    return st.session_state.sql_llm_agent.execute_sql_query(base_prompt)


def generate_visualization(df: pd.DataFrame) -> Optional[Any]:
    """Generate visualization from the analysis results.
    
    Args:
        df: DataFrame containing the analysis results
        
    Returns:
        Plotly figure or None if generation fails
    """
    # Create and run the analysis workflow
    graph = create_analysis_graph()
      
    result = graph.invoke({"df": df, "retry_count": 0, "max_retries": 3})
    
    # Return the figure if successful, None if there was an error
    return result.get("fig") if not result.get("error") else None


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


def get_analysis_config(analysis_type: str) -> Tuple[str, str]:
    """Get the appropriate query template and summary prompt for the analysis type.
    
    Args:
        analysis_type: Type of analysis to perform
        
    Returns:
        Tuple of (query_template, summary_prompt)
    """
    configs = {
        "general": (GENERAL_ANALYSIS_QUERY, GENERAL_SUMMARY_PROMPT),
        "cause": (CAUSE_ANALYSIS_QUERY, CAUSE_SUMMARY_PROMPT),
        "outcome": (OUTCOME_ANALYSIS_QUERY, OUTCOME_SUMMARY_PROMPT)
    }
    
    if analysis_type not in configs:
        raise ValueError(f"Invalid analysis type: {analysis_type}")
        
    return configs[analysis_type]


def create_sql_parser() -> Tuple[StructuredOutputParser, str]:
    """Create a SQL query parser and get its format instructions.
    
    Returns:
        Tuple of (parser, format_instructions)
    """
    response_schema = ResponseSchema(
        name="sql_query",
        description="The SQL query to execute"
    )
    parser = StructuredOutputParser.from_response_schemas([response_schema])
    format_instructions = parser.get_format_instructions()
    return parser, format_instructions


def parse_base_query(base_query: str, parser: StructuredOutputParser) -> str:
    """Parse the base query response using StructuredOutputParser.
    
    Args:
        base_query: Raw base query response
        parser: Configured parser instance
        
    Returns:
        Parsed SQL query string
    """
    parsed_response = parser.parse(base_query)
    return parsed_response.get("sql_query")


def execute_analysis_pipeline(
    filter_level: str,
    location_name_dict: dict,
    analysis_type: str
) -> Tuple[Optional[str], Optional[Any], Optional[pd.DataFrame]]:
    """Execute the complete analysis pipeline including query generation, data processing, and visualization.
    
    Args:
        filter_level: Level of location filtering
        location_name_dict: Dictionary of location names
        analysis_type: Type of analysis to perform ("general", "cause", or "outcome")
        
    Returns:
        Tuple containing analysis output, figure, and DataFrame
    """
    try:
        # Create parser and get format instructions
        parser, format_instructions = create_sql_parser()
        
        # Generate and parse base query
        base_query = generate_base_query(filter_level, location_name_dict, format_instructions)
        base_query_sql = parse_base_query(base_query, parser)
        
        # Get analysis configuration
        query_template, summary_prompt = get_analysis_config(analysis_type)
        
        # Create and execute final query
        query_data = SQLQueryResponse(sql_query=query_template.format(base_query=base_query_sql))
        df = st.session_state.db.execute_query(query_data.sql_query)
        
        # Generate visualization and summary
        fig = generate_visualization(df)
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
                    output, fig, df = execute_analysis_pipeline(
                        filter_level,
                        filtered_df[filtered_df[filter_level].isin(filtered_locations)].to_dict(),
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
    analyze_cause, fig, df = execute_analysis_pipeline(
        "ROAD",
        {'ROAD': {0: '70'}, 'SUBURB': {0: None}, 'TOWN': {0: None}, 'CITY': {0: 'Zvulun Regional Council'}, 'CITY_DISTRICT': {0: None}},
        "outcome"
    )
    print(analyze_cause)
