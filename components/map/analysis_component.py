import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import INFERENCE_COLUMNS
from utils.session.session_handler import initialize_session
import streamlit as st
import pandas as pd
import plotly.express as px
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

SHARED_PROMPT = """
First, generate a SQL query to filter accident data with the following location filter:
Filter Level: {}
Location: {}

Location Hierarchy Logic:
- Each location belongs to a hierarchy (Road → Suburb → City District → Town → City)
- Filter should consider the full context of the location in its hierarchy
- Higher level filters (e.g., City) don't need lower level details
- Lower level filters (e.g., Road) should include their parent locations

Query Structure:
1. Base Query:
   - Start with accidents table
   - Join with location tables based on hierarchy
   - Apply location filters
   - Select all relevant columns
   - DO NOT use LIMIT clause

2. Aggregation Query:
   - Use the base query as a CTE (WITH clause)
   - Group by location hierarchy levels
   - Calculate aggregations at each level
   - Include all necessary metrics
   - DO NOT use LIMIT clause
   - Use ROUND() for all division calculations (e.g., averages)

Examples:
1. Filtering by Road "Main St":
   - Base query: Join accidents with road, suburb, district, town, city
   - Aggregation: Group by road, suburb, district, town, city
   - Include all parent locations in results

2. Filtering by City "Big City":
   - Base query: Join accidents with city
   - Aggregation: Group by city
   - No need for lower level details

Requirements for data filtering:
1. Select all relevant columns
2. Apply location filters considering the hierarchy
3. Include necessary joins for location context
4. Create a CTE for base data
5. Add aggregation query with proper grouping
6. DO NOT use LIMIT clause in any query
7. Use ROUND() for all division calculations
"""

class AccidentData(BaseModel):
    year: int = Field(description="The year of the accident")
    severity: str = Field(description="The severity level (fatal, serious, light)")
    accident_count: int = Field(description="Number of accidents")
    people_involved: int = Field(description="Total number of people involved")
    vehicles_involved: int = Field(description="Total number of vehicles involved")
    avg_people_per_accident: float = Field(description="Average people per accident")
    avg_vehicles_per_accident: float = Field(description="Average vehicles per accident")

class AccidentResponse(BaseModel):
    accidents: List[AccidentData] = Field(description="List of accident data")

@st.cache_data()
def analyze_data(filtered_locations, filter_level, filtered_df, analysis_type="general"):
    """Analyze accident data."""
    location_name_dict = filtered_df[filtered_df[filter_level].isin(filtered_locations)].to_dict()
    
    if analysis_type == "general":
        output, _ = analyze_general(filter_level, location_name_dict)
        return output, None
    elif analysis_type == "cause":
        output = analyze_cause(filter_level, location_name_dict)
        return output, None
    elif analysis_type == "outcome":
        output = analyze_outcome(filter_level, location_name_dict)
        return output, None
    else:
        return None, None
    

def analyze_general(filter_level: str, location_name_dict: dict):
    """Analyze general accident statistics."""

    # Create parser first
    parser = PydanticOutputParser(pydantic_object=AccidentResponse)

    GENERAL_ANALYSIS_PROMPT = SHARED_PROMPT + """
Then, analyze the filtered data for the following location:

{format_instructions}

Requirements:
- Include all years and severity levels in the data
- Return valid JSON format
- Ensure the JSON is complete and not truncated
- Make sure to close all brackets and braces properly

Note: If no accidents found, return: {{"accidents": []}}
"""
    # Generate summary of the analysis
    SUMMARY_PROMPT = """
Please provide a concise summary of the following accident analysis:

{}

Requirements:
1. Focus on key findings and trends
2. Highlight significant patterns or anomalies
3. Keep the summary clear and easy to understand
4. Maximum 3-4 sentences
"""

    # Create chart using the analysis results
    CHART_GENERATION_PROMPT = """
Create a line chart using Plotly based on the following analysis results:

Analysis Results:
{}

Requirements:
1. First, create a pandas DataFrame from the analysis results with columns:
   - year: The year of the accident
   - severity: The severity level (fatal, serious, light)
   - accident_count: Number of accidents

2. Then create a Plotly line chart with:
   - X-axis: Years
   - Y-axis: Number of accidents
   - Multiple lines for different severity levels
   - Markers for better visibility
   - Clear legend and labels
   - Interactive hover tooltips

Return the complete code that creates both the DataFrame and the chart:

import pandas as pd
import plotly.express as px
import json

# Parse the analysis results and create DataFrame
data = []
# Extract data from the analysis results string
# Example: data.append({'year': 2020, 'severity': 'fatal', 'accident_count': 5})

# Create DataFrame
df = pd.DataFrame(data)

# Verify DataFrame structure
required_columns = ['year', 'severity', 'accident_count']
if not all(col in df.columns for col in required_columns):
    raise ValueError(f"DataFrame must contain columns: {required_columns}")

# Create line chart
fig = px.line(
    df,
    x='year',
    y='accident_count',
    color='severity',
    title='Accidents Over Time by Severity',
    markers=True
)

# Update layout
fig.update_layout(
    xaxis_title='Year',
    yaxis_title='Number of Accidents',
    showlegend=True,
    legend_title='Severity',
    hovermode='x unified'
)

# Update traces
fig.update_traces(
    line=dict(width=2),
    marker=dict(size=8)
)
"""
    print(filter_level)
    print(location_name_dict)
    prompt = GENERAL_ANALYSIS_PROMPT.format(
        filter_level, 
        location_name_dict,
        format_instructions=parser.get_format_instructions()
    )
    
    output = st.session_state.sql_llm_agent.query_llm(prompt)
    
    # Parse the output
    try:
        parsed_output = parser.parse(output.content)
        # Convert to JSON string for chart generation
        json_str = parsed_output.json()
    except Exception as e:
        st.error(f"Error parsing JSON output: {str(e)}")
        return output, None

    summary_output = st.session_state.llm_model.invoke(SUMMARY_PROMPT.format(json_str))
    
    # chart_output = st.session_state.sql_llm_agent.query_llm(CHART_GENERATION_PROMPT.format(json_str))
    return summary_output, None
    # return summary_output, chart_output

def analyze_cause(filter_level: str, location_name_dict: dict):
    """Analyze accident causes."""

    ANALYSIS_CAUSE_PROMPT = SHARED_PROMPT + """
Then, analyze the filtered data for accident causes by:

1. Severity-based Aggregation:
   For each severity level (fatal, serious, light):
   - Group by location hierarchy and year
   - Calculate:
     * Number of accidents
     * Number of people involved
     * Number of vehicles involved
     * Average people per accident
     * Average vehicles per accident

2. Cause Analysis:
   - Top 3 accident causes with:
     * Number of accidents per cause
     * Number of people involved per cause
     * Number of vehicles involved per cause
     * Severity breakdown for each cause

3. Location-specific Analysis:
   - Accident distribution across hierarchy levels
   - Severity patterns by location
   - Cause patterns by location
   - High-risk locations identification

Output format:
1. Severity-based Summary:
   For each severity level (fatal, serious, light):
   - Per location level and year:
     * Total accidents
     * Total people involved
     * Total vehicles involved
     * Averages per accident

2. Cause Analysis:
   - For each severity level:
     * Top causes with:
       - Number of accidents
       - Number of people involved
       - Number of vehicles involved
       - Severity patterns

3. Location Insights:
   - Patterns across hierarchy levels
   - High-risk areas identification
   - Specific recommendations

Note: If no accidents found, state "No accident data available for this area"
"""

    prompt = ANALYSIS_CAUSE_PROMPT.format(filter_level, location_name_dict)
    return st.session_state.sql_llm_agent.query_llm(prompt)

def analyze_outcome(filter_level: str, location_name_dict: dict):
    """Analyze accident outcomes."""
    
    ANALYSIS_OUTCOME_PROMPT = SHARED_PROMPT + """
Then, analyze the filtered data for accident outcomes by:
1. Basic Statistics:
   - Total number of accidents
   - Accidents by severity:
     * Fatal accidents count and percentage
     * Serious accidents count and percentage
     * Light accidents count and percentage
   - Total number of casualties:
     * Fatalities count and percentage
     * Serious injuries count and percentage
     * Light injuries count and percentage
   - Total vehicles involved:
     * Average vehicles per accident
     * Distribution by vehicle type

2. Environmental Analysis:
   - Road conditions:
     * Count and percentage by condition
     * Severity correlation
   - Weather conditions:
     * Count and percentage by condition
     * Severity correlation
   - Time patterns:
     * Accidents by time of day
     * Accidents by day of week
     * Seasonal patterns

3. Risk Assessment:
   - Calculate risk score (1-10) based on:
     * Accident severity (weight: 0.4)
     * Number of casualties (weight: 0.3)
     * Accident frequency (weight: 0.3)
   - Compare risk scores across:
     * Different locations
     * Different conditions
     * Different time periods

Output format:
1. Summary Statistics:
   - Total accidents and casualties
   - Severity and injury distributions
   - Vehicle involvement statistics

2. Environmental Analysis:
   - Road and weather condition impacts
   - Temporal patterns
   - Risk factors identification

3. Risk Assessment:
   - Risk scores and comparisons
   - High-risk conditions
   - Safety recommendations

Note: Limit analysis to most informative insights if data volume is large
"""
    prompt = ANALYSIS_OUTCOME_PROMPT.format(filter_level, location_name_dict, INFERENCE_COLUMNS)
    return st.session_state.sql_llm_agent.query_llm(prompt)

def analyze_dataframe(filtered_locations, filter_level, filtered_df, analysis_type="general"):
    """Run AI-based accident analysis."""
    
    if analysis_type == "general":
        button_name = "General Analysis"
    elif analysis_type == "cause":
        button_name = "Cause Analysis"
    elif analysis_type == "outcome":
        button_name = "Outcome Analysis"
    else:
        return

    # Create a single level of columns
    button_col, popover_output_col, chart_output_col = st.columns([1.5, 1.5, 10])
    
    with button_col:
        if st.button("📈 " + button_name,
                     key="Accident " + str.capitalize(analysis_type) + " Analysis",
                     help=f"Process {analysis_type} accident data"):

            if filtered_df.empty:
                st.error("Please choose point on the map and filter level to proceed")
            elif not filter_level:
                st.error("Please choose a filter level.")
            elif not filtered_locations:
                st.error("Please choose locations to analyze.")
            else:
                with st.spinner(f"Performing {analysis_type} analysis..."):
                    output, chart_output = analyze_data(filtered_locations, filter_level, filtered_df, analysis_type)
                    
                    with popover_output_col:
                        with st.popover(f"Show {button_name}"):
                            st.markdown(output.content)
                    
                    if chart_output and chart_output.content:
                        with chart_output_col:
                            with st.popover(f"Show {button_name} chart"):
                                try:
                                    # Create a local namespace for the chart code
                                    local_vars = {}
                                    # Execute the chart code in the local namespace
                                    exec(chart_output.content, globals(), local_vars)
                                    # Get the figure from the local namespace
                                    fig = local_vars.get('fig')
                                    if fig:
                                        st.plotly_chart(fig, use_container_width=True)
                                    else:
                                        st.error("Failed to create chart")
                                except Exception as e:
                                    st.error(f"Error creating chart: {str(e)}")
                                    st.markdown(chart_output.content)

if __name__ == "__main__":
    initialize_session()
    analyze_general("TOWN", {'ROAD': {1: '652'}, 'SUBURB': {1: None}, 'TOWN': {1: 'Zikhron Yaakov'}, 'CITY': {1: None}, 'CITY_DISTRICT': {1: None}})
