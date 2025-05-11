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
from utils.plotting import create_line_chart
import json

SHARED_PROMPT = """
First, generate a SQL query to filter accident data with the following location filter:

Filter Level: {}
Location: {}

You need both accident data and location data to generate the query.
Location Hierarchy Logic:
- Each location belongs to a hierarchy (Road → Suburb → City District -> Town) or (Road → Suburb → City District -> City)
- If both Town and City are provided, the query should filter based on the Town
- All locations are columns in the accidents table
- Filter should consider the full context of the location in its hierarchy
- A Higher level filter (e.g., City) in hierarchy does not need lower level details.
- A Lower level filter (e.g., Road) in hierarchy should include their parent locations
- Each level must maintain its relationship with parent locations

Location Hierarchy Examples:
1. City Level:
   - Example: {{'CITY': {{1: 'Tel Aviv'}}, 'CITY_DISTRICT': {{1: None}}, 'TOWN': {{1: None}}, 'SUBURB': {{1: None}}, 'ROAD': {{1: None}}}}
   - Filter Level: "CITY"
   - Query should: Filter on city column
   - Group by: city and year
   - No need for lower level details

2. City District Level:
   - Example: {{'CITY': {{1: 'Tel Aviv'}}, 'CITY_DISTRICT': {{1: 'Central'}}, 'TOWN': {{1: None}}, 'SUBURB': {{1: None}}, 'ROAD': {{1: None}}}}
   - Filter Level: "CITY_DISTRICT"
   - Query should: Filter on city and city_district columns or city_district and Town column
   - Group by: city, city_district, and year
   - Must include city context

3. Town Level:
   - Example: {{'CITY': {{1: None}}, 'CITY_DISTRICT': {{1: None}}, 'TOWN': {{1: 'Zikhron Yaakov'}}, 'SUBURB': {{1: None}}, 'ROAD': {{1: None}}}}
   - Filter Level: "TOWN"
   - Query should: Filter on town column
   - Group by: town and year
   - Independent of city/city_district

4. Suburb Level:
   - Example: {{'CITY': {{1: 'Haifa'}}, 'CITY_DISTRICT': {{1: None}}, 'TOWN': {{1: 'Haifa'}}, 'SUBURB': {{1: 'Neve Sha'anan'}}, 'ROAD': {{1: None}}}}
   - Filter Level: "SUBURB"
   - Query should: Filter on suburb, and town, or city columns
   - Group by: suburb, town or city, and year
   - Must include town or city context, not both

5. Road Level:
   - Example: {{'CITY': {{1: 'Haifa'}}, 'CITY_DISTRICT': {{1: None}}, 'TOWN': {{1: 'Haifa'}}, 'SUBURB': {{1: 'Neve Sha'anan'}}, 'ROAD': {{1: 'Herzl'}}}}
   - Filter Level: "ROAD"
   - Query should: Filter on road, suburb, town, aornd city columns
   - Group by: road, suburb, town or city, and year
   - Must include all parent locations (suburb, town or city)

Query Structure:
1. Base Query:
   - Start with accidents table
   - Filter on appropriate location columns based on filter level
   - Select all relevant columns
   - DO NOT use LIMIT clause

2. Aggregation Query:
   - Use the base query as a CTE (WITH clause)
   - Group by location hierarchy levels and year
   - Calculate aggregations at each level
   - Include all necessary metrics
   - DO NOT use LIMIT clause
   - Use ROUND() for all division calculations (e.g., averages)

Requirements for data filtering:
1. Select all relevant columns
2. Apply location filters considering the hierarchy
3. Filter on appropriate location columns
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
        output, fig = analyze_general(filter_level, location_name_dict)
        return output, fig
    elif analysis_type == "cause":
        output, _ = analyze_cause(filter_level, location_name_dict)
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
        json_str = parsed_output.model_dump_json()
        
        # Create chart using our plotting utility
        # Convert the parsed output to the format expected by create_line_chart
        chart_data = [
            {
                'year': accident.year,
                'severity': accident.severity,
                'accident_count': accident.accident_count
            }
            for accident in parsed_output.accidents
        ]
        
        if chart_data:
            fig = create_line_chart(
                data=chart_data,
                x_column='year',
                y_column='accident_count',
                group_column='severity',
                title='Accidents Over Time by Severity',
                x_label='Year',
                y_label='Number of Accidents',
                group_label='Accident Severity'
            )
        else:
            fig = None
            
    except Exception as e:
        st.error(f"Error parsing JSON output: {str(e)}")
        return output, None

    summary_output = st.session_state.llm_model.invoke(SUMMARY_PROMPT.format(json_str))
    
    return summary_output, fig

CAUSE_COLUMNS = [
    "SUG_YOM",      # Type of Day
    "YOM_LAYLA",    # Day/Night
    "RAMZOR",       # Traffic Light
    "SUG_TEUNA",    # Accident Type
    "ZURAT_DEREH",  # Road Shape
    "SUG_DEREH",    # Road Type
    "ROHAV",        # Road Width
    "HAD_MASLUL",   # Single Lane
    "RAV_MASLUL",   # Multi Lane
    "SIMUN_TIMRUR", # Traffic Sign
    "TEURA",        # Lighting
    "BAKARA",       # Visibility
    "LO_HAZA",      # Visibility Obstruction
    "OFEN_HAZIYA",  # Visibility Method
    "MEKOM_HAZIYA", # Visibility Location
    "KIVUN_HAZIYA", # Visibility Direction
    "MEHIRUT_MUTERET", # Speed Limit
    "TKINUT",       # Road Marking
    "MEZEG_AVIR",   # Weather
    "PNE_KVISH"     # Road Surface
]

def analyze_cause(filter_level: str, location_name_dict: dict):
    """Analyze accident causes by severity level."""

    # Load the mapping file
    with open('data/accident_data_mapping.json', 'r') as f:
        mappings = json.load(f)

    # Create a mapping of column names to their categories
    column_categories = {
        "SUG_YOM": mappings["sug_yom_mapping"],
        "YOM_LAYLA": mappings["yom_layla_mapping"],
        "RAMZOR": mappings["ramzor_mapping"],
        "SUG_TEUNA": mappings["sug_teuna_mapping"],
        "ZURAT_DEREH": mappings["zurat_derech_mapping"],
        "SUG_DEREH": mappings["sug_derech_mapping"],
        "HAD_MASLUL": mappings["had_maslul_mapping"],
        "RAV_MASLUL": mappings["rav_maslul_mapping"],
        "MEHIRUT_MUTERET": mappings["mehirut_muteret_mapping"],
        "TKINUT": mappings["tkinut_mapping"],
        "ROHAV": mappings["rohav_mapping"],
        "SIMUN_TIMRUR": mappings["simun_timrur_mapping"],
        "TEURA": mappings["teura_mapping"],
        "BAKARA": mappings["bakara_mapping"],
        "MEZEG_AVIR": mappings["mezeg_avir_mapping"],
        "PNE_KVISH": mappings["pne_kvish_mapping"]
    }

    ANALYSIS_CAUSE_PROMPT = SHARED_PROMPT + """
Based on the filtered data, create a single pivot table that counts occurrences of each category in the following columns, grouped by severity level (fatal, serious, light):

{columns}

For each column, count occurrences of each category:
{category_details}

Output format:
Create a single table with the following structure:
- First column: Severity Level (fatal, serious, light)
- Second column: Cause Column Name
- Third column: Category Name
- Fourth column: Count of occurrences
- Fifth column: Percentage of total for that severity level

Note: Calculate percentages within each severity level (e.g., all percentages for Fatal should sum to 100%)
""".format(
        columns="\n".join([f"- {col}" for col in column_categories.keys()]),
        category_details="\n".join([
            f"* {col} categories: {', '.join(cats.values())}"
            for col, cats in column_categories.items()
        ])
    )

    prompt = ANALYSIS_CAUSE_PROMPT.format(filter_level, location_name_dict)
    output = st.session_state.sql_llm_agent.query_llm(prompt)

    # Add summary analysis
    SUMMARY_PROMPT = """
As an expert road engineer and car accident analyst, review the following accident data analysis results:

{query_results}

If there is no data in the results (empty table or no records), respond with:
"There is no accident data available for the given location and filter criteria. No analysis can be performed."

If there is data, provide a professional analysis report in the following format:

EXECUTIVE SUMMARY
----------------
A concise overview of the key findings and their implications for road safety.

MAIN FINDINGS
------------
1. Overall Top 5 Causes:
   - Ranked list of the most significant accident causes
   - Total count and percentage for each cause
   - Impact analysis of each cause on road safety
   - Summary of how these causes affect accident severity and frequency

2. Infrastructure Improvement Recommendations:
   Based on the identified causes, provide specific, actionable recommendations:
   - Road design modifications
   - Traffic control enhancements
   - Safety feature additions
   - Maintenance priorities
   - Implementation timeline suggestions

CONCLUSION
----------
A brief summary of the most critical findings and recommendations, emphasizing the most urgent safety concerns.

Note: Present the information in a clear, professional manner suitable for a technical report. Focus on actionable insights and practical solutions.
""".format(query_results=output)

    summary_output = st.session_state.llm_model.invoke(SUMMARY_PROMPT)
    return summary_output, None

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
                    output, fig = analyze_data(filtered_locations, filter_level, filtered_df, analysis_type)
                    
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

if __name__ == "__main__":
    initialize_session()
    summary_output, fig = analyze_general("TOWN", {'TOWN': {1: 'Zikhron Yaakov'}, 'CITY': {1: None}, 'CITY_DISTRICT': {1: None}})
    print(summary_output)
