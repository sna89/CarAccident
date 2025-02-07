from constants import INFERENCE_COLUMNS
from utils.sql_llm_agent import SqlLLMAgent
import streamlit as st

ANALYSIS_CAUSE_PROMPT = """
    Please find all accidents in the provided locations: {}.
    The locations should be filtered based on the filter level (column) {}.
    Please provide the top 3 reasons for accidents in each location,
    regarding factors which contribute to the possibility of an
    accident. Dont use columns from {} for this explanation.
    Aggregate informative columns to get insights regarding the data
    In addition, please make sure you provide the number of accidents for each location
    and insight.
    Consider missing value while executing the sql query and retrieve all relevant
    records, but When summarizing the accident cause, please only mention informative
    cases, meaning, do not provide summary regarding missing or unknown data.
    After summarizing the accidents causes, please write a short paragraph which
    explains the main reasons for accidents in each location separately. From these
    reasons, suggest steps and actions that can help reduce the number of car accidents
    in each location. If no accidents found on the database, please state that your
    database does not include data regarding accidents in this area"""

ANALYSIS_OUTCOME_PROMPT = """
    Please find all accidents in the provided locations: {}. " The locations should be 
    filtered based on the filter level (column) {} only." Please analyze and provide insights regrading the accident 
    outcome, with respect to the following columns: {}. 
    You should aggregate informative columns to get insights regarding the data. 
    In addition, please make sure you provide the total number of accidents for each insight and location. 
    Consider missing value while executing the sql query and retrieve all relevant records, but when summarizing the 
    accident cause, please only mention informative cases, meaning, do not provide summary regarding missing or unknown 
    data. After summarizing the accidents outcome, please write a short paragraph which explains the main accidents 
    outcome in each location separately. In addition, calculate and present a normalized risk score (from 1 to 10), 
    based on the accident severity (humrat teuna) and the number of accidents, compared to accidents in other similar 
    areas. For example, if you calculating the risk score for Ramot Remez suburb, compare it to a higher level in the 
    heirarchy. It can be city district or city. If the risk score is calculated to a city, compare it with a risk score 
    of a different city in the area. For example, for Haifa, you can compare it to Hadera or Karmiel. If the number of 
    responses is too big, limit it in a way that will let you suggest informative insights"""


@st.cache_data()
def call_llm_query(filtered_locations, filter_level, filtered_df, analysis_type="cause"):
    sql_llm = SqlLLMAgent()
    location_name_dict = filtered_df[filtered_df[filter_level].isin(filtered_locations)].to_dict()

    if analysis_type == "cause":
        prompt = ANALYSIS_CAUSE_PROMPT.format(location_name_dict, str.lower(filter_level), INFERENCE_COLUMNS)
    elif analysis_type == "outcome":
        prompt = ANALYSIS_OUTCOME_PROMPT.format(location_name_dict, str.lower(filter_level), INFERENCE_COLUMNS)
    else:
        return

    output = sql_llm.query_llm(prompt)
    return output


def analyze_dataframe(filtered_locations, filter_level, filtered_df, analysis_type="cause"):
    """Run AI-based accident analysis."""

    if analysis_type == "cause":
        button_name = "Cause Analysis"
    elif analysis_type == "outcome":
        button_name = "Outcome Analysis"
    else:
        return

    if st.button("📊 " + button_name,
                 key="Accident " + str.capitalize(analysis_type) + " Analysis",
                 help=f"Process {analysis_type} accident data"):

        if filtered_df.empty:
            st.error("Please choose point on the map and filter level to proceed")
        elif not filter_level:
            st.error("Please choose a filter level.")
        elif not filtered_locations:
            st.error("Please choose locations to analyze.")
        else:
            output = call_llm_query(filtered_locations, filter_level, filtered_df, analysis_type)
            with st.expander("Show Analysis"):
                st.write(output.content)
