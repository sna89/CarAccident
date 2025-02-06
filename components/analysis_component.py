from constants import INFERENCE_COLUMNS
from utils.sql_db import LangChainSQL
import streamlit as st


@st.cache_data()
def call_llm_query(filter_level, filtered_df):
    sql_llm = LangChainSQL()
    location_name = str.lower(list(filtered_df[filter_level].to_dict().values())[0])

    output = sql_llm.query_llm("Please find all accidents in the provided location: {}. "
                               "The location should be filtered based on the filter level (column) {}."
                               "This is the filtered location data: {}, which can help decide how to correctly "
                               "query the data. "
                               "For example, Haifa should be filtered as a city, while Nesher should be filtered as a town"
                               "If there are no results it might means that the query is incorrect and needed to "
                               "be fixed. "
                               "Please provide the top 5 reasons for accidents in the area, "
                               "regarding factors which contribute to the possibility of an "
                               "accident. Dont use columns from {} for this explanation."
                               "Try to aggregate informative columns to get insights regarding the data"
                               "In addition, please provide data regarding the number of accidents for each "
                               "relevant insight you mention. "
                               "Consider missing value while executing the sql query and retrieve all relevant "
                               "records, but When summarizing the accident cause, please only mention informative "
                               "cases, meaning, dont provide summary regarding missing or unknown data. "
                               "After summarizing the accidents causes, please write a short paragraph which "
                               "explains the main reasons for accidents in this area. From these reasons, "
                               "suggest steps and actions that can help reduce the number of car accidents in the area."
                               "If there are no accidents in this area, please state that your database does not "
                               "include data regarding accidents in this area".format(
        location_name, str.lower(filter_level), filtered_df.to_dict(), INFERENCE_COLUMNS))
    return output


def analyze_dataframe(filter_level, filtered_df):
    """Run AI-based accident analysis."""

    if st.button("📊 Analyze", key="analyze", help="Process accident data"):
        if filtered_df.empty:
            st.error("Please choose point on the map and filter level to proceed")
        elif not filter_level:
            st.error("Please choose a filter level.")
        else:
            output = call_llm_query(filter_level, filtered_df)
            with st.expander("Show Analysis"):
                st.write(output.content)
