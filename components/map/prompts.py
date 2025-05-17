"""Prompts and constants for accident analysis."""

GENERAL_INSTRUCTIONS = """
You MUST use the available SQL tools to complete this task reliably:

- Use `list_tables` to see what tables exist.
- Use `get_table_info` to check the schema of relevant tables. Never assume column names.
- Use `query_sql` to validate and run your query. If it fails, revise and retry.

Only return the final, valid SQL query and result if requested.
"""

FINAL_OUTPUT_FORMAT = """
Return ONLY the valid SQL query as plain text.
- Do NOT include explanations, markdown, or any introductory sentence.
- Do NOT use `LIMIT` in the query.
- Your response must begin directly with the SQL statement
- Absolutely NO text or commentary before or after the query.
"""

# Column categories mapping
CAUSE_COLUMNS = {
    "SUG_YOM": "Type of Day",
    "YOM_LAYLA": "Day/Night",
    "RAMZOR": "Traffic Light",
    "ZURAT_DEREH": "Road Shape",
    "SUG_DEREH": "Road Type",
    "ROHAV": "Road Width",
    "HAD_MASLUL": "Single Lane",
    "RAV_MASLUL": "Multi Lane",
    "SIMUN_TIMRUR": "Traffic Sign",
    "TEURA": "Lighting",
    "BAKARA": "Visibility",
    "LO_HAZA": "Visibility Obstruction",
    "OFEN_HAZIYA": "Visibility Method",
    "MEKOM_HAZIYA": "Visibility Location",
    "KIVUN_HAZIYA": "Visibility Direction",
    "MEHIRUT_MUTERET": "Speed Limit",
    "TKINUT": "Road Marking",
    "MEZEG_AVIR": "Weather",
    "PNE_KVISH": "Road Surface"
}

# Outcome columns mapping
OUTCOME_COLUMNS = {
    "HUMRAT_TEUNA": "Accident Severity",
    "SUG_TEUNA": "Accident Type",
    "SUG_EZEM": "Injury Type",
    "MERHAK_EZEM": "Injury Location",
    "LO_HAZA": "Visibility Obstruction",
    "OFEN_HAZIYA": "Visibility Method",
    "MEKOM_HAZIYA": "Visibility Location",
    "KIVUN_HAZIYA": "Visibility Direction",
    "Num_nifgaim": "Number of Casualties",
    "kle_rehev_huznu": "Number of Vehicles Involved"
}

# Column categories for mapping to accident_data_mapping.json
CAUSE_COLUMN_CATEGORIES = {
    "SUG_YOM": "sug_yom_mapping",
    "YOM_LAYLA": "yom_layla_mapping",
    "RAMZOR": "ramzor_mapping",
    "ZURAT_DEREH": "zurat_derech_mapping",
    "SUG_DEREH": "sug_derech_mapping",
    "HAD_MASLUL": "had_maslul_mapping",
    "RAV_MASLUL": "rav_maslul_mapping",
    "MEHIRUT_MUTERET": "mehirut_muteret_mapping",
    "TKINUT": "tkinut_mapping",
    "ROHAV": "rohav_mapping",
    "SIMUN_TIMRUR": "simun_timrur_mapping",
    "TEURA": "teura_mapping",
    "BAKARA": "bakara_mapping",
    "MEZEG_AVIR": "mezeg_avir_mapping",
    "PNE_KVISH": "pne_kvish_mapping"
}

# Outcome categories for mapping to accident_data_mapping.json
OUTCOME_CATEGORIES = {
    "HUMRAT_TEUNA": "humra_mapping",
    "SUG_TEUNA": "sug_teuna_mapping",
    "SUG_EZEM": "sug_ezem_mapping",
    "MERHAK_EZEM": "merhak_ezem_mapping",
    "LO_HAZA": "lo_haza_mapping",
    "OFEN_HAZIYA": "ofen_haziya_mapping",
    "MEKOM_HAZIYA": "mekom_haziya_mapping",
    "KIVUN_HAZIYA": "kivun_haziya_mapping"
}

# Base join and location filtering prompt
BASE_JOIN_PROMPT = """
You are a SQL agent. Your job is to generate and validate a SQL query.

INPUT:
- Filter Level: {filter_level}
- Location Dictionary: {location_dict}

OBJECTIVE:
1. Use your SQL tools to inspect the schema of the `accidents` and `geographical_locations` tables.
2. Based on the schema, write a valid SQL query that defines a CTE named `base_data`.
3. The CTE should:
   - Join `accidents` and `geographical_locations` on `lat` and `lon`
   - Apply location-based filters using the provided location dictionary and the logic below


LOCATION FILTERING RULES:
- Hierarchy: `ROAD → SUBURB → CITY_DISTRICT → TOWN → CITY`
- If both `TOWN` and `CITY` are present, filter only by `TOWN`
- If a lower-level filter is used (like ROAD), include all higher levels above it
- All filter columns are located in the `geographical_locations` table

GENERAL INSTRUCTIONS:
{general_instructions}

FINAL OUTPUT FORMAT:
{final_output_format}
"""

# Analysis prompts
GENERAL_ANALYSIS_PROMPT = """
TASK:
Create and run a SQL query that aggregates accident data by year and severity level to provide comprehensive accident statistics.

Using the following base query, create a complete SQL query that analyzes general accident statistics:

Base Query:
{base_query}

REQUIREMENTS:
1. Use the base_data CTE from the provided query
2. Group results by:
   - SHNAT_TEUNA (year)
   - HUMRAT_TEUNA (severity level)
3. For each year and severity level combination, calculate:
   - Total number of accidents.
   - Total number of people involved. (num_nifgaim)
   - Total number of vehicles involved. (kle_rehev_huznu)
   - Average people per accident.
   - Average vehicles per accident
4. Use ROUND() to round the results to 2 decimal place.
5. Use column names from the given tables schema.
6. Before returning the final SQL query, confirm that the query contains all the required columns:
   - SHNAT_TEUNA
   - HUMRAT_TEUNA
   - Total number of accidents
   - Total number of people involved
   - Total number of vehicles involved
   - Average people per accident
   - Average vehicles per accident


Run the query and return the results in the following format:
{format_instructions}

GENERAL INSTRUCTIONS:
{general_instructions}

FINAL OUTPUT FORMAT:
{final_output_format}

DO NOT include any text before or after the JSON output. Return ONLY the JSON object.
"""

GENERAL_SUMMARY_PROMPT = """
Please provide a concise summary of the following accident data:

{data}

Requirements:
1. Focus on key findings and trends
2. Highlight significant patterns or anomalies
3. Keep the summary clear and easy to understand
4. Maximum 3-4 sentences
"""

CAUSE_ANALYSIS_PROMPT = """
You are an expert road safety engineer specializing in accident causation analysis. Your role is to identify and analyze the root causes of traffic accidents.

TASK:
Create a SQL query that analyzes accident causes using pivoting to show the distribution of categories across different criteria, grouped by year and severity level.

Using the following base query, create a complete SQL query that analyzes accident causes:

Base Query:
{base_query}

Columns to analyze:
{columns}

Category details:
{category_details}

REQUIREMENTS:
1. Use the base_data CTE from the provided query
2. For each column in the analysis:
   - Create a pivot table that shows:
     * Rows: SHNAT_TEUNA (year) and HUMRAT_TEUNA (severity level)
     * Columns: Categories from the specific column being analyzed
     * Values: Count of occurrences and percentage of total
   - Calculate for each category:
     * Count of occurrences (COUNT)
     * Percentage of total for that severity level (ROUND(COUNT * 100.0 / SUM(COUNT) OVER (PARTITION BY SHNAT_TEUNA, HUMRAT_TEUNA), 1))
3. Use UNION ALL to combine results from all pivot tables
4. Order results by:
   - Year (SHNAT_TEUNA) in ascending order
   - Severity level (HUMRAT_TEUNA) in descending order of severity
   - Count in descending order

Return the complete SQL query that will produce results in this format:
{format_instructions}

DO NOT include any text before or after the JSON output. Return ONLY the JSON object.
"""

CAUSE_SUMMARY_PROMPT = """
As an expert road engineer and car accident analyst, review the following accident data analysis results:

{query_results}

If there is no data in the results (empty table or no records), respond with:
"There is no accident data available for the given location and filter criteria. No analysis can be performed."

If there is data, provide a professional analysis report in the following format:

EXECUTIVE SUMMARY
----------------
A concise overview of the key findings and their implications for road safety, based on the specific patterns identified in the data.

MAIN FINDINGS
------------
1. Data-Driven Cause Analysis:
   - Top 5 most frequent accident causes with their exact percentages
   - Specific severity patterns for each cause (e.g., "X% of accidents in condition Y resulted in fatalities")
   - Correlation between specific conditions and accident severity
   - Statistical significance of identified patterns

2. Location-Specific Recommendations:
   Based on the exact data patterns shown, provide specific recommendations:
   - For each identified high-risk condition, specify the exact location or road section that needs attention
   - For each recommendation, cite the specific data points that support it
   - Prioritize recommendations based on the severity and frequency shown in the data
   - Include specific metrics or thresholds that should be monitored

CONCLUSION
----------
A brief summary of the most critical findings and recommendations, with specific reference to the data points that support each conclusion.

Note: Every recommendation must be directly supported by the data in the analysis. Avoid general recommendations that aren't backed by specific patterns in the data.
"""

OUTCOME_ANALYSIS_PROMPT = """
You are an expert emergency response and medical professional specializing in accident outcomes and injury analysis. Your role is to analyze the severity and impact of traffic accidents.

TASK:
Create a SQL query that analyzes accident outcomes using pivoting to show the distribution of categories across different criteria, including casualty and vehicle metrics, grouped by year and severity level.

Using the following base query, create a complete SQL query that analyzes accident outcomes:

Base Query:
{base_query}

Columns to analyze:
{columns}

Category details:
{category_details}

REQUIREMENTS:
1. Use the base_data CTE from the provided query
2. For each column in the analysis:
   - Create a pivot table that shows:
     * Rows: SHNAT_TEUNA (year) and HUMRAT_TEUNA (severity level)
     * Columns: Categories from the specific column being analyzed
     * Values: 
       - Count of occurrences
       - Percentage of total
       - Average casualties
       - Average vehicles involved
   - Calculate for each category:
     * Count of occurrences (COUNT)
     * Percentage of total for that severity level (ROUND(COUNT * 100.0 / SUM(COUNT) OVER (PARTITION BY SHNAT_TEUNA, HUMRAT_TEUNA), 1))
     * Average number of casualties (ROUND(AVG(Num_nifgaim), 1))
     * Average number of vehicles involved (ROUND(AVG(kle_rehev_huznu), 1))
3. Use UNION ALL to combine results from all pivot tables
4. Order results by:
   - Year (SHNAT_TEUNA) in ascending order
   - Severity level (HUMRAT_TEUNA) in descending order of severity
   - Count in descending order

Return the complete SQL query that will produce results in this format:
{format_instructions}

DO NOT include any text before or after the JSON output. Return ONLY the JSON object.
"""

OUTCOME_SUMMARY_PROMPT = """
As an expert road safety engineer and urban planner, analyze the following accident outcome data:

{query_results}

If there is no data in the results (empty table or no records), respond with:
"There is no accident data available for the given location and filter criteria. No analysis can be performed."

If there is data, provide a professional analysis report in the following format:

EXECUTIVE SUMMARY
----------------
A concise overview of the key findings and their implications for road safety, focusing on the specific patterns identified in the data regarding infrastructure and pedestrian behavior.

MAIN FINDINGS
------------
1. Data-Driven Safety Analysis:
   - Specific accident types with their exact severity distribution
   - Precise injury patterns and locations with statistical significance
   - Exact visibility-related issues and their impact on accident severity
   - Specific pedestrian behaviors that correlate with accident severity
   - Infrastructure vulnerabilities backed by accident frequency data

2. Evidence-Based Infrastructure Recommendations:
   Based on the exact data patterns shown, provide specific recommendations:
   - For each identified issue, specify the exact location or road section
   - Cite specific data points that support each recommendation
   - Include exact metrics that should be monitored
   - Prioritize based on the severity and frequency shown in the data

3. Data-Supported Pedestrian Safety Measures:
   - Educational initiatives targeting specific behaviors shown in the data
   - Infrastructure adaptations for documented high-risk locations
   - Enforcement recommendations based on actual violation patterns
   - Community engagement strategies focused on identified risk factors

CONCLUSION
----------
A brief summary of the most critical findings and recommendations, with specific reference to the data points that support each conclusion.

Note: Every recommendation must be directly supported by the data in the analysis. Avoid general recommendations that aren't backed by specific patterns in the data.
""" 


CHART_PROMPT = """
You are a Python data visualization assistant.

Your task is to generate a single Plotly Express figure (`fig`) using this pandas DataFrame `df`:
{data}

Instructions:
- Analyze the data and choose the most appropriate chart type from: 
  - Line chart: for time series or trend analysis
  - Bar chart: for comparing categories or values
  - Pie chart: for showing proportions of a whole
- Perform simple data transformation or aggregation if needed
- Ensure the figure is informative and well-labeled

Output must be a JSON object with the following structure:
{{
    "reasoning": {{
        "data_analysis": "Briefly describe the structure of the data",
        "visualization_choice": "Explain why this chart type was chosen",
        "data_processing": "List any transformations or aggregations performed",
        "edge_cases": ["Mention potential issues like missing values or low variance"]
    }},
    "code": "<REPLACE THIS with runnable Python code using Plotly Express that assigns the figure to a variable named 'fig'>"
}}

Requirements:
- Use only Plotly Express (no matplotlib, seaborn, or raw Plotly)
- Use the provided DataFrame `df` as-is or with minimal transformation
- Store the resulting figure in a variable named `fig`
- Include error handling to ensure the code runs safely
- Output valid JSON only
- Do not return placeholders. Replace them with real code.
"""