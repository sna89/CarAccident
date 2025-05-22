"""Prompts and constants for accident analysis."""

# Base join and location filtering prompt
BASE_JOIN_PROMPT = """
Your task is to generate and validate a SQL query.

IMPORTANT:
Only return a JSON object in the format below. Do NOT include any explanations or comments. 

{base_query_format_instructions}

INPUT:
- Filter Level: {filter_level}
  This is the PRIMARY level for filtering. Filter according to the hierarchy levels, starting from this level and including all levels above it.
  Example: If filter_level is 'ROAD', filter according to the hierarchy: ROAD → SUBURB → CITY_DISTRICT → TOWN / CITY

- Location Dictionary: {location_dict}
  Contains location values for all levels. Use values according to the hierarchy rules below.

OBJECTIVE:
1. Use your SQL tools to inspect the schema of the `accidents` and `geographical_locations` tables.
2. Based on the schema, write a valid SQL query that defines a CTE named `base_data`.
3. The CTE should:
   - Join `accidents` and `geographical_locations` on `lat` and `lon`
   - Apply location-based filters following the hierarchy rules below

FILTERING RULES:
1. Hierarchy: `ROAD → SUBURB → CITY_DISTRICT → TOWN → CITY`
2. Always start from the specified filter_level and include levels above it
3. NEVER include any levels below the specified filter_level
4. All filter columns are located in the `geographical_locations` table
5. NEVER mix filters from different levels unless following the hierarchy rules above

Examples:
- If filter_level is 'ROAD': include ROAD, SUBURB, CITY_DISTRICT, TOWN, or CITY
- If filter_level is 'SUBURB': include SUBURB, CITY_DISTRICT, TOWN, or CITY
- If filter_level is 'CITY_DISTRICT': include CITY_DISTRICT, TOWN, or CITY
- If filter_level is 'TOWN': include TOWN
- If filter_level is 'CITY': include only CITY
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

CAUSE_SUMMARY_PROMPT = """
As an expert road engineer and car accident analyst, review the following accident data analysis results:

{data}

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

OUTCOME_SUMMARY_PROMPT = """
As an expert road safety engineer and urban planner, analyze the following accident outcome data:

{data}

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

Your task is to generate a single Plotly Express figure (`fig`) using this data dictionary:
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