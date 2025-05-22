"""Static SQL queries for accident analysis."""

# Define columns for cause analysis as a simple list
CAUSE_COLUMNS = [
    "SUG_YOM",
    "YOM_LAYLA",
    "RAMZOR",
    "ZURAT_DEREH",
    "SUG_DEREH",
    "ROHAV",
    "HAD_MASLUL",
    "RAV_MASLUL",
    "SIMUN_TIMRUR",
    "TEURA",
    "BAKARA",
    "LO_HAZA",
    "OFEN_HAZIYA",
    "MEKOM_HAZIYA",
    "KIVUN_HAZIYA",
    "MEHIRUT_MUTERET",
    "TKINUT",
    "MEZEG_AVIR",
    "PNE_KVISH"
]

# Define columns for outcome analysis as a simple list
OUTCOME_COLUMNS = [
    "SUG_TEUNA",
    "SUG_EZEM",
    "MERHAK_EZEM",
    "LO_HAZA",
    "OFEN_HAZIYA",
    "MEKOM_HAZIYA",
    "KIVUN_HAZIYA"
]

# Static general analysis query
GENERAL_ANALYSIS_QUERY = """
{base_query}

SELECT
    SHNAT_TEU as year,
    HUMRAT_TEUNA as severity,
    COUNT(*) as accident_count,
    SUM(Num_nifgaim) as people_involved,
    SUM(kle_rehev_huznu) as vehicles_involved,
    ROUND(AVG(Num_nifgaim), 2) as avg_people_per_accident,
    ROUND(AVG(kle_rehev_huznu), 2) as avg_vehicles_per_accident
FROM base_data
GROUP BY SHNAT_TEU, HUMRAT_TEUNA
ORDER BY 
    year ASC,
    severity DESC;
"""

# Generate cause analysis query
def generate_cause_analysis_parts():
    parts = []
    for column in CAUSE_COLUMNS:
        parts.append(f"""    -- {column} Analysis
    SELECT 
        SHNAT_TEU as year,
        HUMRAT_TEUNA as severity,
        '{column}' as category,
        {column} as value,
        COUNT(*) as count,
    FROM base_data
    GROUP BY SHNAT_TEU, HUMRAT_TEUNA, {column}""")
    return "\n\n    UNION ALL\n\n".join(parts)

# Generate outcome analysis query
def generate_outcome_analysis_parts():
    parts = []
    for column in OUTCOME_COLUMNS:
            parts.append(f"""    -- {column} Analysis
    SELECT 
        SHNAT_TEU as year,
        HUMRAT_TEUNA as severity,
        '{column}' as category,
        {column} as value,
        COUNT(*) as count,
        ROUND(AVG(Num_nifgaim), 1) as avg_casualties,
        ROUND(AVG(kle_rehev_huznu), 1) as avg_vehicles
    FROM base_data
    GROUP BY SHNAT_TEU, HUMRAT_TEUNA, {column}""")
    return "\n\n    UNION ALL\n\n".join(parts)

# Generate the full cause analysis query
CAUSE_ANALYSIS_QUERY = f"""
{{base_query}}
,
cause_analysis AS (
{generate_cause_analysis_parts()}
)
SELECT *
FROM cause_analysis
ORDER BY 
    year ASC,
    severity DESC,
    count DESC;
"""

# Generate the full outcome analysis query
OUTCOME_ANALYSIS_QUERY = f"""
{{base_query}}
,
outcome_analysis AS (
{generate_outcome_analysis_parts()}
)
SELECT *
FROM outcome_analysis
ORDER BY 
    year ASC,
    severity DESC,
    count DESC;
""" 