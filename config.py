import pandas as pd


DF_COLUMNS = ["road", "suburb", "town", "city", "city_district"]
INFERENCE_COLUMNS = ["num_nifgaim", "KLE_REHEV_HUZNU", "SUG_TEUNA", "SUG_EZEM", "MERHAK_EZEM", "LO_HAZA", "OFEN_HAZIYA", "HUMRAT_TEUNA"]
INITIAL_DF = pd.DataFrame(columns=[str.upper(col) for col in DF_COLUMNS])
LLM_MODEL = "gpt-4o"
SQL_LLM_MODEL = "gpt-3.5-turbo"
LLM_RAG_EVAL_MODEL = "gpt-4o-mini"
LLM_SUMMARIZATION_MODEL = "llama3.2:1b-instruct-q2_K"
MAX_LLM_TOKENS = 30000
