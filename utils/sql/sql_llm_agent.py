from dotenv import load_dotenv
from langchain_community.agent_toolkits import SQLDatabaseToolkit

import ast
from langchain import hub
from langgraph.prebuilt import create_react_agent

from utils.llm.model import get_llm_model
from utils.sql.sql_db import SqlDb
from config import INFERENCE_COLUMNS, SQL_LLM_MODEL


class SqlLLMAgent:
    def __init__(self):
        load_dotenv()

        self.db = SqlDb().db
        self.llm = get_llm_model(provider="openai", model_name=SQL_LLM_MODEL)

        if self.llm:
            self.db_tools = self._get_db_tools()
            self.agent = self._get_agent()

    def _get_db_tools(self):
        toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
        tools = toolkit.get_tools()
        return tools

    def _get_system_messege(self):
        prompt_template = """You are an intelligent agent designed to interact with a SQL database and perform data analysis using Python.

Given an input question, first examine the available tables in the database to understand the data. Then:

1. Create a syntactically correct {dialect} SQL query that retrieves only the **relevant columns** needed to answer the question.
2. Validate the SQL syntax before execution. If a query fails, rewrite and retry. Never use DML statements (INSERT, UPDATE, DELETE, DROP, etc.).
3. Once the query runs successfully, examine the results and return the **answer in natural language**, highlighting key insights.
4. If the question involves trends, comparisons, distributions, or other patterns, use the Python tool to generate a **clear and appropriate chart** (e.g., bar chart, line chart, pie chart, histogram). Choose the visualization type that best fits the data and the intent of the question.
5. When using Python, **read the query result** into a dataframe, and write the minimal code needed to generate the chart.
6. Return both the **chart** and a short textual **summary of the insights** it reveals.
7. Always double-check your logic and output before returning the final answer.

Remember:
- Never query all columns from a table — only the ones needed.
- Prioritize **clarity, insight, and relevance** in both SQL and visualizations.
- Think and reason like a data analyst.

To start, ALWAYS inspect the tables and their schema to understand what you can query.
""".format(
    dialect=self.db.dialect,
)

        return prompt_template

    def _get_agent(self):
        agent_executor = create_react_agent(
            self.llm, 
            self.db_tools,
            prompt=self._get_system_messege()
        )
        return agent_executor

    def query_directly(self, query="select * from accidents limit 10"):
        res = self.db.run(query)
        data = ast.literal_eval(res)
        return data

    def execute_sql_query(self, query: str) -> str:
        """Execute a sql query by utilizing the provided SQL tools.
        
        Args:
            query: The sql query to execute.
            
        Returns:
            The result of the executed task
        """
        assert self.llm, "No LLM supplied for task execution."

        try:
            for step in self.agent.stream(
                {"messages": [{"role": "user", "content": query}]},
                stream_mode="values",
            ):
                step["messages"][-1].pretty_print()
            
            return step["messages"][-1].content
        
        except Exception as e:
            print(f"Error executing task: {str(e)}")
            return None


if __name__ == "__main__":
    sql_db = SqlLLMAgent()
    print(sql_db.execute_task(task="Go over all accidents from Simha Golan road in Haifa city, " \
                        "Please provide the top 5 reasons for accidents in the area, "
                        "regarding factors which contribute to the possibility of an accident in this area."
                        "Dont use columns from {} for this explanation."
                        "If there are no accidents in this area, please state that your database does not include "
                        "data regarding accidents in this area".format(INFERENCE_COLUMNS)))
