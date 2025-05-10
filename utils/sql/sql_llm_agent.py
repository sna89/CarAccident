from dotenv import load_dotenv
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_experimental.tools import PythonREPLTool

import ast
from langchain import hub
from langgraph.prebuilt import create_react_agent

from utils.llm.model import get_llm_model
from utils.sql.sql_db import SqlDb
from config import INFERENCE_COLUMNS, LLM_MODEL


class SqlLLMAgent:
    def __init__(self):
        load_dotenv()

        self.db = SqlDb().db
        self.llm = get_llm_model(provider="openai", model_name=LLM_MODEL)

        if self.llm:
            self.db_tools = self._get_db_tools()
            self.python_tool = self._get_python_tool()
            self.system_messege = self._get_system_messege()
            self.agent = self._get_agent()

    def _get_db_tools(self):
        toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
        tools = toolkit.get_tools()
        return tools

    def _get_python_tool(self):
        """Create a Python REPL tool for executing Python code."""
        return PythonREPLTool()

    @staticmethod
    def _get_system_messege():
        prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")
        system_message = prompt_template.format(dialect="SQLite", top_k=5)
        # Add Python REPL instructions to the system message
        system_message += """
You also have access to a Python REPL tool that can execute Python code. Use this tool when you need to:
1. Process or transform data after querying the database
2. Create visualizations or data analysis
3. Perform complex calculations or data manipulations
4. Format or structure the output in a specific way

When using the Python REPL tool:
- Always import required libraries first
- Handle errors appropriately
- Format the output in a clear and readable way
- Use pandas for data manipulation when working with query results
"""
        return system_message

    def _get_agent(self):
        # Combine SQL tools with Python REPL tool
        all_tools = self.db_tools + [self.python_tool]
        agent_executor = create_react_agent(
            self.llm, all_tools, state_modifier=self.system_messege
        )
        return agent_executor

    def query_directly(self, query="select * from accidents limit 10"):
        res = self.db.run(query)
        data = ast.literal_eval(res)
        return data

    def query_llm(self,
                  query):

        assert self.llm, "No LLM supplied for sql query execution."

        events = self.agent.stream(
            {"messages": [("user", query)]},
            stream_mode="values",
        )
        for event in events:
            event["messages"][-1].pretty_print()

        return event["messages"][-1]


if __name__ == "__main__":
    sql_db = SqlLLMAgent()
    print(sql_db.query_llm(query="Go over all accidents from Simha Golan road in Haifa city, " \
                        "Please provide the top 5 reasons for accidents in the area, "
                        "regarding factors which contribute to the possibility of an accident in this area."
                        "Dont use columns from {} for this explanation."
                        "If there are no accidents in this area, please state that your database does not include "
                        "data regarding accidents in this area".format(INFERENCE_COLUMNS)))
