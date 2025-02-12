from dotenv import load_dotenv
from langchain_community.agent_toolkits import SQLDatabaseToolkit

import ast
from langchain import hub
from langgraph.prebuilt import create_react_agent

from utils.llm_utils import get_llm_model
from utils.sql_db import SqlDb
from config import INFERENCE_COLUMNS, LLM_MODEL


class SqlLLMAgent:
    def __init__(self):
        load_dotenv()

        self.db = SqlDb().db
        self.llm = get_llm_model(provider="openai", model_name=LLM_MODEL)

        if self.llm:
            self.db_tools = self._get_db_tools()
            self.system_messege = self._get_system_messege()
            self.agent = self._get_agent()

    def _get_db_tools(self):
        toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
        tools = toolkit.get_tools()
        return tools

    @staticmethod
    def _get_system_messege():
        prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")
        system_message = prompt_template.format(dialect="SQLite", top_k=5)
        return system_message

    def _get_agent(self):
        agent_executor = create_react_agent(
            self.llm, self.db_tools, state_modifier=self.system_messege
        )
        return agent_executor

    def query_directly(self, query="select * from accidents limit 10"):
        res = self.db.run(query)
        data = ast.literal_eval(res)
        return data

    def query_llm(self,
                  query="Go over all accidents from Simha Golan road in Haifa city, " \
                        "Please provide the top 5 reasons for accidents in the area, "
                        "regarding factors which contribute to the possibility of an accident in this area."
                        "Dont use columns from {} for this explanation."
                        "If there are no accidents in this area, please state that your database does not include "
                        "data regarding accidents in this area".format(INFERENCE_COLUMNS)):

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
    print(sql_db.query_llm())
