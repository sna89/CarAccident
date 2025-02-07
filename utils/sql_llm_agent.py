import os
from dotenv import load_dotenv
from langchain_community.agent_toolkits import SQLDatabaseToolkit

import ast
from langchain_openai import ChatOpenAI
from langchain import hub
from langgraph.prebuilt import create_react_agent
from utils.sql_db import SqlDb
from constants import INFERENCE_COLUMNS, LLM_MODEL

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.llms import HuggingFacePipeline


class SqlLLMAgent:
    def __init__(self):
        load_dotenv()

        self.db = SqlDb().db
        self.llm = self._get_llm_model(provider="OpenAI", model_name=LLM_MODEL)

        if self.llm:
            self.db_tools = self._get_db_tools()
            self.system_messege = self._get_system_messege()
            self.agent = self._get_agent()

    @staticmethod
    def _get_llm_model(provider, model_name):
        if provider == "OpenAI":
            openai_api_key = os.environ.get("OPENAI_API_KEY")
            llm = ChatOpenAI(model=model_name, openai_api_key=openai_api_key, temperature=0)
            return llm

        elif provider == "HuggingFace":
            token = os.environ.get("HUGGINGFACE_TOKEN")

            tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)
            model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=token, device_map="auto")

            # Create a Hugging Face pipeline for text generation.
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=256,
                temperature=0,
            )

            # Create the LangChain LLM wrapper.
            llm = HuggingFacePipeline(pipeline=pipe)
            return llm

        else:
            return None

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
