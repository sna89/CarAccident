import uuid
from datetime import datetime
import pandas as pd
import math

from dotenv import load_dotenv
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from config import LLM_MODEL
from utils.llm_utils import get_llm_model
from utils.sql_db import SqlDb
from langchain_community.vectorstores import SupabaseVectorStore
import json
import warnings
from sqlalchemy.exc import SAWarning

warnings.filterwarnings(
    "ignore",
    message="Did not recognize type 'vector' of column 'embedding'",
    category=SAWarning
)

system_summarization_prompt = """You are a summarization engine for a question-answering system focused on reducing 
car accidents. Your task is to read through a provided text (extracted from PDF files) and generate a concise summary 
of the specified text, focusing only on relevant information for understanding accident trends. 
Don't add any details that are not specified directly in the text itself.
"""


class RAG:
    def __init__(self,
                 chunk_size=1000,
                 chunk_overlap=200,
                 top_k=5,
                 embedding_model_name="openai",
                 db_name="supabase",
                 llm_model=None,
                 llm_provider="openai",
                 llm_model_name=LLM_MODEL):
        load_dotenv()

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.embedding_model_name = embedding_model_name
        self.db_name = db_name

        self.embedding_model = self._get_embedding_model()
        self.db = self._get_db()
        # self.ollama_via_openai = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')

        llm_model = llm_model if llm_model else get_llm_model(llm_provider, llm_model_name)
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

        self.conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm_model,
                                                                        retriever=self._get_retriever().as_retriever(),
                                                                        memory=memory)

    def _get_embedding_model(self):
        if str.lower(self.embedding_model_name) == "openai":
            return OpenAIEmbeddings()
        else:
            return None

    def _get_db(self):
        if str.lower(self.db_name) == "supabase":
            return SqlDb()
        else:
            return None

    @staticmethod
    def _load_documents(folder_path):
        loader = PyPDFDirectoryLoader(folder_path)
        documents = loader.load()
        print(f"Loaded {len(documents)} documents")
        return documents

    # def _summarize_documents(self, documents):
    #     print("Summarizing documents:")
    #     for i, document in enumerate(documents):
    #         response = self.ollama_via_openai.chat.completions.create(
    #             model=self.summarization_model_name,
    #             messages=[
    #             {"role": "system", "content": system_summarization_prompt},
    #             {"role": "user", "content": document.page_content}
    #         ]
    #         )
    #         result = response.choices[0].message.content
    #         documents[i].page_content = result
    #
    #     if i % 10 == 0:
    #         print(f"Summarized {i} documents")
    #     return documents

    def _split_text(self, documents):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size,
                                                       chunk_overlap=self.chunk_overlap)
        chunk_list = text_splitter.split_documents(documents)
        print(f"Splitted to {len(chunk_list)} chunks")

        return chunk_list

    def _store_embeddings(self, chunk_list, table_name="documents", batch_size=100):
        total_chunks = len(chunk_list)
        num_batches = math.ceil(total_chunks / batch_size)
        for i in range(0, total_chunks, batch_size):
            print(f"Processing batch {i // batch_size} / {num_batches}")

            chunk_batch_list = chunk_list[i: i + batch_size]
            contents = [chunk.page_content for chunk in chunk_batch_list]
            sources = [json.dumps(chunk.metadata) for chunk in chunk_batch_list]

            try:
                embeddings = self.embedding_model.embed_documents(contents)
            except AttributeError:
                embeddings = [self.embedding_model.embed_query(content) for content in contents]

            ids = [str(uuid.uuid4()) for _ in range(len(chunk_batch_list))]

            df = pd.DataFrame({
                "id": ids,
                "content": contents,
                "embedding": embeddings,
                "metadata": sources
            })
            self.db.upload_table_from_pandas_df(table_name, df)

    def _get_retriever(self):
        vector_store = SupabaseVectorStore(
            client=self.db.client,
            table_name="documents",
            embedding=self.embedding_model
        )
        retriever = vector_store
        return retriever

    def load_and_embed_data(self, folder_path):
        documents = self._load_documents(folder_path)
        # documents = self._summarize_documents(documents)
        chunk_list = self._split_text(documents)
        self._store_embeddings(chunk_list)

    def retrieve_docs(self, query):
        retriever = self._get_retriever()
        retrieved_docs = retriever.similarity_search(query, k=self.top_k)
        return retrieved_docs

    def chat(self, question=""):
        result = self.conversation_chain.invoke(({"question": question}))
        return result["answer"]


if __name__ == "__main__":
    rag = RAG()
    # rag.load_and_embed_data(folder_path=r"C:\Users\sna89\PycharmProjects\car_accident_app\data\pdf")
    rag.chat("What are the main causes for car accidents, which are related to road infrastructure?")
