import logging
from typing import List

import pandas as pd
import math

from langchain.docstore.document import Document
from langchain_community.vectorstores import SupabaseVectorStore

from rag.rag_config import RAG_CONFIG
from rag.rag_embedding_helper import EmbeddingHelper
from utils.sql_db import SqlDb

logger = logging.getLogger(__name__)


class VectorStoreHelper:
    def __init__(self,
                 db_name: str = None,
                 embedding_model_name: str = None,
                 table_name: str = None):
        self.db_name = db_name or RAG_CONFIG["db_name"]
        self.embedding_helper = EmbeddingHelper(embedding_model_name)
        self.table_name = table_name or RAG_CONFIG["documents_table"]
        self.db = self._get_db()
        self.vector_store = self._initialize_vector_store()
        logger.info(f"Initialized VectorStoreManager with db: {self.db_name}, "
                    f"table: {self.table_name}")

    def _get_db(self):
        try:
            if self.db_name.lower() == "supabase":
                return SqlDb()
            else:
                # Add support for other database backends here
                logger.warning(f"Unsupported database: {self.db_name}, using Supabase as fallback")
                return SqlDb()
        except Exception as e:
            logger.error(f"Error initializing database connection: {str(e)}")
            raise

    def _initialize_vector_store(self) -> SupabaseVectorStore:
        """
        Initialize the vector store with the current embedding model.

        Returns:
            Initialized vector store instance
        """
        if not self.embedding_helper:
            logger.error("Cannot initialize vector store: embedding model not provided")
            raise ValueError("Embedding model must be provided to initialize vector store")

        try:
            if self.db_name.lower() == "supabase":
                self.vector_store = SupabaseVectorStore(
                    client=self.db.client,
                    table_name=self.table_name,
                    embedding=self.embedding_helper.embedding_model
                )
                return self.vector_store
            else:
                # Add support for other database backends here
                logger.warning(f"Unsupported database: {self.db_name}, using Supabase as fallback")
                self.vector_store = SupabaseVectorStore(
                    client=self.db.client,
                    table_name=self.table_name,
                    embedding=self.embedding_helper.embedding_model
                )
                return self.vector_store

        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            raise

    def embed_and_store(self, chunk_list, table_name="documents", batch_size=100):
        total_chunks = len(chunk_list)
        num_batches = math.ceil(total_chunks / batch_size)
        for i in range(0, total_chunks, batch_size):
            print(f"Processing batch {i // batch_size} / {num_batches}")
            chunk_batch_list = chunk_list[i: i + batch_size]
            embed_dict = self.embedding_helper.embed_documents(chunk_batch_list)
            df = pd.DataFrame(embed_dict)
            self.db.upload_table_from_pandas_df(table_name, df)

    def get_retriever(self):
        return self.vector_store

    def retrieve_similar_documents(self, query: str, top_k: int = None) -> List[Document]:
        top_k = top_k or RAG_CONFIG["top_k"]
        try:
            retriever = self.get_retriever()
            documents = retriever.similarity_search(query, k=top_k)
            logger.info(f"Retrieved {len(documents)} documents for query: {query[:50]}...")
            return documents
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            raise