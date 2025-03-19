import uuid
from typing import List, Dict, Any

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain_openai import OpenAIEmbeddings

from rag.rag_config import RAG_CONFIG


class EmbeddingHelper:
    def __init__(self, embedding_model_name: str = None):
        self.embedding_model_name = embedding_model_name or RAG_CONFIG["embedding_model_name"]
        self.embedding_model = self._get_embedding_model()
        print(f"Initialized EmbeddingManager with model: {self.embedding_model_name}")

    def _get_embedding_model(self) -> Embeddings:
        try:
            if self.embedding_model_name.lower() == "openai":
                return OpenAIEmbeddings()
            else:
                # Add support for other embedding models here
                print(f"Unsupported embedding model: {self.embedding_model_name}, "
                               f"using OpenAI as fallback")
                return OpenAIEmbeddings()
        except Exception as e:
            print(f"Error initializing embedding model: {str(e)}")
            raise

    def embed_documents(self,
                        documents: List[Document]) -> Dict[str, Any]:
        contents = [doc.page_content for doc in documents]
        metadata = [doc.metadata for doc in documents]
        embeddings = self.compute_embeddings(contents)
        ids = [str(uuid.uuid4()) for _ in range(len(contents))]

        return {
            "ids": ids,
            "contents": contents,
            "embeddings": embeddings,
            "metadata": metadata
        }

    def compute_embeddings(self, content):
        return self.embedding_model.embed_documents(content)