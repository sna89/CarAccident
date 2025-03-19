from typing import List, Dict, Any

from dotenv import load_dotenv

from config import MAX_LLM_TOKENS
from rag.utils import count_tokens, shorten_prompt
from rag.rag_config import RAG_CONFIG
from rag.document_processor import DocumentProcessor
from rag.rag_embedding_helper import EmbeddingHelper
from rag.vector_store_helper import  VectorStoreHelper
from rag.rag_conversation import RAGConversation
from utils.llm_utils import get_llm_model


class RAG:
    def __init__(self, config: Dict[str, Any] = None):

        load_dotenv()
        self.config = config or RAG_CONFIG

        # Initialize components
        self.document_processor = DocumentProcessor(
            chunk_size=self.config["chunk_size"],
            chunk_overlap=self.config["chunk_overlap"]
        )

        self.embedding_helper = EmbeddingHelper(
            embedding_model_name=self.config["embedding_model_name"]
        )

        self.vector_store_manager = VectorStoreHelper(
            db_name=self.config["db_name"],
            embedding_model_name=self.config["embedding_model_name"],
            table_name=self.config["documents_table"]
        )

        self.llm_model = get_llm_model(
            provider=self.config["llm_provider"],
            model_name=self.config["llm_model_name"]
        )

        # Initialize conversation after setting up the vector store
        self.conversation = None
        self.setup_conversation()

    def setup_conversation(self):
        """
        Set up the conversation component with the retriever.
        """
        retriever = self.vector_store_manager.get_retriever()
        self.conversation = RAGConversation(
            retriever=retriever,
            llm_model=self.llm_model
        )

    def load_and_index_documents(self, folder_path: str) -> None:
        chunks = self.document_processor.process_documents(folder_path)

        self.vector_store_manager.embed_and_store(
            chunks,
            batch_size=self.config["batch_size"]
        )

    def retrieve_documents(self, query: str, top_k: int = None) -> List:
        top_k = top_k or self.config["top_k"]
        return self.vector_store_manager.retrieve_similar_documents(query, top_k)

    def chat(self, question: str = "") -> str:
        if not self.conversation:
            self.setup_conversation()

        return self.conversation.answer_question(question)



# Example usage
if __name__ == "__main__":
    rag = RAG()
    # rag.load_and_index_documents(folder_path="./data/pdf")

    # Example query
    answer = rag.chat("What are the main causes for car accidents, which are related to road infrastructure?")
    print(answer)