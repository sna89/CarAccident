import logging
from typing import List

from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm

from rag.rag_config import RAG_CONFIG

logger = logging.getLogger(__name__)


class DocumentProcessor:
    def __init__(self,
                 chunk_size: int = None,
                 chunk_overlap: int = None):
        """
        Initialize the document processor.

        Args:
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks for context preservation
        """
        self.chunk_size = chunk_size or RAG_CONFIG["chunk_size"]
        self.chunk_overlap = chunk_overlap or RAG_CONFIG["chunk_overlap"]
        logger.debug(f"Initialized DocumentProcessor with chunk_size={self.chunk_size}, "
                     f"chunk_overlap={self.chunk_overlap}")

    @staticmethod
    def load_documents(folder_path: str) -> List[Document]:
        print(f"Loading documents from {folder_path}")
        try:
            loader = PyPDFDirectoryLoader(folder_path)
            documents = loader.load()
            print(f"Loaded {len(documents)} documents")
            return documents
        except Exception as e:
            print(f"Error loading documents: {str(e)}")
            raise

    def split_documents(self, documents: List[Document]) -> List[Document]:
        print(f"Splitting {len(documents)} documents into chunks")
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            chunks = text_splitter.split_documents(documents)
            print(f"Split into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            print(f"Error splitting documents: {str(e)}")
            raise

    def process_documents(self, folder_path: str) -> List[Document]:
        documents = self.load_documents(folder_path)
        chunks = self.split_documents(documents)
        return chunks

    @staticmethod
    def build_knowledge_base(dataset):
        knowledge_base = [
            Document(page_content=doc["text"], metadata={"source": doc["source"]}) for doc in tqdm(dataset)
        ]
        return knowledge_base
