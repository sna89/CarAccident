import logging
from typing import  Optional

from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain, \
    BaseConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema.language_model import BaseLanguageModel
import time

class RAGConversation:
    """
    Manages conversational retrieval for question answering.
    """

    def __init__(self,
                 retriever,
                 llm_model: BaseLanguageModel,
                 memory: Optional[ConversationBufferMemory] = None):
        """
        Initialize the RAG conversation manager.

        Args:
            retriever: Document retriever to use for finding relevant context
            llm_model: Language model to use for answering questions
            memory: Memory instance to maintain conversation history
        """
        self.retriever = retriever
        self.llm_model = llm_model
        self.memory = memory or ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True
        )

        self.conversation_chain = self._setup_conversation_chain()
        print("Initialized RAGConversation with retriever and LLM")

    def _setup_conversation_chain(self) -> BaseConversationalRetrievalChain:
        try:
            chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm_model,
                retriever=self.retriever.as_retriever(),
                memory=self.memory
            )
            return chain
        except Exception as e:
            print(f"Error setting up conversation chain: {str(e)}")
            raise

    def answer_question(self, question: str, retry=3) -> str:
        for i in range(retry):
            try:
                result = self.conversation_chain({"question": question})
                answer = result.get("answer", "No answer found.")
                print(f"Question answered successfully")
                return answer
            except Exception as e:
                print(f"Error answering question: {str(e)}")
                error_msg = f"I encountered an error while processing your question: {str(e)}"
                print(error_msg)
                time.sleep(10)