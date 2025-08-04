"""RAG-based question answering system with conversation history.

This module implements a Retrieval Augmented Generation (RAG) system that combines
document retrieval with language model generation to provide accurate, contextual
answers. It uses FAISS for document similarity search, HuggingFace models for
text generation, and LangGraph for workflow orchestration.

Main Components:
    - AssistantInit: Main class implementing the RAG pipeline
    - State: TypedDict defining the workflow state structure

Environment Variables:
    - LLM_ID: HuggingFace model ID for text generation
    - EMBEDDING_ID: Model ID for text embeddings
    - VECTORSTORE_PATH: Path to FAISS vector store
    - PROMPT_PATH: Path to system prompt configuration
"""

import os
import json

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, trim_messages
from langchain_core.documents import Document
from langchain.vectorstores import FAISS
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEndpoint,
    HuggingFaceEmbeddings,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict, Optional, Annotated, Union
from operator import add
from dotenv import load_dotenv

load_dotenv("./config/.env")


class State(TypedDict):
    """Represents the state of the RAG (Retrieval Augmented Generation) pipeline.

    This TypedDict defines the structure of the state object that flows through the
    LangGraph workflow. It contains the user's question, retrieved context documents,
    generated answer, and conversation history.

    Attributes:
        question (str): The current user question or query
        context (List[Document]): List of retrieved documents for context
        answer (str): The generated answer from the LLM
        history (Annotated[Optional[List[str]], add]): Conversation history with
            accumulator annotation for maintaining chat history across turns
    """

    question: str
    context: List[Document]
    answer: str
    history: Annotated[Optional[List[str]], add]


class AssistantInit:
    """A RAG (Retrieval Augmented Generation) based assistant using LangGraph workflow.

    This class implements a question-answering system that combines document retrieval
    with language model generation. It uses FAISS for similarity search, HuggingFace
    for embeddings and text generation, and LangGraph for workflow management.

    Key Components:
        - FAISS vectorstore for efficient similarity search
        - HuggingFace model for text generation
        - LangGraph for orchestrating the RAG workflow
        - Conversation history management with token-based trimming
        - Environment-based configuration for models and paths

    Example:
        >>> assistant = AssistantInit()
        >>> response = assistant.get_answer("What is Article 370?")
        >>> print(response["answer"])
    """

    def __init__(self):
        # Initialize the LLM with the model name from environment variable
        self.model_name = os.getenv("LLM_ID", "meta-llama/Meta-Llama-3-8B-Instruct")
        llm = HuggingFaceEndpoint(
            model=self.model_name,
            task="text-generation",
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.03,
        )
        self.chat_model = ChatHuggingFace(llm=llm)
        embeddings = HuggingFaceEmbeddings(
            model_name=os.getenv(
                "EMBEDDING_ID", "sentence-transformers/all-mpnet-base-v2"
            )
        )
        self.vectorstore = FAISS.load_local(
            os.getenv("VECTORSTORE_PATH", "./vectorstore"),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        # Load the prompt from the JSON file
        with open(os.getenv("PROMPT_PATH", "./config/prompt.json"), "r") as f:
            prompt_data = json.load(f)

        self.prompt_template = ChatPromptTemplate(
            [
                SystemMessage(prompt_data["prompt"]),
                ("human", "{history}"),
                ("human", "{question}"),
                ("human", "{context}"),
            ]
        )
        self.trimmer = trim_messages(
            max_tokens=700,
            strategy="last",
            token_counter=self.chat_model,
            allow_partial=False,
            start_on="human",
        )
        graph_builder = StateGraph(State)
        graph_builder.add_node("retrieve", self.retrieve)
        graph_builder.add_node("generate", self.generate)
        graph_builder.add_edge("retrieve", "generate")
        graph_builder.add_edge(START, "retrieve")
        memory = MemorySaver()
        self.graph = graph_builder.compile(checkpointer=memory)
        self.config = {"configurable": {"thread_id": "1"}}

    def retrieve(self, state: State) -> dict[str, List[Document]]:
        """Retrieves relevant documents from the vector store based on the input question.

        This function performs a similarity search in the FAISS vector store using the question
        provided in the state. It returns the top 2 most semantically similar documents that can
        be used as context for answering the question.

        Args:
            state (State): A TypedDict containing:
                - question (str): The user's input question
                - context (List[Document]): Previous context (if any)
                - answer (str): Previous answer (if any)
                - history (Optional[List[str]]): Conversation history

        Returns:
            dict[str, List[Document]]: A dictionary containing:
                - context: List of retrieved Document objects, each containing:
                    - page_content: The actual text content
                    - metadata: Associated metadata like source, page numbers, etc.

        Example:
            >>> state = {"question": "What is Article 370?", "context": [], "answer": "", "history": None}
            >>> result = assistant.retrieve(state)
            >>> # Returns {"context": [Document1, Document2]} where Document1 and Document2
            >>> # are the most relevant documents from the vector store
        """
        retrieved_docs = self.vectorstore.similarity_search(state["question"], k=2)
        return {"context": retrieved_docs}

    def generate(self, state: State) -> dict[str, Union[str, List[str]]]:
        """Generates an AI response using retrieved context and conversation history.

        This function combines retrieved documents with the user's question and conversation
        history to generate a contextually relevant response. It manages conversation history
        length using a token-based trimmer.

        Args:
            state (State): A TypedDict containing:
                - question (str): The user's question
                - context (List[Document]): Retrieved documents for context
                - answer (str): Previous answer if any
                - history (Optional[List[str]]): Previous conversation history

        Returns:
            dict: A dictionary containing:
                - answer (str): The AI-generated response
                - history (List[str]): Updated conversation history with new Q&A pair
        """
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        # trim history
        history = state.get("history", [])
        self.trimmer.invoke(history)

        messages = self.prompt_template.invoke(
            {
                "question": state["question"],
                "context": docs_content,
                "history": "\n\n".join(chat for chat in (history or [])),
            }
        )
        response = self.chat_model.invoke(messages)
        return {
            "answer": response.content,
            "history": [
                "Human: " + state["question"],
                "Assistant: " + response.content,  # type: ignore
            ],
        }

    def get_answer(self, query: str) -> dict[str, Union[str, List[str]]]:
        """Processes a user query through the RAG pipeline to generate an answer.

        This function is the main entry point for question answering. It coordinates the
        document retrieval and answer generation process through a LangGraph workflow.
        The conversation history is automatically managed through the graph's checkpointer.

        Args:
            query (str): The user's question or prompt

        Returns:
            dict: A dictionary containing:
                - answer (str): The AI-generated response
                - history (List[str]): The updated conversation history
                - context (List[Document]): The retrieved documents used for the answer

        Example:
            >>> response = assistant.get_answer("What are the fundamental rights in India?")
            >>> print(response["answer"])  # Prints the AI's response
        """
        response = self.graph.invoke(
            {"question": query},  # type: ignore
            config=self.config,  # type: ignore
        )
        return response


assistant = AssistantInit()
