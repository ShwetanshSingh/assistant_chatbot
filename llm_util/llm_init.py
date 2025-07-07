import os
import json

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage, trim_messages
from langchain_core.documents import Document
from langchain.vectorstores import FAISS
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from typing_extensions import List, TypedDict
from dotenv import load_dotenv

load_dotenv("./config/.env")

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

class AssistantInit:
    def __init__(self):
        # Initialize the LLM with the model name from environment variable
        self.model_name = os.getenv("LLM_ID", "meta-llama/Meta-Llama-3-8B-Instruct")
        llm = HuggingFaceEndpoint(
                model=self.model_name,
                task="text-generation",
                max_new_tokens=512,
                do_sample=False,
                repetition_penalty=1.03
        )
        self.chat_model = ChatHuggingFace(
            llm=llm
        )
        embeddings = HuggingFaceEmbeddings(
            model_name=os.getenv("EMBEDDING_ID", "sentence-transformers/all-mpnet-base-v2")
        )
        self.vectorstore = FAISS.load_local(
            os.getenv("VECTORSTORE_PATH", "./vectorstore"),
            embeddings,
            allow_dangerous_deserialization=True
        )
        # Load the prompt from the JSON file
        with open(os.getenv("PROMPT_PATH", "./config/prompt.json"), "r") as f:
            prompt_data = json.load(f)

        self.prompt_template = ChatPromptTemplate(
            [
                SystemMessage(prompt_data["prompt"]),
                ("human", "{question}"),
                ("human", "{context}")
            ]
        )
        self.trimmer = trim_messages(
            max_tokens=400,
            strategy="last",
            token_counter=self.chat_model,
            include_system=True,
            allow_partial=False,
            start_on="human"
        )
        graph_builder = StateGraph(State)
        graph_builder.add_node("retrieve", self.retrieve)
        graph_builder.add_node("generate", self.generate)
        graph_builder.add_edge("retrieve", "generate")
        graph_builder.add_edge(START, "retrieve")
        memory = MemorySaver()
        self.graph = graph_builder.compile(checkpointer=memory)
        self.config = {"configurable": {"thread_id": "1"}}

    def retrieve(self, state: State):
        retrieved_docs = self.vectorstore.similarity_search(state["question"], k=2)
        return {"context": retrieved_docs}

    def generate(self, state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = self.prompt_template.invoke({"question": state["question"], "context": docs_content})
        response = self.chat_model.invoke(messages)
        return {"answer": response.content}

    def get_answer(self, query:str):
        """ Get the answer from the LLM based on the query."""
        response = self.graph.invoke({"question": query}, config=self.config) # type: ignore
        return response

assistant = AssistantInit()
