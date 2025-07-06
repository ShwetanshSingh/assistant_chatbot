import os
import json

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage, trim_messages
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from dotenv import load_dotenv

load_dotenv("./config/.env")

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
        # Load the prompt from the JSON file
        with open(os.getenv("PROMPT_PATH", "./config/prompt.json"), "r") as f:
            prompt_data = json.load(f)

        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                # (
                #     "system",
                #     prompt_data["prompt"]
                # ),
                SystemMessage(prompt_data["prompt"]),
                MessagesPlaceholder(variable_name="messages")
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
        graph_builder = StateGraph(state_schema=MessagesState)
        graph_builder.add_edge(START, "model")
        graph_builder.add_node("model", self.call_model)
        memory = MemorySaver()
        self.graph = graph_builder.compile(checkpointer=memory)
        self.config = {"configurable": {"thread_id": "1"}}

    def call_model(self, state: MessagesState):
        """ Call the LLM with the current state messages."""
        trim_messages = self.trimmer.invoke(state["messages"])
        prompt = self.prompt_template.invoke(trim_messages)
        response = self.chat_model.invoke(prompt)
        return {"messages": [response]}

    def get_answer(self, query:str):
        """ Get the answer from the LLM based on the query."""
        response = self.graph.invoke({"messages": [HumanMessage(query)]}, config=self.config) # type: ignore
        return response["messages"][-1]

assistant = AssistantInit()
