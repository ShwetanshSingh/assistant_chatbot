from langchain import hub
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
import datetime

# loading api keys
from dotenv import load_dotenv
load_dotenv()

def log_message(component: str, message: str):
    date_time = datetime.datetime.now()     # current datetime
    # extracting date and time, excluding milliseconds
    timestamp = f"[{date_time.date()} {date_time.hour}:{date_time.minute}:{date_time.second}]"
    # formatting log message
    message = timestamp+" "+f"[{component}]"+" "+message+"\n"
    print(message)
    with open("logs.txt", "a") as f:
        f.write(message)

class Chatbot():
    def __init__(
            self, 
            llm_model_repo: str ="meta-llama/Llama-3.3-70B-Instruct", # meta-llama/Meta-Llama-3-8B-Instruct | meta-llama/Llama-3.3-70B-Instruct | HuggingFaceH4/zephyr-7b-beta
            embeddings_model_repo: str ="sentence-transformers/all-mpnet-base-v2",
            document_name: str ="the_wonderful_wizard_of_oz.txt"
        ):
        # Instantiation of llm chat model
        log_message("LLM", f"Initializing {llm_model_repo}")
        self.chat_model = ChatHuggingFace(
            llm=HuggingFaceEndpoint(
                repo_id=llm_model_repo,
                task="text-generation",
                max_new_tokens=512,
                do_sample=False,
                repetition_penalty=1.03,
                provider="cerebras"                             # setting provider="cerebras"
                                                                # because featherless-ai
                                                                # is not supported
            )
        )
        log_message("LLM", f"{llm_model_repo} initialized")
        # Instantiation of embeddings model
        log_message("Embedding model", f"Initializing {embeddings_model_repo}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embeddings_model_repo
        )
        log_message("Embedding model", f"{embeddings_model_repo} initialized")
        # Instantiation of vector store
        log_message("Vector Store", "Initializing Vector store")
        self.vector_store = InMemoryVectorStore(self.embeddings)
        log_message("Vector Store", "Vector Store initialized")

        # Document processing
        log_message("Document Loader", f"Loading {document_name}")
        self.loader = DirectoryLoader(
            "./data", 
            glob=document_name, 
            use_multithreading=True, 
            show_progress=True
        )
        self.docs = self.loader.load()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )

        # splits of the document to be added to vector store
        self.all_splits = self.text_splitter.split_documents(self.docs)
        self.vector_store.add_documents(documents=self.all_splits)
        log_message("Document Loader", f"{document_name} loaded")

# Create object of Chatbot
chatbot = Chatbot()

# Define prompt for question-answering
prompt = hub.pull("rlm/rag-prompt")

# Define state for applications
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# Define application steps
def retrieve(state: State):
    retrieved_docs = chatbot.vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke(
        {"question": state["question"], "context": docs_content}
    )
    response = chatbot.chat_model.invoke(messages)
    return {"answer": response.content}

# Compile application
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# def gradio_func():
#     pass