import os

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv("./config/.env")
embedding_model = HuggingFaceEmbeddings(
    model_name=os.getenv("EMBEDDING_ID", "sentence-transformers/all-mpnet-base-v2")
)
loader = DirectoryLoader(
    os.getenv("RAG_DOCUMENTS_PATH", "./documents"),
    glob="*.html",
    use_multithreading=True,
    show_progress=True
)
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
docs = text_splitter.split_documents(documents)
vectorstore = FAISS.from_documents(docs, embedding_model)
vectorstore.save_local(os.getenv("VECTORSTORE_PATH", "./vectorstore"))
