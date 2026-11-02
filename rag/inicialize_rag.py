from pathlib import Path

from google import genai
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_DIR = DATA_DIR / "chroma_db"


def create_embeddings():

    pdf_path = DATA_DIR / "politicas_da_loja.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    db = Chroma.from_documents(chunks, embeddings, persist_directory=str(DB_DIR))
    return db


def get_retriever():
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    db = Chroma(persist_directory=str(DB_DIR), embedding_function=embeddings)
    return db.as_retriever(search_kwargs={"k": 4})
