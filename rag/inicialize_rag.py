import re
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_DIR = DATA_DIR / "chroma_db"


def normalize_text(text: str) -> str:
    """Remove espaços e quebras de linha desnecessários."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def create_embeddings():
    pdf_path = DATA_DIR / "politicas_da_loja.pdf"

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF não encontrado: {pdf_path}")

    documents = PyPDFLoader(str(pdf_path)).load()

    for doc in documents:
        doc.page_content = normalize_text(doc.page_content)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    # Remove trechos exatamente duplicados.
    unique_chunks = []
    seen = set()

    for chunk in chunks:
        content = chunk.page_content.strip()

        if content not in seen:
            seen.add(content)
            unique_chunks.append(chunk)

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    db = Chroma.from_documents(
        unique_chunks,
        embeddings,
        persist_directory=str(DB_DIR),
    )

    return db


def get_retriever():
    # Cria o índice caso ainda não exista
    if not DB_DIR.exists():
        create_embeddings()
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    db = Chroma(
        persist_directory=str(DB_DIR),
        embedding_function=embeddings,
    )

    return db.as_retriever(search_kwargs={"k": 4})
