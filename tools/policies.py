from langchain.tools import tool

from rag.inicialize_rag import get_retriever


@tool
def search_company_policies(query: str) -> str:
    """Busca políticas da empresa com base na consulta fornecida."""
    retriever = get_retriever()
    results = retriever.invoke(query)
    return "\n".join([doc.page_content for doc in results])
