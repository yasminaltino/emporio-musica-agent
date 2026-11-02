from langchain.tools import tool

from rag.inicialize_rag import get_retriever


@tool
def search_company_policies(query: str) -> str:
    """
    Utilize esta ferramenta para consultar as políticas da loja.
    Para perguntas abrangentes sobre uma política, faça uma única consulta com o tema geral.
    Só realize buscas adicionais quando a informação retornada for insuficiente para responder.
    """
    retriever = get_retriever()
    results = retriever.invoke(query)

    if not results:
        return "Nenhuma informação relevante encontrada nas políticas."

    evidence = []
    seen = set()

    for doc in results:
        content = doc.page_content.strip()

        if content in seen:
            continue

        seen.add(content)

        page = doc.metadata.get("page")

        if page is not None:
            page += 1

        evidence.append(
            f"[Página {page if page is not None else 'não identificada'}]\n{content}"
        )

    return "\n\n".join(evidence)
