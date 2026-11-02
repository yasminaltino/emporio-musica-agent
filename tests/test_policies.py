from langchain_core.documents import Document

from tools import policies


class FakeRetriever:
    def __init__(self, documents):
        self.documents = documents

    def invoke(self, query):
        return self.documents


def test_search_policies_formats_page_and_content(monkeypatch):
    documents = [Document(page_content="Prazo de troca: 7 dias.", metadata={"page": 1})]
    monkeypatch.setattr(policies, "get_retriever", lambda: FakeRetriever(documents))

    response = policies.search_company_policies.invoke({"query": "trocas"})

    assert response == "[Página 2]\nPrazo de troca: 7 dias."


def test_search_policies_removes_duplicate_documents(monkeypatch):
    documents = [
        Document(page_content="Entrega via PAC.", metadata={"page": 2}),
        Document(page_content="Entrega via PAC.", metadata={"page": 2}),
    ]
    monkeypatch.setattr(policies, "get_retriever", lambda: FakeRetriever(documents))

    response = policies.search_company_policies.invoke({"query": "entrega"})

    assert response.count("Entrega via PAC.") == 1


def test_search_policies_without_results(monkeypatch):
    monkeypatch.setattr(policies, "get_retriever", lambda: FakeRetriever([]))

    response = policies.search_company_policies.invoke({"query": "informação"})

    assert response == "Nenhuma informação relevante encontrada nas políticas."
