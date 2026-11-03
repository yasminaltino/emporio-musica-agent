import json

import pandas as pd
import pytest

from tools import categories


@pytest.fixture
def sample_categories():
    return pd.DataFrame(
        [
            {
                "category_id": 1,
                "name": "Guitarras",
                "description": "Guitarras elétricas de todos os estilos.",
            },
            {
                "category_id": 2,
                "name": "Baixos",
                "description": "Baixos elétricos e acústicos.",
            },
            {
                "category_id": 5,
                "name": "Violões",
                "description": "Violões clássicos, folk e eletroacústicos.",
            },
            {
                "category_id": 6,
                "name": "Instrumentos de Sopro (Madeiras)",
                "description": "Flautas, clarinetes, saxofones e oboés.",
            },
            {
                "category_id": 7,
                "name": "Instrumentos de Sopro (Metais)",
                "description": "Trompetes, trombones e tubas.",
            },
        ]
    )


@pytest.fixture(autouse=True)
def mock_load_categories(monkeypatch, sample_categories):
    monkeypatch.setattr(categories, "load_categories", lambda: sample_categories.copy())


def get_results(**kwargs):
    response = categories.search_categories.invoke(kwargs)
    return json.loads(response)


def test_search_category_by_name():
    results = get_results(name="Violões")

    assert len(results) == 1
    assert results[0]["category_id"] == 5


def test_search_categories_by_partial_name():
    results = get_results(name="sopro")

    assert len(results) == 2
    assert all("Sopro" in c["name"] for c in results)


def test_search_categories_by_specification():
    results = get_results(specifications="saxofones")

    assert len(results) == 1
    assert results[0]["category_id"] == 6


def test_search_categories_case_insensitive():
    results = get_results(name="gUiTaRrAs")

    assert len(results) == 1
    assert results[0]["name"] == "Guitarras"


def test_search_categories_without_filters():
    results = get_results()

    assert len(results) == 5


def test_search_nonexistent_category():
    response = categories.search_categories.invoke(
        {"name": "Instrumentos inexistentes"}
    )

    assert response == "No categories found."


def test_search_categories_returns_all_results(monkeypatch, sample_categories):
    data = pd.concat([sample_categories] * 3, ignore_index=True)
    monkeypatch.setattr(categories, "load_categories", lambda: data.copy())

    assert len(get_results()) == 15
