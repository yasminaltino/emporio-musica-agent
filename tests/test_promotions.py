import json

import pandas as pd
import pytest

from tools import promotions


@pytest.fixture(autouse=True)
def mock_load_promotions(monkeypatch):
    data = pd.DataFrame(
        [
            {
                "promotion_id": 1,
                "product_id": 81,
                "discount_percent": 10,
                "description": "Oferta Especial",
                "is_active": 1,
            },
            {
                "promotion_id": 2,
                "product_id": 90,
                "discount_percent": 20,
                "description": "Promoção de Inverno",
                "is_active": 0,
            },
        ]
    )
    monkeypatch.setattr(promotions, "load_promotions", lambda: data.copy())


def get_results(**kwargs):
    return json.loads(promotions.search_promotions.invoke(kwargs))


def test_search_active_promotions_by_default():
    results = get_results()

    assert len(results) == 1
    assert results[0]["promotion_id"] == 1


def test_search_promotions_by_product():
    results = get_results(product_ids=[81])

    assert results[0]["product_id"] == 81


def test_search_inactive_promotions():
    results = get_results(description="inverno", active_only=False)

    assert len(results) == 1
    assert results[0]["is_active"] == 0


def test_search_nonexistent_promotion():
    response = promotions.search_promotions.invoke({"product_ids": [9999]})

    assert response == "No promotions found."
