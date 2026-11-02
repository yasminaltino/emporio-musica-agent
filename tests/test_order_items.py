import json

import pandas as pd
import pytest

from tools import order_items


@pytest.fixture(autouse=True)
def mock_load_order_items(monkeypatch):
    data = pd.DataFrame(
        [
            {"order_id": 1, "product_id": 81, "quantity": 1},
            {"order_id": 2, "product_id": 81, "quantity": 2},
            {"order_id": 2, "product_id": 90, "quantity": 1},
        ]
    )
    monkeypatch.setattr(order_items, "load_order_items", lambda: data.copy())


def get_results(**kwargs):
    return json.loads(order_items.search_order_items.invoke(kwargs))


def test_search_items_by_order():
    results = get_results(order_id=2)

    assert len(results) == 2
    assert all(item["order_id"] == 2 for item in results)


def test_search_items_by_product():
    results = get_results(product_id=81)

    assert {item["order_id"] for item in results} == {1, 2}


def test_search_nonexistent_order_item():
    response = order_items.search_order_items.invoke({"order_id": 9999})

    assert response == "No order items found."
