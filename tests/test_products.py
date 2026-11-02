import json

import pandas as pd
import pytest

from tools import products


@pytest.fixture
def sample_products():
    return pd.DataFrame(
        [
            {
                "product_id": 81,
                "name": "Yamaha C40 Nylon Natural",
                "price_brl": 599.90,
                "category_id": 5,
                "description": "Violão clássico para iniciantes.",
                "stock_quantity": 12,
                "status": "active",
                "specs": '{"strings": "nylon"}',
            },
            {
                "product_id": 82,
                "name": "Yamaha C70",
                "price_brl": 849.00,
                "category_id": 5,
                "description": "Violão clássico de nylon.",
                "stock_quantity": 8,
                "status": "active",
                "specs": '{"strings": "nylon"}',
            },
            {
                "product_id": 86,
                "name": "Tagima Dallas",
                "price_brl": 749.00,
                "category_id": 5,
                "description": "Violão folk.",
                "stock_quantity": 9,
                "status": "active",
                "specs": '{"strings": "steel"}',
            },
            {
                "product_id": 96,
                "name": "Giannini GF-3D",
                "price_brl": 799.90,
                "category_id": 5,
                "description": "Violão acústico.",
                "stock_quantity": 0,
                "status": "active",
                "specs": '{"strings": "steel"}',
            },
            {
                "product_id": 113,
                "name": "Produto Descontinuado",
                "price_brl": 299.00,
                "category_id": 5,
                "description": "Produto fora de linha.",
                "stock_quantity": 0,
                "status": "discontinued",
                "specs": "{}",
            },
        ]
    )


@pytest.fixture(autouse=True)
def mock_load_products(monkeypatch, sample_products):
    monkeypatch.setattr(products, "load_products", lambda: sample_products.copy())


def get_results(**kwargs):
    response = products.search_products.invoke(kwargs)
    return json.loads(response)


def test_search_product_by_name():
    results = get_results(name="Yamaha C40")

    assert len(results) == 1
    assert results[0]["product_id"] == 81


def test_search_products_by_category():
    results = get_results(category_id=5)

    assert len(results) > 0
    assert all(p["product_id"] for p in results)


def test_search_products_by_max_price():
    results = get_results(max_price=700)

    assert len(results) > 0
    assert all(p["price_brl"] <= 700 for p in results)


def test_search_products_by_min_price():
    results = get_results(min_price=700)

    assert len(results) > 0
    assert all(p["price_brl"] >= 700 for p in results)


def test_search_products_by_specification():
    results = get_results(specifications="nylon")

    assert {product["product_id"] for product in results} == {81, 82}


def test_search_products_without_stock():
    results = get_results(name="Giannini GF-3D", available_only=False)

    assert len(results) == 1
    assert results[0]["stock_quantity"] == 0


def test_search_nonexistent_product():
    response = products.search_products.invoke({"name": "Produto Inexistente"})

    assert response == "No products found."
