import json

import pandas as pd
import pytest

from tools import orders


@pytest.fixture
def sample_orders():
    return pd.DataFrame(
        [
            {
                "order_id": 1,
                "customer_id": 3,
                "order_date": "2025-10-15",
                "status": "delivered",
                "total_brl": 11499.0,
                "payment_method": "pix",
                "tracking_code": "BRAB1234567BR",
                "estimated_delivery": "2025-10-25",
                "notes": None,
            },
            {
                "order_id": 2,
                "customer_id": 7,
                "order_date": "2025-11-02",
                "status": "delivered",
                "total_brl": 5999.0,
                "payment_method": "credit_6x",
                "tracking_code": "BRCX9876543BR",
                "estimated_delivery": "2025-11-15",
                "notes": None,
            },
            {
                "order_id": 8,
                "customer_id": 2,
                "order_date": "2026-02-15",
                "status": "shipped",
                "total_brl": 349.9,
                "payment_method": "debit",
                "tracking_code": "BRJL5544332BR",
                "estimated_delivery": "2026-02-28",
                "notes": None,
            },
            {
                "order_id": 13,
                "customer_id": 8,
                "order_date": "2026-03-12",
                "status": "pending",
                "total_brl": 1899.0,
                "payment_method": "pix",
                "tracking_code": None,
                "estimated_delivery": None,
                "notes": None,
            },
            {
                "order_id": 17,
                "customer_id": 15,
                "order_date": "2026-01-28",
                "status": "cancelled",
                "total_brl": 689.0,
                "payment_method": "pix",
                "tracking_code": None,
                "estimated_delivery": None,
                "notes": "Pedido cancelado pelo cliente antes do processamento",
            },
        ]
    )


@pytest.fixture(autouse=True)
def mock_load_orders(monkeypatch, sample_orders):
    monkeypatch.setattr(orders, "load_orders", lambda: sample_orders.copy())


def get_results(**kwargs):
    response = orders.search_orders.invoke(kwargs)
    return json.loads(response)


def test_search_order_by_id():
    results = get_results(order_ids=[1])

    assert len(results) == 1
    assert results[0]["order_id"] == 1


def test_search_orders_by_customer_id():
    results = get_results(customer_id=3)

    assert len(results) == 1
    assert results[0]["customer_id"] == 3


def test_search_orders_by_status():
    results = get_results(status="shipped")

    assert len(results) == 1
    assert results[0]["order_id"] == 8


def test_search_orders_by_payment_method():
    results = get_results(payment_method="pix")

    assert len(results) == 3
    assert all(o["payment_method"] == "pix" for o in results)


def test_search_orders_by_tracking_code():
    results = get_results(tracking_code="BRJL5544332BR")

    assert len(results) == 1
    assert results[0]["order_id"] == 8


def test_search_orders_by_order_date():
    results = get_results(order_date="2026-02")

    assert len(results) == 1
    assert results[0]["order_id"] == 8


def test_search_orders_by_estimated_delivery():
    results = get_results(estimated_delivery="2026-02-28")

    assert len(results) == 1
    assert results[0]["order_id"] == 8


def test_search_orders_by_notes():
    results = get_results(notes="cancelado")

    assert len(results) == 1
    assert results[0]["order_id"] == 17


def test_search_orders_with_multiple_filters():
    results = get_results(customer_id=2, status="shipped", payment_method="debit")

    assert len(results) == 1
    assert results[0]["order_id"] == 8


def test_search_orders_without_filters():
    results = get_results()

    assert len(results) == 5


def test_search_nonexistent_order():
    response = orders.search_orders.invoke({"order_ids": [9999]})

    assert response == "No orders found."


def test_search_orders_by_min_total():
    results = get_results(min_total_brl=5000)

    assert len(results) == 2
    assert all(o["total_brl"] >= 5000 for o in results)


def test_search_orders_by_max_total():
    results = get_results(max_total_brl=700)

    assert len(results) == 2
    assert all(o["total_brl"] <= 700 for o in results)


def test_search_orders_by_total_range():
    results = get_results(min_total_brl=500, max_total_brl=2000)

    assert len(results) == 2
    assert all(500 <= o["total_brl"] <= 2000 for o in results)
