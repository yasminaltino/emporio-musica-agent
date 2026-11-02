import json
import pandas as pd
import pytest

from tools import customers


@pytest.fixture
def sample_customers():
    return pd.DataFrame(
        [
            {
                "customer_id": 1,
                "name": "Lucas Mendes da Silva",
                "phone": "(67) 99812-3456",
                "email": "lucas.mendes@jmail.com",
                "city": "Campo Grande",
            },
            {
                "customer_id": 2,
                "name": "Ana Carolina Ferreira",
                "phone": "(67) 99654-7821",
                "email": "anacarol.ferreira@coldmail.com",
                "city": "Campo Grande",
            },
            {
                "customer_id": 3,
                "name": "Pedro Henrique Oliveira",
                "phone": "(67) 98432-1098",
                "email": "pedro.oliveira@jmail.com",
                "city": "Dourados",
            },
            {
                "customer_id": 4,
                "name": "Mariana Costa Santos",
                "phone": "(67) 99123-4567",
                "email": "mariana.costa@inlook.com",
                "city": "Campo Grande",
            },
            {
                "customer_id": 5,
                "name": "Rafael Augusto Pereira",
                "phone": "(67) 98765-4321",
                "email": "rafael.pereira@jmail.com",
                "city": "Corumbá",
            },
        ]
    )


@pytest.fixture(autouse=True)
def mock_load_customers(monkeypatch, sample_customers):
    monkeypatch.setattr(customers, "load_customers", lambda: sample_customers.copy())


def get_results(**kwargs):
    response = customers.search_customers.invoke(kwargs)
    return json.loads(response)


def test_search_customer_by_name():
    results = get_results(name="Lucas Mendes")

    assert len(results) == 1
    assert results[0]["customer_id"] == 1


def test_search_customer_by_email():
    results = get_results(email="pedro.oliveira@jmail.com")

    assert len(results) == 1
    assert results[0]["customer_id"] == 3


def test_search_customer_by_phone():
    results = get_results(phone="99812-3456")

    assert len(results) == 1
    assert results[0]["customer_id"] == 1


def test_search_customers_by_city():
    results = get_results(city="Campo Grande")

    assert len(results) == 3
    assert all(c["city"] == "Campo Grande" for c in results)


def test_search_customers_with_multiple_filters():
    results = get_results(name="Pedro", city="Dourados")

    assert len(results) == 1
    assert results[0]["customer_id"] == 3


def test_search_customers_case_insensitive():
    results = get_results(name="lUcAs")

    assert len(results) == 1
    assert results[0]["customer_id"] == 1


def test_search_customers_without_filters():
    results = get_results()

    assert len(results) == 5


def test_search_nonexistent_customer():
    response = customers.search_customers.invoke({"name": "Cliente Inexistente"})

    assert response == "No customers found."
