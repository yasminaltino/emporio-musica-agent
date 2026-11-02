from pathlib import Path

import pandas as pd
from langchain.tools import tool

from tools.helpers import sanitize_text

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

STATUS_MAP = {
    "entregue": "delivered",
    "entregues": "delivered",
    "enviado": "shipped",
    "enviada": "shipped",
    "em_transito": "shipped",
    "confirmado": "confirmed",
    "confirmada": "confirmed",
    "confirmed": "confirmed",
    "pendente": "pending",
    "cancelado": "cancelled",
    "cancelada": "cancelled",
    "canceled": "cancelled",
}

PAYMENT_METHOD_MAP = {
    "credito": "credit",
    "cartao de credito": "credit",
    "credit": "credit",
    "debito": "debit",
    "cartao de debito": "debit",
    "debit": "debit",
}


def load_orders():
    return pd.read_csv(DATA_DIR / "desafio_tecnico_ai_eng - orders.csv")


@tool
def search_orders(
    order_ids: list[int] | None = None,
    customer_id: int | None = None,
    order_date: str = "",
    status: str = "",
    min_total_brl: float | None = None,
    max_total_brl: float | None = None,
    payment_method: str = "",
    tracking_code: str = "",
    estimated_delivery: str = "",
    notes: str = "",
) -> str:
    """
    Busca pedidos por uma lista de IDs, cliente, datas, status,
    valores, métodos de pagamento ou rastreio.

    Use order_ids para consultar um ou vários pedidos.
    Combine filtros quando necessário.
    Se retornar 'No orders found.', não tente variações
    de parâmetros sem uma justificativa.
    """
    df = load_orders()

    if order_ids is not None:
        df = df[df["order_id"].isin(order_ids)]

    if customer_id is not None:
        df = df[df["customer_id"] == customer_id]

    if order_date:
        df = df[
            df["order_date"].str.contains(order_date, case=False, na=False, regex=False)
        ]

    if status:
        status_clean = sanitize_text(status)
        status_target = STATUS_MAP.get(status_clean, status_clean)

        df = df[df["status"].apply(sanitize_text) == status_target]

    if min_total_brl is not None:
        df = df[df["total_brl"] >= min_total_brl]
    if max_total_brl is not None:
        df = df[df["total_brl"] <= max_total_brl]

    if payment_method:
        method_clean = sanitize_text(payment_method)
        method_target = PAYMENT_METHOD_MAP.get(method_clean, method_clean)

        df = df[
            df["payment_method"]
            .apply(sanitize_text)
            .str.startswith(method_target, na=False)
        ]

    if tracking_code:
        df = df[
            df["tracking_code"].str.contains(
                tracking_code, case=False, na=False, regex=False
            )
        ]

    if estimated_delivery:
        df = df[
            df["estimated_delivery"].str.contains(
                estimated_delivery, case=False, na=False, regex=False
            )
        ]

    if notes:
        notes_clean = sanitize_text(notes)
        df = df[
            df["notes"]
            .apply(sanitize_text)
            .str.contains(notes_clean, regex=False, na=False)
        ]

    if df.empty:
        return "No orders found."

    return df.to_json(orient="records", force_ascii=False)
