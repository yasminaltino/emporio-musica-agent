from pathlib import Path

import pandas as pd
from langchain.tools import tool

from tools.helpers import sanitize_text

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_orders():
    return pd.read_csv(DATA_DIR / "desafio_tecnico_ai_eng - orders.csv")


@tool
def search_orders(
    order_id: int | None = None,
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
    Busca pedidos de clientes usando ID do pedido, ID do cliente, datas, status,
    valores, métodos de pagamento ou rastreio.
    Crucial: Se retornar 'No orders found.', o pedido ou critério não existe.
    Não tente variações ou outros parâmetros após o retorno vazio.
    """
    df = load_orders()

    if order_id is not None:
        df = df[df["order_id"] == order_id]

    if customer_id is not None:
        df = df[df["customer_id"] == customer_id]

    if order_date:
        df = df[
            df["order_date"].str.contains(order_date, case=False, na=False, regex=False)
        ]

    if status:
        status_clean = sanitize_text(status)
        df = df[
            df["status"]
            .apply(sanitize_text)
            .str.contains(status_clean, regex=False, na=False)
        ]

    if min_total_brl is not None:
        df = df[df["total_brl"] >= min_total_brl]
    if max_total_brl is not None:
        df = df[df["total_brl"] <= max_total_brl]

    # CORREÇÃO: Mapeia termos comuns em português para o padrão salvo no CSV (credit, pix, debit, boleto)
    if payment_method:
        method_clean = sanitize_text(payment_method)
        if (
            "credito" in method_clean
            or "cartao" in method_clean
            or "credit" in method_clean
        ):
            method_target = "credit"
        elif "debito" in method_clean or "debit" in method_clean:
            method_target = "debit"
        else:
            method_target = method_clean

        df = df[
            df["payment_method"]
            .apply(sanitize_text)
            .str.contains(method_target, regex=False, na=False)
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

    return df.head(10).to_json(orient="records", force_ascii=False)
