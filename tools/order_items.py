from pathlib import Path

import pandas as pd
from langchain.tools import tool

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_order_items():
    return pd.read_csv(DATA_DIR / "desafio_tecnico_ai_eng - order_items.csv")


@tool
def search_order_items(
    order_ids: list[int] | None = None,
    product_id: int | None = None,
) -> str:
    """
    Busca produtos incluídos em pedidos.
    Pode filtrar por IDs dos pedidos ou ID do produto.
    Use order_ids para consultar um ou vários pedidos em uma chamada.
    Retorna o ID do pedido, ID do produto e quantidade.
    Retorna uma string JSON com os resultados da busca.
    """
    df = load_order_items()

    if order_ids is not None:
        df = df[df["order_id"].isin(order_ids)]

    if product_id is not None:
        df = df[df["product_id"] == product_id]

    if df.empty:
        return "No order items found."

    return df.head(20).to_json(orient="records", force_ascii=False)
