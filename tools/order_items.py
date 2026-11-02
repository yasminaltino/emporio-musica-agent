from pathlib import Path

import pandas as pd
from langchain.tools import tool

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_order_items():
    return pd.read_csv(DATA_DIR / "desafio_tecnico_ai_eng - order_items.csv")


@tool
def search_order_items(
    order_id: int | None = None,
    product_id: int | None = None,
) -> str:
    """
    Busca produtos incluídos em pedidos.
    Pode filtrar por ID do pedido ou ID do produto.
    Retorna o ID do pedido, ID do produto e quantidade.
    Retorna uma string JSON com os resultados da busca.
    """
    df = load_order_items()

    if order_id is not None:
        df = df[df["order_id"] == order_id]

    if product_id is not None:
        df = df[df["product_id"] == product_id]

    if df.empty:
        return "No order items found."

    return df.head(20).to_json(orient="records", force_ascii=False)
