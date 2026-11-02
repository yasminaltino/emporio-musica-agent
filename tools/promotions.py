from pathlib import Path

import pandas as pd
from langchain.tools import tool

from tools.helpers import sanitize_text

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_promotions():
    return pd.read_csv(DATA_DIR / "desafio_tecnico_ai_eng - promotions.csv")


@tool
def search_promotions(
    product_ids: list[int] | None = None,
    description: str = "",
    active_only: bool = True,
) -> str:
    """
    Busca promoções por IDs de produtos ou descrição da promoção.
    Por padrão, apenas promoções ativas são retornadas.
    Também pode retornar promoções inativas quando active_only for False.
    Use product_ids para consultar promoções de um ou vários produtos específicos.
    Retorna uma string JSON com os resultados da busca.
    """
    df = load_promotions()

    if active_only:
        df = df[df["is_active"] == 1]

    if product_ids is not None:
        df = df[df["product_id"].isin(product_ids)]

    if description:
        df = df[
            df["description"]
            .apply(sanitize_text)
            .str.contains(
                sanitize_text(description),
                case=False,
                na=False,
                regex=False,
            )
        ]

    if df.empty:
        return "No promotions found."

    return df.head(20).to_json(orient="records", force_ascii=False)
