from pathlib import Path

import pandas as pd
from langchain.tools import tool

from tools.helpers import sanitize_text

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_promotions():
    return pd.read_csv(DATA_DIR / "desafio_tecnico_ai_eng - promotions.csv")


@tool
def search_promotions(
    product_id: int | None = None,
    description: str = "",
    active_only: bool = True,
) -> str:
    """
    Busca promoções por ID do produto ou descrição da promoção.
    Por padrão, apenas promoções ativas são retornadas.
    Também pode retornar promoções inativas quando active_only for False.
    Retorna uma string JSON com os resultados da busca.
    """
    df = load_promotions()

    if active_only:
        df = df[df["is_active"] == 1]

    if product_id is not None:
        df = df[df["product_id"] == product_id]

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
