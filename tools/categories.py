from pathlib import Path

import pandas as pd
from langchain.tools import tool

from tools.helpers import sanitize_text

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


CATEGORY_ALIASES = {
    "violao": "violoes",
    "ukulele": "ukuleles",
    "guitarra": "guitarras",
    "baixo": "baixos",
}


def load_categories():
    return pd.read_csv(DATA_DIR / "desafio_tecnico_ai_eng - categories.csv")


@tool
def search_categories(name: str = "", specifications: str = "") -> str:
    """
    Busca categorias no catálogo da loja por nome ou características descritivas.
    """
    df = load_categories()

    if name:
        name_clean = sanitize_text(name)
        name_clean = CATEGORY_ALIASES.get(name_clean, name_clean)

        df = df[
            df["name"]
            .apply(sanitize_text)
            .str.contains(name_clean, regex=False, na=False)
        ]

    if specifications:
        spec_clean = sanitize_text(specifications)
        df = df[
            df["description"]
            .apply(sanitize_text)
            .str.contains(spec_clean, regex=False, na=False)
        ]

    if df.empty:
        return "No categories found."

    return df.head(10).to_json(orient="records", force_ascii=False)
