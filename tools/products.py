from pathlib import Path

import pandas as pd
from langchain.tools import tool

from tools.helpers import sanitize_text

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_products():
    return pd.read_csv(DATA_DIR / "desafio_tecnico_ai_eng - products.csv")


@tool
def search_products(
    product_ids: list[int] | None = None,
    name: str = "",
    category_id: int | None = None,
    max_price: float | None = None,
    min_price: float | None = None,
    available_only: bool = False,
    specifications: str = "",
    limit: int | None = 10,
    out_of_stock: bool = False,
) -> str:
    """
    Busca instrumentos e produtos no catálogo da loja.
    Use limit para restringir o número de resultados retornados ou None para retornar todos os resultados.
    Crucial: Se a busca por 'name' ou 'specifications' retornar 'No products found.',
    significa que o item não existe na loja. Não tente palavras-chave alternativas,
    não remova os filtros informados pelo cliente e não faça buscas genéricas apenas por preço.
    Informe imediatamente ao cliente que o produto não está disponível.
    Use product_ids para buscar um ou mais produtos específicos pelo ID.
    """
    df = load_products()

    if product_ids is not None:
        df = df[df["product_id"].isin(product_ids)]

    if available_only:
        df = df[(df["status"] == "active") & (df["stock_quantity"] > 0)]

    if out_of_stock:
        df = df[df["stock_quantity"] == 0]

    if name:
        name_clean = sanitize_text(name)

        name_match = (
            df["name"]
            .apply(sanitize_text)
            .str.contains(name_clean, regex=False, na=False)
        )
        desc_match = (
            df["description"]
            .apply(sanitize_text)
            .str.contains(name_clean, regex=False, na=False)
        )

        df = df[name_match | desc_match]

    if category_id is not None:
        df = df[df["category_id"] == category_id]

    if max_price is not None:
        df = df[df["price_brl"] <= max_price]

    if min_price is not None:
        df = df[df["price_brl"] >= min_price]

    # Busca inteligente por Especificações (ignorando acentos e maiúsculas)
    if specifications:
        specs_clean = sanitize_text(specifications)
        desc_match = (
            df["description"]
            .apply(sanitize_text)
            .str.contains(specs_clean, regex=False, na=False)
        )
        specs_match = (
            df["specs"]
            .apply(sanitize_text)
            .str.contains(specs_clean, regex=False, na=False)
        )
        df = df[desc_match | specs_match]

    if df.empty:
        return "No products found."

    if limit is not None:
        df = df.head(limit)

    results = df[["product_id", "name", "price_brl", "stock_quantity", "description"]]

    return results.to_json(orient="records", force_ascii=False)
