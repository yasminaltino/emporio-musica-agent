import re
from pathlib import Path

import pandas as pd
from langchain.tools import tool

from tools.helpers import sanitize_text

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_customers():
    return pd.read_csv(DATA_DIR / "desafio_tecnico_ai_eng - customers.csv")


def extract_digits(text: str) -> str:
    """Remove qualquer caractere não numérico para correspondência segura de números de telefone."""
    if not isinstance(text, str):
        return ""
    return re.sub(r"\D", "", text)


@tool
def search_customers(
    name: str = "",
    email: str = "",
    phone: str = "",
    city: str = "",
) -> str:
    """
    Busca clientes no cadastro da loja por nome, email, telefone ou cidade.
    Crucial: Se a busca retornar 'No customers found.', o cliente não existe.
    Não tente variações ou outras palavras-chave após o retorno vazio.
    """
    df = load_customers()

    if name:
        name_clean = sanitize_text(name)
        df = df[
            df["name"]
            .apply(sanitize_text)
            .str.contains(name_clean, regex=False, na=False)
        ]

    if email:
        df = df[
            df["email"].str.contains(email.strip(), case=False, na=False, regex=False)
        ]

    if phone:
        phone_clean = extract_digits(phone)
        if phone_clean:
            df = df[
                df["phone"]
                .apply(extract_digits)
                .str.contains(phone_clean, regex=False, na=False)
            ]

    if city:
        city_clean = sanitize_text(city)
        df = df[
            df["city"]
            .apply(sanitize_text)
            .str.contains(city_clean, regex=False, na=False)
        ]

    if df.empty:
        return "No customers found."

    # Mantém o retorno leve com apenas 5 registros para evitar loops de token
    results = df[["customer_id", "name", "city"]].head(8)

    return results.to_json(orient="records", force_ascii=False)
