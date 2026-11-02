import unicodedata


def sanitize_text(text: str) -> str:
    """Remove acentos e converte para minúsculas para correspondência uniforme de texto."""
    if not isinstance(text, str):
        return ""
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    ).lower()
