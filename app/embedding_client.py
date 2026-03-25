from __future__ import annotations

from openai import OpenAI

from app.settings import settings


def generate_embedding(input_text: str) -> tuple[str, list[float]]:
    if not settings.embedding_openai_api_key:
        raise ValueError("EMBEDDING_OPENAI_API_KEY is missing. Please set it in your .env file.")

    client_kwargs: dict[str, str] = {"api_key": settings.embedding_openai_api_key}
    if settings.embedding_openai_base_url:
        client_kwargs["base_url"] = settings.embedding_openai_base_url
    client = OpenAI(**client_kwargs)

    response = client.embeddings.create(
        model=settings.embedding_model,
        input=input_text,
        encoding_format=settings.embedding_encoding_format,
    )

    if not response.data:
        raise RuntimeError("Embedding response is empty.")

    embedding = response.data[0].embedding
    if not embedding:
        raise RuntimeError("Embedding vector is empty.")

    return response.model, embedding

