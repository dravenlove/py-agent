from __future__ import annotations

import httpx

from app.settings import settings


def _extract_doc_text(doc: object, fallback: str) -> str:
    if isinstance(doc, str):
        return doc
    if isinstance(doc, dict):
        text = doc.get("text")
        if isinstance(text, str):
            return text
    return fallback


def generate_rerank(query: str, documents: list[str], top_n: int | None = None) -> tuple[str, list[dict[str, object]]]:
    if not settings.rerank_openai_api_key:
        raise ValueError("RERANK_OPENAI_API_KEY is missing. Please set it in your .env file.")
    if not settings.rerank_openai_base_url:
        raise ValueError("RERANK_OPENAI_BASE_URL is missing. Please set it in your .env file.")

    payload: dict[str, object] = {
        "model": settings.rerank_model,
        "query": query,
        "documents": documents,
    }
    if top_n is not None:
        payload["top_n"] = top_n

    url = f"{settings.rerank_openai_base_url.rstrip('/')}/rerank"
    headers = {
        "Authorization": f"Bearer {settings.rerank_openai_api_key}",
        "Content-Type": "application/json",
    }

    response = httpx.post(url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()

    body = response.json()
    if not isinstance(body, dict):
        raise RuntimeError("Rerank response is invalid.")

    results = body.get("results", [])
    if not isinstance(results, list):
        raise RuntimeError("Rerank response does not contain a valid results list.")

    normalized: list[dict[str, object]] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        index = item.get("index")
        score = item.get("relevance_score")
        document = item.get("document")
        fallback = ""
        if isinstance(index, int) and 0 <= index < len(documents):
            fallback = documents[index]
        normalized.append(
            {
                "index": index,
                "score": score,
                "document": _extract_doc_text(document, fallback),
            }
        )

    return settings.rerank_model, normalized
