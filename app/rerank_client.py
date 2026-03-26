from __future__ import annotations

import httpx

from app.errors import UpstreamAuthError, UpstreamNotFoundError, UpstreamServiceError, UpstreamTimeoutError
from app.retry import run_with_retries
from app.settings import settings


def _extract_doc_text(doc: object, fallback: str) -> str:
    if isinstance(doc, str):
        return doc
    if isinstance(doc, dict):
        text = doc.get("text")
        if isinstance(text, str):
            return text
    return fallback


async def generate_rerank(query: str, documents: list[str], top_n: int | None = None) -> tuple[str, list[dict[str, object]]]:
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

    try:
        async with httpx.AsyncClient(timeout=settings.rerank_timeout_seconds) as client:
            response = await run_with_retries(
                operation_name="rerank",
                attempt_fn=lambda: _request_rerank_response(client, url, headers, payload),
                is_retryable=_is_retryable_httpx_error,
                max_retries=settings.rerank_max_retries,
                base_delay_seconds=settings.retry_backoff_seconds,
            )
    except httpx.TimeoutException as exc:
        raise UpstreamTimeoutError("Rerank request to upstream provider timed out.") from exc
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 401:
            raise UpstreamAuthError("Rerank authentication failed with upstream provider.") from exc
        if exc.response.status_code == 404:
            raise UpstreamNotFoundError("Rerank model or endpoint was not found upstream.") from exc
        raise UpstreamServiceError(f"Rerank provider returned status {exc.response.status_code}.") from exc
    except httpx.RequestError as exc:
        raise UpstreamServiceError("Rerank provider connection failed.") from exc

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


async def _request_rerank_response(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    payload: dict[str, object],
) -> httpx.Response:
    response = await client.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response


def _is_retryable_httpx_error(exc: Exception) -> bool:
    if isinstance(exc, (httpx.TimeoutException, httpx.RequestError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in {408, 409, 429, 500, 502, 503, 504}
    return False
