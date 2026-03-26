from __future__ import annotations

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    NotFoundError,
)

from app.errors import (
    UpstreamAuthError,
    UpstreamNotFoundError,
    UpstreamServiceError,
    UpstreamTimeoutError,
)
from app.retry import run_with_retries
from app.settings import settings


async def generate_embedding(input_text: str) -> tuple[str, list[float]]:
    if not settings.embedding_openai_api_key:
        raise ValueError("EMBEDDING_OPENAI_API_KEY is missing. Please set it in your .env file.")

    client_kwargs: dict[str, str | float] = {
        "api_key": settings.embedding_openai_api_key,
        "timeout": settings.embedding_timeout_seconds,
    }
    if settings.embedding_openai_base_url:
        client_kwargs["base_url"] = settings.embedding_openai_base_url
    client = AsyncOpenAI(**client_kwargs)

    try:
        response = await run_with_retries(
            operation_name="embedding",
            attempt_fn=lambda: client.embeddings.create(
                model=settings.embedding_model,
                input=input_text,
                encoding_format=settings.embedding_encoding_format,
            ),
            is_retryable=_is_retryable_openai_error,
            max_retries=settings.embedding_max_retries,
            base_delay_seconds=settings.retry_backoff_seconds,
        )
    except APITimeoutError as exc:
        raise UpstreamTimeoutError("Embedding request to upstream provider timed out.") from exc
    except AuthenticationError as exc:
        raise UpstreamAuthError("Embedding authentication failed with upstream provider.") from exc
    except NotFoundError as exc:
        raise UpstreamNotFoundError("Embedding model or endpoint was not found upstream.") from exc
    except APIConnectionError as exc:
        raise UpstreamServiceError("Embedding provider connection failed.") from exc
    except APIStatusError as exc:
        raise UpstreamServiceError(f"Embedding provider returned status {exc.status_code}.") from exc

    if not response.data:
        raise UpstreamServiceError("Embedding provider returned an empty response.")

    embedding = response.data[0].embedding
    if not embedding:
        raise UpstreamServiceError("Embedding vector is empty.")

    return response.model, embedding


def _is_retryable_openai_error(exc: Exception) -> bool:
    if isinstance(exc, (APITimeoutError, APIConnectionError)):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code in {408, 409, 429, 500, 502, 503, 504}
    return False
