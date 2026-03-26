from __future__ import annotations

from typing import Any

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


def _get_value(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _extract_responses_text(response: Any) -> str:
    output_text = _get_value(response, "output_text")
    if isinstance(output_text, str):
        cleaned = output_text.strip()
        if cleaned:
            return cleaned

    parts: list[str] = []
    outputs = _get_value(response, "output") or []
    for item in outputs:
        contents = _get_value(item, "content") or []
        for content in contents:
            text = _get_value(content, "text")
            if isinstance(text, str):
                cleaned = text.strip()
                if cleaned:
                    parts.append(cleaned)

    return "\n".join(parts).strip()


def _extract_chat_completions_text(response: Any) -> str:
    choices = _get_value(response, "choices") or []
    if not choices:
        return ""

    message = _get_value(choices[0], "message")
    content = _get_value(message, "content")
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            text = _get_value(item, "text")
            if isinstance(text, str):
                cleaned = text.strip()
                if cleaned:
                    parts.append(cleaned)
        return "\n".join(parts).strip()

    return ""


def _normalize_chat_base_url(base_url: str | None) -> str | None:
    if not base_url:
        return None

    normalized = base_url.rstrip("/")
    suffix = "/chat/completions"
    if normalized.lower().endswith(suffix):
        normalized = normalized[: -len(suffix)]
    return normalized or None


def _resolve_chat_api_style(base_url: str | None, style_raw: str) -> str:
    style = style_raw.strip().lower()
    if style not in {"responses", "chat_completions"}:
        raise ValueError("CHAT_API_STYLE must be 'responses' or 'chat_completions'.")

    # If base url already contains /chat/completions, auto-switch mode.
    if style == "responses" and base_url and base_url.rstrip("/").lower().endswith("/chat/completions"):
        return "chat_completions"
    return style


async def generate_reply(user_message: str) -> str:
    if not settings.chat_openai_api_key:
        raise ValueError("CHAT_OPENAI_API_KEY is missing. Please set it in your .env file.")

    style = _resolve_chat_api_style(settings.chat_openai_base_url, settings.chat_api_style)
    base_url = settings.chat_openai_base_url
    if style == "chat_completions":
        base_url = _normalize_chat_base_url(base_url)

    client_kwargs: dict[str, str | float] = {
        "api_key": settings.chat_openai_api_key,
        "timeout": settings.chat_timeout_seconds,
    }
    if base_url:
        client_kwargs["base_url"] = base_url
    client = AsyncOpenAI(**client_kwargs)

    try:
        response = await run_with_retries(
            operation_name="chat",
            attempt_fn=lambda: _request_chat_response(client, style, user_message),
            is_retryable=_is_retryable_openai_error,
            max_retries=settings.chat_max_retries,
            base_delay_seconds=settings.retry_backoff_seconds,
        )
    except APITimeoutError as exc:
        raise UpstreamTimeoutError("Chat request to upstream provider timed out.") from exc
    except AuthenticationError as exc:
        raise UpstreamAuthError("Chat authentication failed with upstream provider.") from exc
    except NotFoundError as exc:
        raise UpstreamNotFoundError("Chat model or endpoint was not found upstream.") from exc
    except APIConnectionError as exc:
        raise UpstreamServiceError("Chat provider connection failed.") from exc
    except APIStatusError as exc:
        raise UpstreamServiceError(f"Chat provider returned status {exc.status_code}.") from exc

    if style == "chat_completions":
        reply = _extract_chat_completions_text(response)
    else:
        reply = _extract_responses_text(response)

    if not reply:
        raise UpstreamServiceError("Chat provider returned an empty response.")
    return reply


async def _request_chat_response(client: AsyncOpenAI, style: str, user_message: str) -> Any:
    if style == "chat_completions":
        return await client.chat.completions.create(
            model=settings.chat_model,
            messages=[{"role": "user", "content": user_message}],
        )

    return await client.responses.create(
        model=settings.chat_model,
        input=[
            {
                "role": "user",
                "content": [{"type": "input_text", "text": user_message}],
            }
        ],
    )


def _is_retryable_openai_error(exc: Exception) -> bool:
    if isinstance(exc, (APITimeoutError, APIConnectionError)):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code in {408, 409, 429, 500, 502, 503, 504}
    return False
