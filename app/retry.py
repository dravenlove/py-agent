from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import TypeVar

from app.observability import get_request_id

T = TypeVar("T")

logger = logging.getLogger("ai_agent.upstream")


async def run_with_retries(
    *,
    operation_name: str,
    attempt_fn: Callable[[], Awaitable[T]],
    is_retryable: Callable[[Exception], bool],
    max_retries: int,
    base_delay_seconds: float,
) -> T:
    total_attempts = max_retries + 1

    for attempt in range(1, total_attempts + 1):
        try:
            logger.info("[%s] %s attempt %s/%s", get_request_id(), operation_name, attempt, total_attempts)
            return await attempt_fn()
        except Exception as exc:
            should_retry = attempt < total_attempts and is_retryable(exc)
            logger.warning(
                "[%s] %s failed on attempt %s/%s with %s: %s",
                get_request_id(),
                operation_name,
                attempt,
                total_attempts,
                exc.__class__.__name__,
                exc,
            )
            if not should_retry:
                raise

            delay_seconds = base_delay_seconds * (2 ** (attempt - 1))
            logger.info("[%s] %s retrying in %.2fs", get_request_id(), operation_name, delay_seconds)
            await asyncio.sleep(delay_seconds)

    raise RuntimeError(f"{operation_name} retry loop exited unexpectedly.")
