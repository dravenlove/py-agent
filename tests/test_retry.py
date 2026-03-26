import pytest

from app.retry import run_with_retries


@pytest.mark.anyio
async def test_run_with_retries_succeeds_after_retry() -> None:
    attempts = {"count": 0}

    async def attempt_fn() -> str:
        attempts["count"] += 1
        if attempts["count"] < 2:
            raise RuntimeError("transient")
        return "ok"

    result = await run_with_retries(
        operation_name="test-op",
        attempt_fn=attempt_fn,
        is_retryable=lambda exc: isinstance(exc, RuntimeError),
        max_retries=2,
        base_delay_seconds=0,
    )

    assert result == "ok"
    assert attempts["count"] == 2


@pytest.mark.anyio
async def test_run_with_retries_raises_when_not_retryable() -> None:
    attempts = {"count": 0}

    async def attempt_fn() -> str:
        attempts["count"] += 1
        raise ValueError("fatal")

    with pytest.raises(ValueError, match="fatal"):
        await run_with_retries(
            operation_name="test-op",
            attempt_fn=attempt_fn,
            is_retryable=lambda exc: False,
            max_retries=3,
            base_delay_seconds=0,
        )

    assert attempts["count"] == 1
