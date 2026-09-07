"""Exponential backoff + jitter retry logic."""
from __future__ import annotations

import random
import time
from typing import Callable, TypeVar

T = TypeVar("T")


class RetryExhausted(Exception):
    """All retry attempts failed."""
    pass


def retry_with_backoff(
    fn: Callable[[], T],
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retryable_exceptions: tuple = (Exception,),
) -> T:
    """
    Retry fn() up to max_attempts times with exponential backoff + jitter.
    Delay = min(base_delay * 2^attempt + uniform(0, base_delay), max_delay)
    """
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except retryable_exceptions as exc:
            last_exc = exc
            if attempt < max_attempts - 1:
                delay = min(base_delay * (2 ** attempt) + random.uniform(0, base_delay), max_delay)
                time.sleep(delay)
    raise RetryExhausted(f"All {max_attempts} attempts failed. Last error: {last_exc}") from last_exc


def idempotency_key(request_id: str, provider: str) -> str:
    """Generate a per-provider idempotency key from the request_id."""
    return f"idem:{request_id}:{provider}"
