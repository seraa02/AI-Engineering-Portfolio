"""Hedged request execution: send to 2 providers, use first response."""
from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Callable

from src.metrics import gateway_hedge_overhead_total


def hedged_call(
    primary_fn: Callable,
    secondary_fn: Callable,
    timeout_seconds: float = 2.0,
) -> dict:
    """
    Execute two provider calls concurrently. Return the first successful response.
    Record the secondary call as overhead in metrics.

    Both functions should return a dict with at least {"response": ..., "provider": ...}.
    """
    results: list[dict] = []
    errors: list[Exception] = []

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures: dict[Future, str] = {}
        futures[executor.submit(primary_fn)] = "primary"
        futures[executor.submit(secondary_fn)] = "secondary"

        for future in as_completed(futures, timeout=timeout_seconds * 2):
            try:
                result = future.result(timeout=timeout_seconds)
                results.append(result)
                # Cancel remaining futures (best-effort)
                for f in futures:
                    if f is not future:
                        f.cancel()
                break
            except Exception as exc:
                errors.append(exc)

    if results:
        gateway_hedge_overhead_total.inc()
        return results[0]

    if errors:
        raise errors[0]

    raise RuntimeError("Hedged call: no result and no error (timeout?)")
