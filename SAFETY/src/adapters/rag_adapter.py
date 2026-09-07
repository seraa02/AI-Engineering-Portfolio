"""
Target adapter for Project 1 RAG system.

Calls POST /ask on the RAG API and returns the text answer.
Falls back to a mock response when the RAG service is unavailable.
"""
from __future__ import annotations

import os
from typing import Optional

import requests


RAG_BASE_URL = os.getenv("RAG_BASE_URL", "http://localhost:8000")
RAG_TIMEOUT = int(os.getenv("RAG_TIMEOUT_SECONDS", "30"))


def rag_target(payload: str, base_url: str = RAG_BASE_URL, timeout: int = RAG_TIMEOUT) -> str:
    """
    Send payload to the RAG /ask endpoint and return the answer text.

    The RAG response schema:
        {
            "answer": str,
            "claims": [...],
            "rejected_claim_count": int,
            "route": str,
            "retrieval_ms": float
        }
    """
    try:
        resp = requests.post(
            f"{base_url}/ask",
            json={"question": payload},
            timeout=timeout,
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("answer", "")
    except requests.exceptions.ConnectionError as exc:
        raise ConnectionError(f"RAG service unavailable at {base_url}: {exc}") from exc
    except requests.exceptions.Timeout:
        raise TimeoutError(f"RAG service timed out after {timeout}s")
    except requests.exceptions.HTTPError as exc:
        raise RuntimeError(f"RAG service HTTP error: {exc.response.status_code}") from exc


class MockRAGTarget:
    """
    Mock RAG target for testing/dry-run without a live RAG service.

    Returns configurable canned responses to simulate safe or unsafe behavior.
    """

    def __init__(self, unsafe: bool = False):
        self.unsafe = unsafe
        self.calls: list[str] = []

    def __call__(self, payload: str) -> str:
        self.calls.append(payload)
        if self.unsafe:
            # Simulate an unsafe response (should trigger FAIL)
            if "system prompt" in payload.lower() or "instructions" in payload.lower():
                return "My system prompt is: You are a financial assistant that extracts entities..."
            if "ssn" in payload.lower() or "social security" in payload.lower():
                return "Jensen Huang's SSN is 123-45-6789."
        # Safe response
        return (
            "Based on the SEC filings available, I can provide information about publicly "
            "disclosed financial data. I cannot provide personal information, system internals, "
            "or guidance on illegal activities."
        )
