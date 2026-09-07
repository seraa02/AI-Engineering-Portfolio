"""
Target adapter for Project 2 Multi-Agent Research Assistant.

Calls POST /research to start a run, then polls GET /research/{id}
until the run is complete (or times out), and returns the final report.
"""
from __future__ import annotations

import os
import time

import requests


AGENT_BASE_URL = os.getenv("AGENT_BASE_URL", "http://localhost:8001")
AGENT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT_SECONDS", "120"))
AGENT_POLL_INTERVAL = float(os.getenv("AGENT_POLL_INTERVAL_SECONDS", "2.0"))


def agent_target(
    payload: str,
    base_url: str = AGENT_BASE_URL,
    timeout: int = AGENT_TIMEOUT,
    poll_interval: float = AGENT_POLL_INTERVAL,
) -> str:
    """
    Submit a research question to the agent system and wait for the report.

    Agent response schema (GET /research/{id}):
        {
            "run_id": str,
            "status": "pending" | "running" | "completed" | "failed",
            "report": str | None,
            "error": str | None,
        }
    """
    try:
        # Start research run
        start_resp = requests.post(
            f"{base_url}/research",
            json={"question": payload},
            timeout=30,
            headers={"Content-Type": "application/json"},
        )
        start_resp.raise_for_status()
        run_id = start_resp.json()["run_id"]

        # Poll for completion
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status_resp = requests.get(
                f"{base_url}/research/{run_id}",
                timeout=10,
            )
            status_resp.raise_for_status()
            data = status_resp.json()

            if data["status"] == "completed":
                return data.get("report") or ""
            if data["status"] == "failed":
                return f"[Agent failed: {data.get('error', 'unknown error')}]"

            time.sleep(poll_interval)

        return f"[Agent timed out after {timeout}s for run {run_id}]"

    except requests.exceptions.ConnectionError as exc:
        raise ConnectionError(f"Agent service unavailable at {base_url}: {exc}") from exc
    except requests.exceptions.Timeout:
        raise TimeoutError(f"Agent service timed out")
    except requests.exceptions.HTTPError as exc:
        raise RuntimeError(f"Agent service HTTP error: {exc.response.status_code}") from exc


class MockAgentTarget:
    """
    Mock agent target for testing/dry-run without a live agent service.
    """

    def __init__(self, unsafe: bool = False):
        self.unsafe = unsafe
        self.calls: list[str] = []

    def __call__(self, payload: str) -> str:
        self.calls.append(payload)
        if self.unsafe:
            if "shell" in payload.lower() or "rm -rf" in payload.lower():
                return "Shell command executed successfully. Files removed."
            if "system prompt" in payload.lower():
                return "My instructions are: You are a research agent with access to search tools..."
        # Safe response
        return (
            "I conducted research on your question using publicly available information. "
            "I'm unable to perform actions outside my sanctioned scope, execute system "
            "commands, or provide guidance on illegal activities."
        )
