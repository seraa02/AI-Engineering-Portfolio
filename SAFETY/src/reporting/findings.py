"""
Finding storage, deduplication, and GitHub issue automation.

Findings are deduplicated by (attack_id, response_signature) to avoid
filing duplicate issues for the same root vulnerability.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from src.attacks.taxonomy import AttackOutcome, AttackResult


DB_PATH = Path(os.getenv("SAFETY_DB_PATH", "data/findings.db"))


# ---------------------------------------------------------------------------
# Finding record
# ---------------------------------------------------------------------------

@dataclass
class Finding:
    run_id: str
    attack_id: str
    category: str
    severity: str
    target_system: str
    payload: str
    response: str
    result: str           # PASS | FAIL | NEEDS_REVIEW
    reasoning: str
    response_signature: str   # sha256 of response (for deduplication)
    model_version: str
    app_version: str
    timestamp: str
    github_issue_url: Optional[str] = None
    deduplicated: bool = False


def _response_signature(response: str) -> str:
    """Short fingerprint of a response for deduplication."""
    return hashlib.sha256(response.encode()).hexdigest()[:16]


def outcome_to_finding(outcome: AttackOutcome, run_id: str) -> Finding:
    return Finding(
        run_id=run_id,
        attack_id=outcome.attack.id,
        category=str(outcome.attack.category),
        severity=str(outcome.attack.severity),
        target_system=outcome.target_system,
        payload=outcome.request,
        response=outcome.response,
        result=str(outcome.result),
        reasoning=outcome.reasoning,
        response_signature=_response_signature(outcome.response),
        model_version=outcome.model_version,
        app_version=outcome.app_version,
        timestamp=outcome.timestamp,
    )


# ---------------------------------------------------------------------------
# SQLite storage
# ---------------------------------------------------------------------------

def _connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path = DB_PATH) -> None:
    """Create tables if they don't exist."""
    with _connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS findings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                attack_id TEXT NOT NULL,
                category TEXT NOT NULL,
                severity TEXT NOT NULL,
                target_system TEXT NOT NULL,
                payload TEXT NOT NULL,
                response TEXT NOT NULL,
                result TEXT NOT NULL,
                reasoning TEXT NOT NULL,
                response_signature TEXT NOT NULL,
                model_version TEXT NOT NULL,
                app_version TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                github_issue_url TEXT,
                deduplicated INTEGER DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_attack_sig
            ON findings (attack_id, response_signature)
        """)


def insert_finding(finding: Finding, db_path: Path = DB_PATH) -> bool:
    """
    Insert a finding. Returns True if inserted, False if deduplicated.

    Deduplication: if (attack_id, response_signature) already exists,
    we mark the new finding as deduplicated and skip insertion.
    """
    with _connect(db_path) as conn:
        existing = conn.execute(
            "SELECT id FROM findings WHERE attack_id = ? AND response_signature = ?",
            (finding.attack_id, finding.response_signature),
        ).fetchone()

        if existing:
            return False  # Deduplicated

        conn.execute(
            """
            INSERT INTO findings
                (run_id, attack_id, category, severity, target_system, payload,
                 response, result, reasoning, response_signature, model_version,
                 app_version, timestamp, github_issue_url, deduplicated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                finding.run_id, finding.attack_id, finding.category, finding.severity,
                finding.target_system, finding.payload, finding.response, finding.result,
                finding.reasoning, finding.response_signature, finding.model_version,
                finding.app_version, finding.timestamp, finding.github_issue_url,
                int(finding.deduplicated),
            ),
        )
        return True


def get_findings(
    result_filter: Optional[str] = None,
    db_path: Path = DB_PATH,
) -> list[Finding]:
    with _connect(db_path) as conn:
        if result_filter:
            rows = conn.execute(
                "SELECT * FROM findings WHERE result = ? ORDER BY timestamp DESC",
                (result_filter,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM findings ORDER BY timestamp DESC"
            ).fetchall()

    findings = []
    for row in rows:
        f = Finding(
            run_id=row["run_id"],
            attack_id=row["attack_id"],
            category=row["category"],
            severity=row["severity"],
            target_system=row["target_system"],
            payload=row["payload"],
            response=row["response"],
            result=row["result"],
            reasoning=row["reasoning"],
            response_signature=row["response_signature"],
            model_version=row["model_version"],
            app_version=row["app_version"],
            timestamp=row["timestamp"],
            github_issue_url=row["github_issue_url"],
            deduplicated=bool(row["deduplicated"]),
        )
        findings.append(f)
    return findings


# ---------------------------------------------------------------------------
# GitHub issue automation
# ---------------------------------------------------------------------------

def _build_issue_body(finding: Finding) -> str:
    return f"""## Red-Team Finding: {finding.attack_id}

**Severity:** {finding.severity}
**Category:** {finding.category}
**Target:** {finding.target_system}
**Run ID:** {finding.run_id}
**Timestamp:** {finding.timestamp}
**Model version:** {finding.model_version}
**App version:** {finding.app_version}

---

### Attack Payload

```
{finding.payload}
```

### System Response

```
{finding.response[:2000]}{"... [truncated]" if len(finding.response) > 2000 else ""}
```

### Verdict

**{finding.result}** — {finding.reasoning}

---

*Filed automatically by the red-team harness.*
"""


def file_github_issue(
    finding: Finding,
    repo: str,
    token: str,
    dry_run: bool = False,
) -> Optional[str]:
    """
    File a GitHub issue for a confirmed FAIL finding.
    Returns the issue URL, or None on failure / dry_run.

    repo: "owner/repo"
    token: GitHub personal access token
    """
    if dry_run:
        return f"https://github.com/{repo}/issues/DRY_RUN"

    try:
        import urllib.request

        title = f"[Red-Team FAIL] {finding.attack_id} — {finding.category} ({finding.severity})"
        body = _build_issue_body(finding)

        payload = json.dumps({"title": title, "body": body, "labels": ["security", "red-team"]})
        req = urllib.request.Request(
            f"https://api.github.com/repos/{repo}/issues",
            data=payload.encode(),
            headers={
                "Authorization": f"token {token}",
                "Content-Type": "application/json",
                "Accept": "application/vnd.github.v3+json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            return data.get("html_url")
    except Exception as exc:
        print(f"[WARNING] Failed to file GitHub issue: {exc}")
        return None


def store_and_report(
    outcomes: list[AttackOutcome],
    run_id: str,
    db_path: Path = DB_PATH,
    github_repo: Optional[str] = None,
    github_token: Optional[str] = None,
    dry_run: bool = False,
) -> tuple[int, int]:
    """
    Convert outcomes → findings, store (with deduplication), file GitHub issues for FAILs.
    Returns (inserted_count, deduplicated_count).
    """
    init_db(db_path)
    inserted = 0
    deduped = 0

    for outcome in outcomes:
        if outcome.result == AttackResult.PASS:
            continue  # Don't store PASSes — only failures and reviews

        finding = outcome_to_finding(outcome, run_id)

        was_inserted = insert_finding(finding, db_path)
        if was_inserted:
            inserted += 1
            # Auto-file GitHub issue for confirmed FAILs
            if (
                outcome.result == AttackResult.FAIL
                and github_repo
                and github_token
            ):
                url = file_github_issue(finding, github_repo, github_token, dry_run=dry_run)
                if url:
                    # Update the DB record with the issue URL
                    with _connect(db_path) as conn:
                        conn.execute(
                            "UPDATE findings SET github_issue_url = ? WHERE attack_id = ? AND response_signature = ?",
                            (url, finding.attack_id, finding.response_signature),
                        )
        else:
            deduped += 1

    return inserted, deduped
