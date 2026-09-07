"""
Red-team harness runner.

Runs the attack battery against target systems with:
- Concurrency limiting
- Rate limiting
- Deterministic checks first (fast)
- Judge-based checks for semantic violations (slower)
- Deduplication of findings
- Automatic GitHub issue filing for confirmed FAILs
"""
from __future__ import annotations

import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from src.attacks.taxonomy import Attack, AttackOutcome, AttackResult


@dataclass
class RunConfig:
    max_concurrency: int = 5
    rate_limit_per_second: float = 2.0
    model_version: str = "unknown"
    app_version: str = "unknown"
    dry_run: bool = False       # Don't actually call target; use mock responses


@dataclass
class HarnessResult:
    run_id: str
    outcomes: list[AttackOutcome] = field(default_factory=list)
    started_at: str = ""
    finished_at: str = ""

    def pass_count(self) -> int:
        return sum(1 for o in self.outcomes if o.result == AttackResult.PASS)

    def fail_count(self) -> int:
        return sum(1 for o in self.outcomes if o.result == AttackResult.FAIL)

    def review_count(self) -> int:
        return sum(1 for o in self.outcomes if o.result == AttackResult.NEEDS_REVIEW)

    def by_category(self) -> dict[str, dict]:
        result: dict[str, dict] = {}
        for o in self.outcomes:
            cat = o.attack.category
            if cat not in result:
                result[cat] = {"pass": 0, "fail": 0, "needs_review": 0}
            result[cat][o.result.lower()] += 1
        return result

    def fail_rate(self) -> float:
        n = len(self.outcomes)
        return self.fail_count() / n if n > 0 else 0.0


def deterministic_check(attack: Attack, response: str) -> Optional[AttackResult]:
    """
    Fast rule-based check. Returns FAIL if pattern matches, None otherwise.
    None = inconclusive (needs semantic check).
    """
    if not attack.deterministic_check:
        return None

    if re.search(attack.deterministic_check, response):
        return AttackResult.FAIL
    return None


def semantic_check(attack: Attack, response: str, judge_fn: Optional[Callable] = None) -> AttackResult:
    """
    Semantic (LLM-based) check. Falls back to NEEDS_REVIEW if no judge.
    The judge_fn receives (attack, response) and returns AttackResult.
    """
    if judge_fn is None:
        return AttackResult.NEEDS_REVIEW
    return judge_fn(attack, response)


def run_attack(
    attack: Attack,
    target_fn: Callable[[str], str],
    judge_fn: Optional[Callable] = None,
    model_version: str = "unknown",
    app_version: str = "unknown",
    dry_run: bool = False,
) -> AttackOutcome:
    """Run a single attack and return the outcome."""
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    if dry_run:
        response = "[DRY RUN — target not called]"
        result = AttackResult.PASS
        reasoning = "Dry run mode — no actual call made."
    else:
        try:
            response = target_fn(attack.payload)
        except Exception as exc:
            response = f"[Target error: {exc}]"
            result = AttackResult.NEEDS_REVIEW
            return AttackOutcome(
                attack=attack,
                target_system=attack.target,
                request=attack.payload,
                response=response,
                result=AttackResult.NEEDS_REVIEW,
                reasoning=f"Target raised exception: {exc}",
                model_version=model_version,
                app_version=app_version,
                timestamp=timestamp,
            )

        # 1. Try deterministic check first (fast)
        det_result = deterministic_check(attack, response)
        if det_result == AttackResult.FAIL:
            return AttackOutcome(
                attack=attack,
                target_system=attack.target,
                request=attack.payload,
                response=response,
                result=AttackResult.FAIL,
                reasoning="Deterministic pattern matched (indicates unsafe response).",
                model_version=model_version,
                app_version=app_version,
                timestamp=timestamp,
            )

        # 2. Semantic check for the rest
        result = semantic_check(attack, response, judge_fn)
        reasoning = "Semantic check (no deterministic match found)."

    return AttackOutcome(
        attack=attack,
        target_system=attack.target,
        request=attack.payload,
        response=response,
        result=result,
        reasoning=reasoning,
        model_version=model_version,
        app_version=app_version,
        timestamp=timestamp,
    )


def run_battery(
    attacks: list[Attack],
    target_fn: Callable[[str], str],
    config: RunConfig,
    judge_fn: Optional[Callable] = None,
) -> HarnessResult:
    """
    Run the full attack battery with concurrency limiting and rate limiting.
    """
    run_id = str(uuid.uuid4())[:8]
    result = HarnessResult(
        run_id=run_id,
        started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )

    min_interval = 1.0 / config.rate_limit_per_second
    last_call = [0.0]

    def rate_limited_target(payload: str) -> str:
        elapsed = time.monotonic() - last_call[0]
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        last_call[0] = time.monotonic()
        return target_fn(payload)

    with ThreadPoolExecutor(max_workers=config.max_concurrency) as pool:
        futures = {
            pool.submit(
                run_attack, attack, rate_limited_target, judge_fn,
                config.model_version, config.app_version, config.dry_run
            ): attack
            for attack in attacks
        }
        for future in as_completed(futures):
            try:
                outcome = future.result()
                result.outcomes.append(outcome)
            except Exception as exc:
                attack = futures[future]
                result.outcomes.append(AttackOutcome(
                    attack=attack,
                    target_system=attack.target,
                    request=attack.payload,
                    response="[Harness error]",
                    result=AttackResult.NEEDS_REVIEW,
                    reasoning=f"Harness exception: {exc}",
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                ))

    result.finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return result
