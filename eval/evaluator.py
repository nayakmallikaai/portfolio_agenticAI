"""
Core evaluation logic.

For each TestCase:
  1. POST /api/analyze with the test request
  2. Run every Check against the response
  3. Produce a CheckResult (pass/fail + reason) per check
  4. Aggregate into a CaseResult with an overall score
"""

import uuid
import time
import requests
from dataclasses import dataclass, field
from typing import List, Optional, Any

from eval.test_cases import (
    TestCase,
    ShouldReject, ShouldHaveTrades,
    TickerInTrades, SideForTicker,
    RiskApproved, SummaryContains,
)


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class CheckResult:
    check_description: str
    passed: bool
    reason: str


@dataclass
class CaseResult:
    test_id: str
    description: str
    notes: str
    passed: bool                    # True only if ALL checks pass
    score: str                      # "PASS" | "FAIL" | "ERROR"
    faithful: bool                  # output matches expected intent
    relevant: bool                  # output addresses the question
    check_results: List[CheckResult] = field(default_factory=list)
    error: Optional[str] = None
    elapsed_s: float = 0.0
    response: Optional[dict] = None


# ── Check evaluators ──────────────────────────────────────────────────────────

def _evaluate_check(check: Any, response: dict) -> CheckResult:
    trades = response.get("proposed_trades", [])
    summary = (response.get("decision_summary") or "").lower()

    if isinstance(check, ShouldReject):
        passed = len(trades) == 0
        return CheckResult(
            check_description=check.description,
            passed=passed,
            reason="0 trades proposed — guardrail held" if passed
                   else f"{len(trades)} trade(s) proposed — guardrail failed",
        )

    if isinstance(check, ShouldHaveTrades):
        passed = len(trades) >= check.min_trades
        return CheckResult(
            check_description=check.description,
            passed=passed,
            reason=f"{len(trades)} trade(s) found (need >= {check.min_trades})",
        )

    if isinstance(check, TickerInTrades):
        tickers = [t["ticker"].upper() for t in trades]
        passed = check.ticker.upper() in tickers
        return CheckResult(
            check_description=check.description,
            passed=passed,
            reason=f"Tickers in trades: {tickers}",
        )

    if isinstance(check, SideForTicker):
        match = next(
            (t for t in trades if t["ticker"].upper() == check.ticker.upper()),
            None,
        )
        if match is None:
            return CheckResult(
                check_description=check.description,
                passed=False,
                reason=f"Ticker {check.ticker} not found in trades",
            )
        passed = match["side"].upper() == check.side.upper()
        return CheckResult(
            check_description=check.description,
            passed=passed,
            reason=f"{check.ticker} proposed as {match['side']} (expected {check.side})",
        )

    if isinstance(check, RiskApproved):
        actual = response.get("risk_approved")
        passed = actual == check.expected
        return CheckResult(
            check_description=check.description,
            passed=passed,
            reason=f"risk_approved={actual} (expected {check.expected})",
        )

    if isinstance(check, SummaryContains):
        hit = next((kw for kw in check.keywords if kw.lower() in summary), None)
        passed = hit is not None
        return CheckResult(
            check_description=check.description,
            passed=passed,
            reason=f"Keyword '{hit}' found" if passed
                   else f"None of {check.keywords} found in summary",
        )

    return CheckResult(
        check_description=str(check),
        passed=False,
        reason="Unknown check type",
    )


# ── Single test runner ────────────────────────────────────────────────────────

def run_test(case: TestCase, base_url: str, user_id: str) -> CaseResult:
    session_id = "eval_" + uuid.uuid4().hex[:8]
    payload = {
        "user_id": user_id,
        "session_id": session_id,
        **case.request,
    }

    start = time.time()
    try:
        resp = requests.post(
            f"{base_url}/api/analyze",
            json=payload,
            timeout=300,
        )
        elapsed = time.time() - start

        if resp.status_code != 200:
            return CaseResult(
                test_id=case.id,
                description=case.description,
                notes=case.notes,
                passed=False,
                score="ERROR",
                faithful=False,
                relevant=False,
                error=f"HTTP {resp.status_code}: {resp.text[:300]}",
                elapsed_s=round(elapsed, 1),
            )

        response = resp.json()

    except Exception as e:
        elapsed = time.time() - start
        return CaseResult(
            test_id=case.id,
            description=case.description,
            notes=case.notes,
            passed=False,
            score="ERROR",
            faithful=False,
            relevant=False,
            error=str(e),
            elapsed_s=round(elapsed, 1),
        )

    # ── Run checks ────────────────────────────────────────────────────────────
    check_results = [_evaluate_check(c, response) for c in case.checks]
    all_pass = all(r.passed for r in check_results)

    # Faithful = output structurally matches expected (all checks pass)
    faithful = all_pass

    # Relevant = model produced a substantive response (has summary or trades)
    relevant = bool(
        response.get("decision_summary") or response.get("proposed_trades")
    )

    return CaseResult(
        test_id=case.id,
        description=case.description,
        notes=case.notes,
        passed=all_pass,
        score="PASS" if all_pass else "FAIL",
        faithful=faithful,
        relevant=relevant,
        check_results=check_results,
        elapsed_s=round(elapsed, 1),
        response=response,
    )


# ── Suite runner ──────────────────────────────────────────────────────────────

def run_suite(
    test_cases: List[TestCase],
    base_url: str = "http://localhost:8000",
    user_id: str = "eval_user",
    test_ids: Optional[List[str]] = None,
) -> List[CaseResult]:
    """
    Run all (or a filtered subset of) test cases and return results.
    test_ids: if provided, only run cases whose id is in this list.
    """
    cases = test_cases
    if test_ids:
        cases = [c for c in test_cases if c.id in test_ids]

    results = []
    for i, case in enumerate(cases):
        print(f"  Running {case.id}: {case.description}…", flush=True)
        result = run_test(case, base_url, user_id)
        results.append(result)
        status = f"\033[92mPASS\033[0m" if result.score == "PASS" else (
                  f"\033[93mERROR\033[0m" if result.score == "ERROR" else
                  f"\033[91mFAIL\033[0m"
        )
        print(f"    → {status} ({result.elapsed_s}s)", flush=True)
        # Brief cooldown between tests so the local LLM isn't overloaded
        if i < len(cases) - 1:
            time.sleep(5)

    return results
