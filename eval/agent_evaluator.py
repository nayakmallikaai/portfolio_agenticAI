"""
Unified evaluator for both the original test suite and the new agent-capability suite.

check evaluation logic:
  - Original checks  (from eval.test_cases)   : ShouldReject, ShouldHaveTrades, etc.
  - Agent checks     (from eval.agent_checks)  : tool call, retry, precision/recall, etc.

compute_metrics() produces aggregate scores across a list of CaseResult objects:
  - guardrail_score      : % of guardrail tests passed
  - tool_call_score      : % of tool-call tests passed
  - retry_score          : % of retry tests passed
  - price_grounding_score: % of grounding tests passed
  - precision_score      : avg context precision across all cases (0-1)
  - recall_score         : avg context recall across all cases (0-1)
  - relevance_score      : avg answer relevance across all cases (0-1)
  - faithfulness_score   : % of faithfulness tests passed
  - overall_score        : % of ALL checks passed across ALL cases
  - avg_tool_calls       : mean tool calls per test
  - avg_retry_count      : mean retry_count per test
  - avg_response_time_s  : mean elapsed seconds per test
"""
import uuid
import time
import requests
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict

from eval.test_cases import (
    TestCase,
    ShouldReject, ShouldHaveTrades, TickerInTrades, SideForTicker,
    RiskApproved, SummaryContains,
)
from eval.agent_checks import (
    ToolWasCalled, ToolNotCalled, ToolCallCount,
    PortfolioFetchedFirst, SpecificTickerFetched, AllHoldingsFetched,
    NoToolsOnOffTopic, ToolCallOrder,
    RetryOccurred, RetryConverged, MaxRetriesHit, RetryCountAtMost,
    PriceGrounded,
    ContextPrecision, ContextRecall, AnswerRelevance, Faithfulness,
    GetPricesBatchCalled, GetPricesBatchNotCalled, BatchTickerCount,
    TradeCountAtMost,
)


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class CheckResult:
    check_description: str
    passed: bool
    reason: str
    check_category: str = "general"   # "tool_call" | "retry" | "grounding" | "precision" |
                                       # "recall"    | "relevance" | "faithfulness" | "guardrail"


@dataclass
class CaseResult:
    test_id: str
    description: str
    notes: str
    passed: bool
    score: str           # "PASS" | "FAIL" | "ERROR"
    faithful: bool
    relevant: bool
    check_results: List[CheckResult] = field(default_factory=list)
    error: Optional[str] = None
    elapsed_s: float = 0.0
    response: Optional[dict] = None
    # Computed per-case metrics (filled by evaluator)
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    answer_relevance: Optional[float] = None
    price_grounding: Optional[float] = None
    tool_calls_count: int = 0
    retry_count: int = 0


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tool_log(response: dict) -> List[dict]:
    return response.get("tool_calls_log") or []


def _fetched_tickers(response: dict) -> List[str]:
    """Return all tickers for which a price was fetched (single or batch)."""
    tickers = []
    for e in _tool_log(response):
        if e.get("name") == "get_live_price" and e.get("ticker"):
            tickers.append(e["ticker"].upper())
        elif e.get("name") == "get_prices_batch" and e.get("tickers"):
            tickers.extend(t.upper() for t in e["tickers"])
    return tickers


def _summary(response: dict) -> str:
    return (response.get("decision_summary") or "").lower()


def _trades(response: dict) -> List[dict]:
    return response.get("proposed_trades") or []


# ── Per-check evaluators ──────────────────────────────────────────────────────

def _evaluate_check(check: Any, response: dict) -> CheckResult:
    log   = _tool_log(response)
    tools = [e["name"] for e in log]
    tickers_fetched = _fetched_tickers(response)
    summary = _summary(response)
    trades  = _trades(response)
    retry   = response.get("retry_count", 0)

    # ── Original checks ───────────────────────────────────────────────────────
    if isinstance(check, ShouldReject):
        passed = len(trades) == 0
        return CheckResult(
            check.description, passed,
            "0 trades — guardrail held" if passed else f"{len(trades)} trade(s) — guardrail failed",
            "guardrail",
        )

    if isinstance(check, ShouldHaveTrades):
        passed = len(trades) >= check.min_trades
        return CheckResult(
            check.description, passed,
            f"{len(trades)} trade(s) found (need >= {check.min_trades})",
            "general",
        )

    if isinstance(check, TickerInTrades):
        tickers = [t["ticker"].upper() for t in trades]
        passed = check.ticker.upper() in tickers
        return CheckResult(
            check.description, passed, f"Tickers in trades: {tickers}", "general",
        )

    if isinstance(check, SideForTicker):
        match = next((t for t in trades if t["ticker"].upper() == check.ticker.upper()), None)
        if match is None:
            return CheckResult(check.description, False, f"{check.ticker} not found in trades", "general")
        passed = match["side"].upper() == check.side.upper()
        return CheckResult(
            check.description, passed,
            f"{check.ticker} proposed as {match['side']} (expected {check.side})", "general",
        )

    if isinstance(check, RiskApproved):
        actual = response.get("risk_approved")
        passed = actual == check.expected
        return CheckResult(
            check.description, passed,
            f"risk_approved={actual} (expected {check.expected})", "guardrail",
        )

    if isinstance(check, SummaryContains):
        hit = next((kw for kw in check.keywords if kw.lower() in summary), None)
        passed = hit is not None
        return CheckResult(
            check.description, passed,
            f"Keyword '{hit}' found" if passed else f"None of {check.keywords} found",
            "relevance",
        )

    # ── Tool call checks ──────────────────────────────────────────────────────
    if isinstance(check, ToolWasCalled):
        passed = check.tool_name in tools
        return CheckResult(
            check.description, passed,
            f"Tools called: {tools}" if not passed else f"'{check.tool_name}' found in log",
            "tool_call",
        )

    if isinstance(check, ToolNotCalled):
        passed = check.tool_name not in tools
        return CheckResult(
            check.description, passed,
            f"'{check.tool_name}' absent" if passed else f"'{check.tool_name}' was called — must not be",
            "tool_call",
        )

    if isinstance(check, ToolCallCount):
        n = len(log)
        passed = check.min_calls <= n <= check.max_calls
        return CheckResult(
            check.description, passed,
            f"{n} tool call(s) (allowed [{check.min_calls}, {check.max_calls}])",
            "tool_call",
        )

    if isinstance(check, PortfolioFetchedFirst):
        portfolio_orders = [e["order"] for e in log if e["name"] == "get_portfolio"]
        price_orders     = [e["order"] for e in log if e["name"] == "get_live_price"]
        if not portfolio_orders:
            return CheckResult(check.description, False, "get_portfolio never called", "tool_call")
        if not price_orders:
            return CheckResult(check.description, True, "No price calls — trivially satisfied", "tool_call")
        passed = min(portfolio_orders) < min(price_orders)
        return CheckResult(
            check.description, passed,
            f"portfolio order={min(portfolio_orders)}, first price order={min(price_orders)}",
            "tool_call",
        )

    if isinstance(check, SpecificTickerFetched):
        passed = check.ticker.upper() in tickers_fetched
        return CheckResult(
            check.description, passed,
            f"Tickers fetched: {tickers_fetched}",
            "tool_call",
        )

    if isinstance(check, AllHoldingsFetched):
        missing = [t for t in check.seed_tickers if t.upper() not in tickers_fetched]
        passed = len(missing) == 0
        return CheckResult(
            check.description, passed,
            f"All fetched: {tickers_fetched}" if passed else f"Missing: {missing}",
            "tool_call",
        )

    if isinstance(check, NoToolsOnOffTopic):
        passed = len(log) == 0
        return CheckResult(
            check.description, passed,
            "0 tool calls" if passed else f"{len(log)} tool calls made — should be 0",
            "tool_call",
        )

    if isinstance(check, ToolCallOrder):
        first_orders  = [e["order"] for e in log if e["name"] == check.first]
        second_orders = [e["order"] for e in log if e["name"] == check.second]
        if not first_orders:
            return CheckResult(check.description, False, f"'{check.first}' never called", "tool_call")
        if not second_orders:
            return CheckResult(check.description, True, f"'{check.second}' never called — trivially satisfied", "tool_call")
        passed = min(first_orders) < min(second_orders)
        return CheckResult(
            check.description, passed,
            f"'{check.first}' order={min(first_orders)}, '{check.second}' order={min(second_orders)}",
            "tool_call",
        )

    # ── Retry checks ──────────────────────────────────────────────────────────
    if isinstance(check, RetryOccurred):
        passed = retry >= 1
        return CheckResult(check.description, passed, f"retry_count={retry}", "retry")

    if isinstance(check, RetryConverged):
        passed = response.get("risk_approved", False)
        return CheckResult(
            check.description, passed,
            f"risk_approved={response.get('risk_approved')} after {retry} retries",
            "retry",
        )

    if isinstance(check, MaxRetriesHit):
        passed = retry >= check.max_retries
        return CheckResult(
            check.description, passed,
            f"retry_count={retry} (need >= {check.max_retries})",
            "retry",
        )

    if isinstance(check, RetryCountAtMost):
        passed = retry <= check.n
        return CheckResult(
            check.description, passed,
            f"retry_count={retry} (need <= {check.n})",
            "retry",
        )

    # ── Price grounding ───────────────────────────────────────────────────────
    if isinstance(check, PriceGrounded):
        if not trades:
            return CheckResult(check.description, True, "No trades — trivially grounded", "grounding")

        price_map: Dict[str, float] = {}
        for e in log:
            if e.get("name") == "get_live_price" and e.get("ticker"):
                try:
                    price_map[e["ticker"].upper()] = float(e["result"])
                except (ValueError, TypeError):
                    pass

        grounded, total = 0, 0
        details = []
        for t in trades:
            tkr = t.get("ticker", "").upper()
            proposed = t.get("proposed_price") or t.get("price")
            live = price_map.get(tkr)
            if proposed is None or live is None or live == 0:
                details.append(f"{tkr}: no price data")
                continue
            total += 1
            deviation = abs(proposed - live) / live
            if deviation <= check.tolerance:
                grounded += 1
                details.append(f"{tkr}: {deviation:.2%} ✓")
            else:
                details.append(f"{tkr}: {deviation:.2%} ✗ (proposed={proposed}, live={live})")

        fraction = grounded / total if total else 1.0
        passed = fraction >= check.min_grounded_fraction
        return CheckResult(
            check.description, passed,
            f"Grounded {grounded}/{total} trades ({fraction:.0%}). {'; '.join(details)}",
            "grounding",
        )

    # ── LLM quality: precision ────────────────────────────────────────────────
    if isinstance(check, ContextPrecision):
        fetched = set(tickers_fetched)
        if not fetched:
            return CheckResult(check.description, True, "No tickers fetched — trivially precise", "precision")
        used = {t.upper() for t in fetched if t.lower() in summary}
        # also count tickers appearing in proposed trades as "used"
        used |= {t["ticker"].upper() for t in trades if t.get("ticker", "").upper() in fetched}
        precision = len(used) / len(fetched)
        passed = precision >= check.min_precision
        return CheckResult(
            check.description, passed,
            f"Precision={precision:.2f} ({len(used)} used / {len(fetched)} fetched). Used: {used}, Fetched: {fetched}",
            "precision",
        )

    # ── LLM quality: recall ───────────────────────────────────────────────────
    if isinstance(check, ContextRecall):
        portfolio = [t.upper() for t in check.portfolio_tickers]
        mentioned = [t for t in portfolio if t.lower() in summary]
        # also count if in trade tickers
        mentioned_set = set(mentioned) | {t["ticker"].upper() for t in trades if t.get("ticker", "").upper() in portfolio}
        recall = len(mentioned_set) / len(portfolio) if portfolio else 1.0
        passed = recall >= check.min_recall
        return CheckResult(
            check.description, passed,
            f"Recall={recall:.2f} ({len(mentioned_set)}/{len(portfolio)} holdings). Mentioned: {mentioned_set}",
            "recall",
        )

    # ── LLM quality: answer relevance ─────────────────────────────────────────
    if isinstance(check, AnswerRelevance):
        hits = [kw for kw in check.keywords if kw.lower() in summary]
        passed = len(hits) >= check.min_keywords
        return CheckResult(
            check.description, passed,
            f"Found {len(hits)}/{check.min_keywords} required keywords: {hits}",
            "relevance",
        )

    # ── LLM quality: faithfulness ─────────────────────────────────────────────
    if isinstance(check, Faithfulness):
        if not trades:
            return CheckResult(check.description, True, "No trades — trivially faithful", "faithfulness")
        fetched = set(tickers_fetched)
        unfaithful = [t["ticker"].upper() for t in trades if t.get("ticker", "").upper() not in fetched]
        passed = len(unfaithful) == 0
        return CheckResult(
            check.description, passed,
            "All trade tickers had prices fetched" if passed
            else f"Tickers in trades without fetched price: {unfaithful}",
            "faithfulness",
        )

    # ── Batch pricing checks ──────────────────────────────────────────────────
    if isinstance(check, GetPricesBatchCalled):
        batch_calls = [e for e in log if e.get("name") == "get_prices_batch"]
        passed = len(batch_calls) > 0
        return CheckResult(
            check.description, passed,
            f"get_prices_batch called {len(batch_calls)} time(s)" if passed
            else "get_prices_batch was never called",
            "tool_call",
        )

    if isinstance(check, GetPricesBatchNotCalled):
        batch_calls = [e for e in log if e.get("name") == "get_prices_batch"]
        passed = len(batch_calls) == 0
        return CheckResult(
            check.description, passed,
            "get_prices_batch absent — correct for targeted query" if passed
            else f"get_prices_batch was called {len(batch_calls)} time(s) — should not be",
            "tool_call",
        )

    if isinstance(check, BatchTickerCount):
        batch_calls = [e for e in log if e.get("name") == "get_prices_batch"]
        if not batch_calls:
            return CheckResult(check.description, False, "get_prices_batch was never called", "tool_call")
        max_count = max(len(e.get("tickers") or []) for e in batch_calls)
        passed = max_count >= check.min_tickers
        return CheckResult(
            check.description, passed,
            f"Largest batch call had {max_count} tickers (need >= {check.min_tickers})",
            "tool_call",
        )

    # ── Trade count check ─────────────────────────────────────────────────────
    if isinstance(check, TradeCountAtMost):
        n = len(trades)
        passed = n <= check.max_trades
        return CheckResult(
            check.description, passed,
            f"{n} trade(s) proposed (limit <= {check.max_trades})",
            "guardrail",
        )

    return CheckResult(str(check), False, "Unknown check type", "general")


# ── Single test runner ────────────────────────────────────────────────────────

def run_test(case: TestCase, base_url: str, user_id: str) -> CaseResult:
    session_id = "eval_" + uuid.uuid4().hex[:8]
    payload = {"user_id": user_id, "session_id": session_id, **case.request}

    start = time.time()
    try:
        resp = requests.post(f"{base_url}/api/analyze", json=payload, timeout=300)
        elapsed = time.time() - start

        if resp.status_code != 200:
            return CaseResult(
                test_id=case.id, description=case.description, notes=case.notes,
                passed=False, score="ERROR", faithful=False, relevant=False,
                error=f"HTTP {resp.status_code}: {resp.text[:300]}",
                elapsed_s=round(elapsed, 1),
            )
        response = resp.json()

    except Exception as e:
        elapsed = time.time() - start
        return CaseResult(
            test_id=case.id, description=case.description, notes=case.notes,
            passed=False, score="ERROR", faithful=False, relevant=False,
            error=str(e), elapsed_s=round(elapsed, 1),
        )

    check_results = [_evaluate_check(c, response) for c in case.checks]
    all_pass = all(r.passed for r in check_results)

    # Compute per-case numeric metrics
    fetched = set(_fetched_tickers(response))
    summary = _summary(response)
    trades  = _trades(response)
    log     = _tool_log(response)

    # Context precision
    if fetched:
        used = fetched & {t.lower() for t in fetched if t.lower() in summary}
        used |= {t["ticker"].upper() for t in trades if t.get("ticker", "").upper() in fetched}
        cp = len(used) / len(fetched)
    else:
        cp = None

    # Context recall over seed holdings (DJI 30 seed: AAPL, MSFT, JPM)
    seed = {"AAPL", "MSFT", "JPM"}
    mentioned = seed & {t.upper() for t in seed if t.lower() in summary}
    mentioned |= {t["ticker"].upper() for t in trades if t.get("ticker", "").upper() in seed}
    cr = len(mentioned) / len(seed)

    # Answer relevance: fraction of summary words that are non-trivial
    # Simple heuristic: check for financial signal words
    signal_words = [
        "buy", "sell", "portfolio", "cash", "price", "risk", "trade", "stock",
        "holdings", "concentration", "diversif", "rebalanc", "sector", "equity",
    ]
    ar_hits = sum(1 for w in signal_words if w in summary)
    ar = min(ar_hits / 5, 1.0)   # normalised; 5+ hits = 1.0

    return CaseResult(
        test_id=case.id, description=case.description, notes=case.notes,
        passed=all_pass, score="PASS" if all_pass else "FAIL",
        faithful=all_pass,
        relevant=bool(response.get("decision_summary") or trades),
        check_results=check_results,
        elapsed_s=round(elapsed, 1),
        response=response,
        context_precision=round(cp, 3) if cp is not None else None,
        context_recall=round(cr, 3),
        answer_relevance=round(ar, 3),
        price_grounding=None,   # populated per-check above; aggregate computed in compute_metrics
        tool_calls_count=len(log),
        retry_count=response.get("retry_count", 0),
    )


# ── Suite runner ──────────────────────────────────────────────────────────────

def run_suite(
    test_cases: List[TestCase],
    base_url: str = "http://localhost:8000",
    user_id: str = "eval_user",
    test_ids: Optional[List[str]] = None,
    sleep_between: int = 5,
) -> List[CaseResult]:
    cases = test_cases
    if test_ids:
        cases = [c for c in test_cases if c.id in test_ids]

    results = []
    for i, case in enumerate(cases):
        print(f"  Running {case.id}: {case.description}…", flush=True)
        result = run_test(case, base_url, user_id)
        results.append(result)
        status = (
            "\033[92mPASS\033[0m" if result.score == "PASS"
            else "\033[93mERROR\033[0m" if result.score == "ERROR"
            else "\033[91mFAIL\033[0m"
        )
        print(f"    → {status} ({result.elapsed_s}s)", flush=True)
        if i < len(cases) - 1:
            time.sleep(sleep_between)

    return results


# ── Aggregate metrics ─────────────────────────────────────────────────────────

def compute_metrics(results: List[CaseResult]) -> Dict[str, Any]:
    """
    Compute aggregate metrics across all CaseResult objects.
    Returns a dict suitable for the report generator.
    """
    if not results:
        return {}

    total = len(results)
    passed = sum(1 for r in results if r.score == "PASS")
    errors = sum(1 for r in results if r.score == "ERROR")

    # Bucket results by category based on which checks they contain
    def has_category(result: CaseResult, cat: str) -> bool:
        return any(cr.check_category == cat for cr in result.check_results)

    def category_pass_rate(cat: str) -> Optional[float]:
        bucket = [r for r in results if has_category(r, cat) and r.score != "ERROR"]
        if not bucket:
            return None
        cat_checks = [
            cr for r in bucket for cr in r.check_results if cr.check_category == cat
        ]
        if not cat_checks:
            return None
        return round(sum(1 for c in cat_checks if c.passed) / len(cat_checks), 3)

    # Per-case numeric averages (exclude ERROR results)
    valid = [r for r in results if r.score != "ERROR"]

    def avg(values):
        v = [x for x in values if x is not None]
        return round(sum(v) / len(v), 3) if v else None

    # Grounding: extract from PriceGrounded check results
    grounding_checks = [
        cr for r in valid for cr in r.check_results if cr.check_category == "grounding"
    ]
    grounding_pass = (
        round(sum(1 for c in grounding_checks if c.passed) / len(grounding_checks), 3)
        if grounding_checks else None
    )

    return {
        "overall_score":          round(passed / total, 3),
        "pass_count":             passed,
        "fail_count":             sum(1 for r in results if r.score == "FAIL"),
        "error_count":            errors,
        "total_count":            total,
        # Category scores
        "guardrail_score":        category_pass_rate("guardrail"),
        "tool_call_score":        category_pass_rate("tool_call"),
        "retry_score":            category_pass_rate("retry"),
        "price_grounding_score":  grounding_pass,
        "faithfulness_score":     category_pass_rate("faithfulness"),
        # LLM eval metrics
        "avg_context_precision":  avg(r.context_precision for r in valid),
        "avg_context_recall":     avg(r.context_recall for r in valid),
        "avg_answer_relevance":   avg(r.answer_relevance for r in valid),
        # Efficiency metrics
        "avg_tool_calls":         avg(r.tool_calls_count for r in valid),
        "avg_retry_count":        avg(r.retry_count for r in valid),
        "avg_response_time_s":    avg(r.elapsed_s for r in results),
    }
