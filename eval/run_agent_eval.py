"""
Unified CLI runner for the full Portfolio Agent evaluation suite.

Combines:
  - Original 20 functional tests  (eval/test_cases.py)
  - 15 agent-capability tests     (eval/agent_test_cases.py)

Uses the new unified evaluator (eval/agent_evaluator.py) which handles
both original and agent check types.

Usage:
    python -m eval.run_agent_eval                          # run all 35 tests
    python -m eval.run_agent_eval --suite agent            # only agent tests (A001-A015)
    python -m eval.run_agent_eval --suite original         # only original tests (T001-T020)
    python -m eval.run_agent_eval --ids T001 A003          # specific tests
    python -m eval.run_agent_eval --url http://localhost:8000
    python -m eval.run_agent_eval --out eval/report.html   # custom output path
    python -m eval.run_agent_eval --no-sleep               # skip inter-test delays (faster)
"""
import argparse
import sys
from datetime import datetime, timezone

from eval.test_cases import TEST_CASES
from eval.agent_test_cases import AGENT_TEST_CASES
from eval.agent_evaluator import run_suite, compute_metrics, CaseResult
from eval.report_generator import generate_report


# ── Terminal colours ──────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
DIM    = "\033[2m"
BLUE   = "\033[94m"
MAGENTA= "\033[95m"


def _col(score: str) -> str:
    return {"PASS": GREEN, "FAIL": RED, "ERROR": YELLOW}.get(score, RESET)


def _bool_icon(v: bool) -> str:
    return f"{GREEN}Yes{RESET}" if v else f"{RED}No{RESET}"


def _pct(v) -> str:
    return f"{round(v * 100)}%" if v is not None else "N/A"

def _fmt_num(v, decimals=1) -> str:
    return f"{v:.{decimals}f}" if v is not None else "N/A"


# ── Terminal report ───────────────────────────────────────────────────────────

def print_report(results: list, metrics: dict) -> None:
    total  = metrics["total_count"]
    passed = metrics["pass_count"]
    failed = metrics["fail_count"]
    errors = metrics["error_count"]
    pct    = round(passed / total * 100) if total else 0

    print()
    print(f"{BOLD}{'='*80}{RESET}")
    print(f"{BOLD}  PORTFOLIO AGENT — FULL EVALUATION REPORT{RESET}")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{BOLD}{'='*80}{RESET}")

    # ── Metric scorecard ──────────────────────────────────────────────────────
    print(f"\n{BOLD}  AGGREGATE METRICS{RESET}")
    print(f"  {DIM}{'─'*76}{RESET}")

    def mrow(label, value, colour=RESET, note=""):
        note_str = f"  {DIM}{note}{RESET}" if note else ""
        print(f"  {label:<30} {colour}{value}{RESET}{note_str}")

    mrow("Overall Score",          f"{pct}%  ({passed}/{total})",
         GREEN if pct >= 70 else RED)
    print()
    mrow("Guardrail Score",        _pct(metrics.get("guardrail_score")),      CYAN,    "% of guardrail checks passed")
    mrow("Tool Call Score",        _pct(metrics.get("tool_call_score")),       BLUE,    "% of tool-call checks passed")
    mrow("Retry Loop Score",       _pct(metrics.get("retry_score")),           MAGENTA, "% of retry checks passed")
    mrow("Price Grounding Score",  _pct(metrics.get("price_grounding_score")), GREEN,   "proposed price within 2% of live price")
    mrow("Faithfulness Score",     _pct(metrics.get("faithfulness_score")),    GREEN,   "trades only for fetched tickers")
    print()
    mrow("Avg Context Precision",  _pct(metrics.get("avg_context_precision")), CYAN,    "tickers used / tickers fetched")
    mrow("Avg Context Recall",     _pct(metrics.get("avg_context_recall")),    MAGENTA, "holdings mentioned / total holdings")
    mrow("Avg Answer Relevance",   _pct(metrics.get("avg_answer_relevance")),  BLUE,    "financial signal word density")
    print()
    mrow("Avg Tool Calls / Test",  _fmt_num(metrics.get("avg_tool_calls")), DIM)
    mrow("Avg Retries / Test",     _fmt_num(metrics.get("avg_retry_count")), DIM)
    mrow("Avg Response Time",      (_fmt_num(metrics.get("avg_response_time_s")) + "s"
                                    if metrics.get("avg_response_time_s") is not None else "N/A"), DIM)

    # ── Per-test rows ─────────────────────────────────────────────────────────
    print(f"\n{BOLD}  PER-TEST RESULTS{RESET}")
    print(f"  {DIM}{'─'*76}{RESET}")
    header = f"{'ID':<8} {'Score':<8} {'Prec':>6} {'Rec':>6} {'Rel':>6} {'Tools':>5} {'Ret':>4}  Description"
    print(f"  {BOLD}{header}{RESET}")
    print(f"  {DIM}{'-'*76}{RESET}")

    for r in results:
        sc  = f"{_col(r.score)}{r.score:<8}{RESET}"
        cp  = f"{CYAN}{_pct(r.context_precision):>6}{RESET}"
        cr  = f"{MAGENTA}{_pct(r.context_recall):>6}{RESET}"
        ar  = f"{BLUE}{_pct(r.answer_relevance):>6}{RESET}"
        tc  = f"{r.tool_calls_count:>5}"
        ret = f"{r.retry_count:>4}"
        desc = r.description[:50] + ("…" if len(r.description) > 50 else "")
        print(f"  {r.test_id:<8} {sc} {cp} {cr} {ar} {tc} {ret}  {desc}")

        for cr_item in r.check_results:
            icon = f"{GREEN}✓{RESET}" if cr_item.passed else f"{RED}✗{RESET}"
            cat  = f"{DIM}[{cr_item.check_category}]{RESET}"
            print(f"           {icon} {cat} {DIM}{cr_item.check_description}{RESET}")
            if not cr_item.passed:
                print(f"             {YELLOW}↳ {cr_item.reason}{RESET}")

        if r.error:
            print(f"           {RED}ERROR: {r.error}{RESET}")
        print()

    # ── Summary bar ───────────────────────────────────────────────────────────
    print(f"  {BOLD}{'-'*76}{RESET}")
    overall_lbl = f"{GREEN}GOOD{RESET}" if pct >= 70 else f"{RED}NEEDS WORK{RESET}"
    print(f"  Total: {total}  |  "
          f"{GREEN}PASS: {passed}{RESET}  |  "
          f"{RED}FAIL: {failed}{RESET}  |  "
          f"{YELLOW}ERROR: {errors}{RESET}  |  "
          f"Score: {BOLD}{pct}%{RESET}  →  {overall_lbl}")
    print(f"  {BOLD}{'='*80}{RESET}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Portfolio Agent — Full Evaluation Suite")
    parser.add_argument("--url",      default="http://localhost:8000")
    parser.add_argument("--user",     default="eval_user")
    parser.add_argument("--suite",    choices=["all", "original", "agent"], default="all",
                        help="Which test suite to run (default: all)")
    parser.add_argument("--ids",      nargs="*", help="Run only these test IDs")
    parser.add_argument("--out",      default="eval/report.html", help="HTML report output path")
    parser.add_argument("--no-sleep", action="store_true", help="Skip inter-test delays")
    args = parser.parse_args()

    # Select cases
    if args.suite == "original":
        cases = TEST_CASES
        suite_label = "Portfolio Agent — Original Test Suite (T001–T020)"
    elif args.suite == "agent":
        cases = AGENT_TEST_CASES
        suite_label = "Portfolio Agent — Agent Capability Suite (A001–A021 + B001–B009)"
    else:
        cases = TEST_CASES + AGENT_TEST_CASES
        suite_label = "Portfolio Agent — Full Evaluation Suite (T001–T020 + A001–A021 + B001–B009)"

    print(f"\n{BOLD}Running: {suite_label}{RESET}")
    print(f"  URL  : {args.url}")
    print(f"  User : {args.user}")
    print(f"  Tests: {args.ids or 'all'}\n")

    results = run_suite(
        cases,
        base_url=args.url,
        user_id=args.user,
        test_ids=args.ids,
        sleep_between=0 if args.no_sleep else 5,
    )

    metrics = compute_metrics(results)
    print_report(results, metrics)

    # Generate HTML report
    path = generate_report(results, output_path=args.out, suite_label=suite_label)
    print(f"  {GREEN}HTML report saved → {path}{RESET}\n")

    any_bad = any(r.score != "PASS" for r in results)
    sys.exit(1 if any_bad else 0)


if __name__ == "__main__":
    main()
