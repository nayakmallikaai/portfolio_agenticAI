"""
CLI entrypoint for the evaluation suite.

Usage:
    python -m eval.run_eval                         # run all tests
    python -m eval.run_eval --ids T001 T002         # run specific tests
    python -m eval.run_eval --url http://localhost:8000 --user eval_user
    python -m eval.run_eval --out results.json      # also save JSON report
"""

import argparse
import json
import sys
from datetime import datetime, timezone

from eval.test_cases import TEST_CASES
from eval.evaluator import run_suite, CaseResult


# ── Formatting helpers ────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
DIM    = "\033[2m"


def _score_color(score: str) -> str:
    return {"PASS": GREEN, "FAIL": RED, "ERROR": YELLOW}.get(score, RESET)


def _bool_icon(v: bool) -> str:
    return f"{GREEN}Yes{RESET}" if v else f"{RED}No{RESET}"


def print_report(results: list[CaseResult]) -> None:
    total   = len(results)
    passed  = sum(1 for r in results if r.score == "PASS")
    failed  = sum(1 for r in results if r.score == "FAIL")
    errors  = sum(1 for r in results if r.score == "ERROR")
    pct     = round(passed / total * 100) if total else 0

    print()
    print(f"{BOLD}{'='*72}{RESET}")
    print(f"{BOLD}  PORTFOLIO AGENT — EVALUATION REPORT{RESET}")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{BOLD}{'='*72}{RESET}")
    print()

    # ── Per-test rows ─────────────────────────────────────────────────────────
    header = f"{'ID':<8} {'Score':<7} {'Faithful':<10} {'Relevant':<10} {'Time':>6}  Description"
    print(f"{BOLD}{header}{RESET}")
    print(f"{DIM}{'-'*72}{RESET}")

    for r in results:
        sc = f"{_score_color(r.score)}{r.score:<7}{RESET}"
        fi = _bool_icon(r.faithful)
        re = _bool_icon(r.relevant)
        t  = f"{r.elapsed_s:>5.1f}s"
        print(f"{r.test_id:<8} {sc} {fi:<18} {re:<18} {t}  {r.description}")

        # Check details (indented)
        for cr in r.check_results:
            icon = f"{GREEN}✓{RESET}" if cr.passed else f"{RED}✗{RESET}"
            print(f"         {icon} {DIM}{cr.check_description}{RESET}")
            if not cr.passed:
                print(f"           {YELLOW}↳ {cr.reason}{RESET}")

        if r.error:
            print(f"         {RED}ERROR: {r.error}{RESET}")

        if r.notes:
            print(f"         {DIM}note: {r.notes}{RESET}")

        print()

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"{BOLD}{'-'*72}{RESET}")
    overall = f"{GREEN}GOOD{RESET}" if pct >= 70 else f"{RED}NEEDS WORK{RESET}"
    print(f"  Total: {total}  |  "
          f"{GREEN}PASS: {passed}{RESET}  |  "
          f"{RED}FAIL: {failed}{RESET}  |  "
          f"{YELLOW}ERROR: {errors}{RESET}  |  "
          f"Score: {BOLD}{pct}%{RESET}  →  {overall}")
    print(f"{BOLD}{'='*72}{RESET}")
    print()


def to_json_report(results: list[CaseResult]) -> dict:
    total  = len(results)
    passed = sum(1 for r in results if r.score == "PASS")
    return {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total": total,
            "passed": passed,
            "failed": sum(1 for r in results if r.score == "FAIL"),
            "errors": sum(1 for r in results if r.score == "ERROR"),
            "score_pct": round(passed / total * 100) if total else 0,
        },
        "cases": [
            {
                "id": r.test_id,
                "description": r.description,
                "score": r.score,
                "faithful": r.faithful,
                "relevant": r.relevant,
                "elapsed_s": r.elapsed_s,
                "notes": r.notes,
                "error": r.error,
                "checks": [
                    {"description": c.check_description, "passed": c.passed, "reason": c.reason}
                    for c in r.check_results
                ],
            }
            for r in results
        ],
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Portfolio Agent Evaluation Suite")
    parser.add_argument("--url",  default="http://localhost:8000", help="API base URL")
    parser.add_argument("--user", default="eval_user",             help="User ID for test calls")
    parser.add_argument("--ids",  nargs="*",                       help="Run only these test IDs")
    parser.add_argument("--out",  default=None,                    help="Save JSON report to file")
    args = parser.parse_args()

    print(f"\n{BOLD}Running evaluation suite against {args.url}{RESET}")
    print(f"User: {args.user}  |  Tests: {args.ids or 'all'}\n")

    results = run_suite(
        TEST_CASES,
        base_url=args.url,
        user_id=args.user,
        test_ids=args.ids,
    )

    print_report(results)

    if args.out:
        report = to_json_report(results)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"JSON report saved to {args.out}\n")

    # Exit non-zero if any test failed or errored
    any_bad = any(r.score != "PASS" for r in results)
    sys.exit(1 if any_bad else 0)


if __name__ == "__main__":
    main()
