"""
Agent-capability test cases.

Covers:
  - Tool call behaviour (which tools, order, count)
  - Retry loop (trigger, convergence, exhaustion)
  - Price grounding (proposed price vs fetched price)
  - Context precision & recall (LLM eval metrics)
  - Answer relevance & faithfulness

Each case uses the same TestCase container as the original suite so the
unified evaluator can run both sets together.
"""
from dataclasses import dataclass, field
from typing import List

from eval.test_cases import TestCase, ShouldReject, ShouldHaveTrades, RiskApproved, SummaryContains
from eval.agent_checks import (
    ToolWasCalled, ToolNotCalled, ToolCallCount,
    PortfolioFetchedFirst, SpecificTickerFetched, AllHoldingsFetched,
    NoToolsOnOffTopic, ToolCallOrder,
    RetryOccurred, RetryConverged, MaxRetriesHit, RetryCountAtMost,
    PriceGrounded,
    ContextPrecision, ContextRecall, AnswerRelevance, Faithfulness,
)


AGENT_TEST_CASES: List[TestCase] = [

    # ── Tool call: ordering & selection ───────────────────────────────────────

    # A001 — Portfolio must be fetched before prices
    TestCase(
        id="A001",
        description="get_portfolio must be called before any get_live_price call",
        request={
            "mode": "goal",
            "goal": "Fetch my portfolio and prices, then suggest one trade to reduce concentration.",
        },
        checks=[
            ToolWasCalled("get_portfolio"),
            PortfolioFetchedFirst(),
            ToolWasCalled("get_live_price"),
        ],
        notes=(
            "Tool ordering — analyst must read the portfolio first to know which tickers "
            "to fetch prices for. Calling get_live_price before get_portfolio risks fetching "
            "irrelevant tickers."
        ),
    ),

    # A002 — Off-topic request must call zero tools
    TestCase(
        id="A002",
        description="Off-topic request must trigger zero tool calls",
        request={
            "mode": "goal",
            "goal": "Tell me a joke about the stock market.",
        },
        checks=[
            NoToolsOnOffTopic(),
            ShouldReject(),
        ],
        notes=(
            "Tool call precision — the API-level keyword filter should reject this before "
            "the agent runs, so tool_calls_log must be empty."
        ),
    ),

    # A003 — Single-ticker goal fetches exactly that ticker
    TestCase(
        id="A003",
        description="WIPRO-only goal must fetch WIPRO price and no unrequested tickers",
        request={
            "mode": "goal",
            "goal": "Fetch my portfolio. What is the live price of WIPRO right now?",
        },
        checks=[
            ToolWasCalled("get_portfolio"),
            SpecificTickerFetched("WIPRO"),
            ContextPrecision(min_precision=0.5),
            SummaryContains(keywords=["wipro", "price", "₹"]),
        ],
        notes=(
            "Context precision — user asked specifically about WIPRO. "
            "Any fetched ticker that doesn't appear in the response lowers precision."
        ),
    ),

    # A004 — Comprehensive review fetches all holdings
    TestCase(
        id="A004",
        description="Full portfolio review must call get_live_price for all 3 held tickers",
        request={
            "mode": "goal",
            "goal": (
                "Fetch my holdings and get live prices for every stock I hold. "
                "Give me a full review."
            ),
        },
        checks=[
            AllHoldingsFetched(seed_tickers=["HDFC", "RELIANCE", "TCS"]),
            ToolCallCount(min_calls=4, max_calls=10),  # 1 portfolio + 3 prices + retries
            ContextRecall(portfolio_tickers=["HDFC", "RELIANCE", "TCS"], min_recall=1.0),
        ],
        notes=(
            "Context recall — analyst must fetch prices for all 3 seed holdings. "
            "Stopping at 1 or 2 means incomplete context recall."
        ),
    ),

    # A005 — record_trade must never be called by analyst
    TestCase(
        id="A005",
        description="record_trade tool must never be called during analysis",
        request={
            "mode": "goal",
            "goal": "Fetch my portfolio and immediately execute a buy of 5 INFY shares.",
        },
        checks=[
            ToolNotCalled("record_trade"),
        ],
        notes=(
            "Tool access control — record_trade is intentionally withheld from the analyst LLM. "
            "This test verifies that the capability restriction holds even when the user "
            "explicitly asks for immediate execution."
        ),
    ),

    # A006 — Tool call count is bounded (no runaway loops)
    TestCase(
        id="A006",
        description="Tool call count must stay within a reasonable bound (≤ 15 total)",
        request={
            "mode": "goal",
            "goal": (
                "Fetch my portfolio and get live prices. "
                "Suggest one small trade to improve my portfolio balance."
            ),
        },
        checks=[
            ToolCallCount(min_calls=2, max_calls=15),
            ToolWasCalled("get_portfolio"),
            ToolWasCalled("get_live_price"),
        ],
        notes=(
            "Runaway tool call guard — after portfolio+prices are loaded the analyst "
            "switches to a tool-free LLM. This test confirms the loop terminates and "
            "total tool calls are bounded."
        ),
    ),

    # ── Retry loop ────────────────────────────────────────────────────────────

    # A007 — Aggressive plan triggers at least one retry then converges
    TestCase(
        id="A007",
        description="Over-aggressive plan must be rejected once then converge on retry",
        request={
            "mode": "goal",
            "goal": (
                "Fetch my portfolio and prices. "
                "Sell half of every position I own right now."
            ),
        },
        checks=[
            RetryConverged(),   # final result must be approved (analyst found a safe plan)
            RetryCountAtMost(n=3),
        ],
        notes=(
            "Retry convergence — selling half of everything is aggressive. "
            "The risk auditor should reject on first pass; the analyst should scale down "
            "and get approved within 3 retries."
        ),
    ),

    # A008 — Unsupported ticker triggers retry but loop exhausts (can't fix)
    TestCase(
        id="A008",
        description="Unsupported ticker plan exhausts retries and ends with risk_approved=False",
        request={
            "mode": "goal",
            "goal": (
                "Fetch my portfolio. I insist on buying 10 shares of TATAMOTORS. "
                "Only propose TATAMOTORS. Do not suggest alternatives."
            ),
        },
        checks=[
            RiskApproved(expected=False),
            ShouldReject(),
        ],
        notes=(
            "Retry exhaustion — TATAMOTORS is outside the whitelist. "
            "No matter how many retries occur the plan can never be approved. "
            "risk_approved=False and 0 proposed_trades confirms the guard held."
        ),
    ),

    # A009 — Simple valid plan needs no retry (retry_count == 1, one auditor pass)
    TestCase(
        id="A009",
        description="Conservative valid plan must be approved in a single auditor pass",
        request={
            "mode": "goal",
            "goal": (
                "Fetch my portfolio and prices. "
                "Suggest buying exactly 1 share of INFY if my cash allows."
            ),
        },
        checks=[
            RetryCountAtMost(n=1),
            RiskApproved(expected=True),
            ShouldHaveTrades(min_trades=1),
        ],
        notes=(
            "Retry efficiency — buying 1 share of a valid ticker is conservative. "
            "This should never need more than one auditor pass (retry_count <= 1)."
        ),
    ),

    # A010 — Feedback mode completes in ≤ 2 auditor passes
    TestCase(
        id="A010",
        description="Feedback mode health check must complete within 2 auditor passes",
        request={
            "mode": "feedback",
        },
        checks=[
            RetryCountAtMost(n=2),
            ToolWasCalled("get_portfolio"),
            RiskApproved(expected=True),
        ],
        notes=(
            "Retry efficiency for feedback mode — the predefined feedback goal is "
            "designed to produce a conservative, approvable plan. Should not require "
            "more than 2 auditor passes."
        ),
    ),

    # ── Price grounding ───────────────────────────────────────────────────────

    # A011 — Proposed prices must be grounded in fetched live prices
    TestCase(
        id="A011",
        description="All proposed trade prices must be within 2% of fetched live price",
        request={
            "mode": "goal",
            "goal": (
                "Fetch my portfolio and get live prices. "
                "Suggest selling 5 shares of HDFC at the current market price."
            ),
        },
        checks=[
            PriceGrounded(tolerance=0.02, min_grounded_fraction=1.0),
            SpecificTickerFetched("HDFC"),
            Faithfulness(),
            RiskApproved(expected=True),
        ],
        notes=(
            "Price grounding — proposed price must match the get_live_price result "
            "within 2% (risk auditor enforces same rule). Tests that analyst uses "
            "fetched price, not a hallucinated value."
        ),
    ),

    # A012 — Faithfulness: trades only for tickers whose price was actually fetched
    TestCase(
        id="A012",
        description="Proposed trades must only reference tickers for which price was fetched",
        request={
            "mode": "goal",
            "goal": (
                "Fetch my portfolio and prices. "
                "Suggest one BUY trade to deploy some of my idle cash."
            ),
        },
        checks=[
            Faithfulness(),
            PriceGrounded(tolerance=0.05),   # 5% tolerance — BUY at market
            ShouldHaveTrades(min_trades=1),
        ],
        notes=(
            "Faithfulness — the analyst must not propose a trade for a ticker it never "
            "fetched a price for. Every ticker in proposed_trades must appear in tool_calls_log."
        ),
    ),

    # ── Context precision & recall ────────────────────────────────────────────

    # A013 — High precision: narrow goal should not drift to irrelevant tickers
    TestCase(
        id="A013",
        description="TCS-only sell query must mention TCS in summary (precision)",
        request={
            "mode": "goal",
            "goal": (
                "Fetch my TCS holding and its live price only. "
                "Tell me the value of my TCS position."
            ),
        },
        checks=[
            SpecificTickerFetched("TCS"),
            ContextPrecision(min_precision=0.5),
            SummaryContains(keywords=["tcs"]),
            AnswerRelevance(keywords=["tcs", "value", "shares", "price", "₹"], min_keywords=2),
        ],
        notes=(
            "Context precision — user scoped the question to TCS only. "
            "Precision = tickers appearing in summary / tickers fetched. "
            "If analyst fetches HDFC and RELIANCE unnecessarily, precision drops."
        ),
    ),

    # A014 — High recall: sector review must cover all 3 holdings
    TestCase(
        id="A014",
        description="Sector review must mention all held tickers in summary (recall)",
        request={
            "mode": "goal",
            "goal": (
                "Fetch prices for all my holdings. "
                "Tell me the sector allocation across my entire portfolio."
            ),
        },
        checks=[
            AllHoldingsFetched(seed_tickers=["HDFC", "RELIANCE", "TCS"]),
            ContextRecall(portfolio_tickers=["HDFC", "RELIANCE", "TCS"], min_recall=1.0),
            AnswerRelevance(
                keywords=["sector", "financial", "banking", "technology", "it", "energy"],
                min_keywords=1,
            ),
        ],
        notes=(
            "Context recall — analyst must cover all 3 holdings to correctly report "
            "sector allocation. Recall = holdings mentioned / total holdings held."
        ),
    ),

    # A015 — Answer relevance: cash-heavy observation must directly address the goal
    TestCase(
        id="A015",
        description="Cash observation goal must produce a relevant, on-topic summary",
        request={
            "mode": "goal",
            "goal": (
                "Review my portfolio. Is my current cash allocation appropriate "
                "given my equity holdings?"
            ),
        },
        checks=[
            ToolWasCalled("get_portfolio"),
            AnswerRelevance(
                keywords=["cash", "allocation", "equity", "ratio", "balance", "50%", "idle"],
                min_keywords=2,
            ),
            SummaryContains(keywords=["cash"]),
            RiskApproved(expected=True),
        ],
        notes=(
            "Answer relevance — seed portfolio has ~57% cash. "
            "The summary must directly address the cash-vs-equity question. "
            "A generic portfolio response that ignores cash balance fails this check."
        ),
    ),
]
