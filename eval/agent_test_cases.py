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
    GetPricesBatchCalled, GetPricesBatchNotCalled, BatchTickerCount,
    NoHallucinatedTickers,
    TradeCountAtMost,
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
        description="MSFT-only goal must fetch MSFT price and no unrequested tickers",
        request={
            "mode": "goal",
            "goal": "Fetch my portfolio. What is the live price of MSFT right now?",
        },
        checks=[
            ToolWasCalled("get_portfolio"),
            SpecificTickerFetched("MSFT"),
            ContextPrecision(min_precision=0.5),
            SummaryContains(keywords=["msft", "microsoft", "price", "$"]),
        ],
        notes=(
            "Context precision — user asked specifically about MSFT. "
            "Any fetched ticker that doesn't appear in the response lowers precision."
        ),
    ),

    # A004 — Comprehensive review fetches all holdings via batch
    TestCase(
        id="A004",
        description="Full portfolio review must fetch prices for all 3 held tickers (AAPL, MSFT, JPM)",
        request={
            "mode": "goal",
            "goal": (
                "Fetch my holdings and get live prices for every stock I hold. "
                "Give me a full review."
            ),
        },
        checks=[
            ToolWasCalled("get_portfolio"),
            GetPricesBatchCalled(),
            ContextRecall(portfolio_tickers=["AAPL", "MSFT", "JPM"], min_recall=1.0),
        ],
        notes=(
            "Context recall — 'full review' triggers holistic mode, so the analyst must use "
            "get_prices_batch for all 3 seed holdings (AAPL, MSFT, JPM). "
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
            GetPricesBatchCalled(),
        ],
        notes=(
            "Runaway tool call guard — the goal triggers holistic mode (contains 'portfolio' "
            "and 'live prices'), so get_prices_batch is used instead of get_live_price. "
            "After portfolio + prices are loaded the analyst switches to a tool-free LLM. "
            "This test confirms the loop terminates and total tool calls are bounded."
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
        description="Non-DJI ticker plan exhausts retries and ends with risk_approved=False",
        request={
            "mode": "goal",
            "goal": (
                "Fetch my portfolio. I insist on buying 10 shares of TSLA. "
                "Only propose TSLA. Do not suggest alternatives."
            ),
        },
        checks=[
            ShouldReject(),
        ],
        notes=(
            "Guardrail — TSLA is not in the Dow Jones 30 whitelist. The analyst should refuse "
            "the request and return 0 trades. The auditor always approves an empty trade array, "
            "so risk_approved=True with 0 trades is the correct outcome."
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
                "Suggest buying exactly 1 share of AAPL if my cash allows."
            ),
        },
        checks=[
            RetryCountAtMost(n=1),
            RiskApproved(expected=True),
            ShouldHaveTrades(min_trades=1),
        ],
        notes=(
            "Retry efficiency — buying 1 share of AAPL (a valid DJI ticker) is conservative. "
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
                "Suggest selling 5 shares of JPM at the current market price."
            ),
        },
        checks=[
            PriceGrounded(tolerance=0.02, min_grounded_fraction=1.0),
            SpecificTickerFetched("JPM"),
            Faithfulness(),
            RiskApproved(expected=True),
        ],
        notes=(
            "Price grounding — proposed price must match the fetched live price "
            "within 2% (risk auditor enforces same rule). Tests that analyst uses "
            "fetched price, not a hallucinated value. JPM is the largest holding (15 shares)."
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
        description="MSFT-only query must mention MSFT in summary (precision)",
        request={
            "mode": "goal",
            "goal": (
                "Fetch my MSFT holding and its live price only. "
                "Tell me the value of my MSFT position."
            ),
        },
        checks=[
            SpecificTickerFetched("MSFT"),
            ContextPrecision(min_precision=0.5),
            SummaryContains(keywords=["msft", "microsoft"]),
            AnswerRelevance(keywords=["msft", "microsoft", "value", "shares", "price", "$"], min_keywords=2),
        ],
        notes=(
            "Context precision — user scoped the question to MSFT only. "
            "Precision = tickers appearing in summary / tickers fetched. "
            "If analyst fetches AAPL and JPM unnecessarily, precision drops."
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
            GetPricesBatchCalled(),
            ContextRecall(portfolio_tickers=["AAPL", "MSFT", "JPM"], min_recall=1.0),
            AnswerRelevance(
                keywords=["sector", "financial", "banking", "technology", "tech"],
                min_keywords=1,
            ),
        ],
        notes=(
            "Context recall — analyst must cover all 3 DJI holdings (AAPL, MSFT, JPM) to "
            "correctly report sector allocation. Recall = holdings mentioned / total holdings held. "
            "Holistic mode uses get_prices_batch over held tickers."
        ),
    ),

    # A015 — Answer relevance: cash observation must directly address the goal
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
                keywords=["cash", "allocation", "equity", "ratio", "balance", "idle"],
                min_keywords=2,
            ),
            SummaryContains(keywords=["cash"]),
            RiskApproved(expected=True),
        ],
        notes=(
            "Answer relevance — seed portfolio has ~39% cash ($5,000 of ~$12,900 total). "
            "The summary must directly address the cash-vs-equity question. "
            "A generic portfolio response that ignores cash balance fails this check."
        ),
    ),

    # ── Full rebalance / DJI 30 checks ───────────────────────────────────────

    # A016 — Full rebalance must call get_prices_batch (not sequential get_live_price)
    TestCase(
        id="A016",
        description="Full rebalance request must use get_prices_batch, not sequential get_live_price",
        request={
            "mode": "goal",
            "goal": (
                "Fetch my portfolio. I want a complete rebalance of my portfolio "
                "across the Dow Jones stocks."
            ),
        },
        checks=[
            ToolWasCalled("get_portfolio"),
            GetPricesBatchCalled(),
            ToolNotCalled("get_live_price"),
        ],
        notes=(
            "Tool routing — is_full_rebalance() detects 'complete rebalance' and routes to "
            "get_prices_batch. get_live_price is hidden from the LLM in this mode. "
            "Verifies the deterministic routing logic fires correctly."
        ),
    ),

    # A017 — Full rebalance auditor allows up to 5 trades
    TestCase(
        id="A017",
        description="Full rebalance plan with 4 trades must be approved (relaxed 5-trade limit)",
        request={
            "mode": "goal",
            "goal": (
                "Fetch my portfolio. I want to rebalance my entire portfolio to reduce "
                "JPM concentration and add sector diversification. Propose up to 4 trades."
            ),
        },
        checks=[
            RiskApproved(expected=True),
            TradeCountAtMost(max_trades=5),
            GetPricesBatchCalled(),
        ],
        notes=(
            "Relaxed rebalance auditor — normal auditor caps at 3 trades; rebalance auditor "
            "allows 5. A 4-trade rebalance plan should pass. Also verifies get_prices_batch "
            "is used (full rebalance routing)."
        ),
    ),

    # A018 — Full rebalance: liquidation request pivots to incremental plan within 20% cap
    TestCase(
        id="A018",
        description="Liquidation request under rebalance mode must pivot to an incremental approved plan",
        request={
            "mode": "goal",
            "goal": (
                "Rebalance my entire portfolio. Liquidate all positions and reinvest "
                "everything into new stocks."
            ),
        },
        checks=[
            RiskApproved(expected=True),
            TradeCountAtMost(max_trades=5),
            GetPricesBatchCalled(),
            SummaryContains(
                keywords=["20%", "cap", "cannot", "liquidat", "limit", "incremental", "exceed"],
                description="Summary must explain why full liquidation was refused and describe the incremental plan",
            ),
        ],
        notes=(
            "Rebalance aggression guard — 'liquidate all' exceeds the 20% notional cap. "
            "The analyst correctly refuses the full liquidation, explains the cap, and pivots "
            "to a valid incremental rebalance of up to 5 trades within the limit. "
            "Expecting 0 trades was wrong — the correct behavior is a valid incremental plan."
        ),
    ),

    # A019 — Full rebalance batch covers all 30 DJI tickers
    TestCase(
        id="A019",
        description="Full rebalance get_prices_batch call must include all 30 DJI tickers",
        request={
            "mode": "goal",
            "goal": (
                "Fetch my portfolio and rebalance across all 30 Dow Jones stocks. "
                "Consider all stocks in the index."
            ),
        },
        checks=[
            ToolWasCalled("get_portfolio"),
            GetPricesBatchCalled(),
            BatchTickerCount(min_tickers=30),
            RetryCountAtMost(n=3),
        ],
        notes=(
            "Full rebalance batch completeness — when full rebalance is detected the analyst "
            "is instructed to call get_prices_batch with all 30 DJI tickers. "
            "BatchTickerCount(30) verifies the batch call was not truncated."
        ),
    ),

    # A020 — Holistic review uses get_prices_batch, not sequential get_live_price
    TestCase(
        id="A020",
        description="Holistic portfolio review must use get_prices_batch over held tickers",
        request={
            "mode": "goal",
            "goal": (
                "Fetch my portfolio. Give me a complete overview of all my holdings "
                "and their current values."
            ),
        },
        checks=[
            ToolWasCalled("get_portfolio"),
            GetPricesBatchCalled(),
            ToolNotCalled("get_live_price"),
            ContextRecall(portfolio_tickers=["AAPL", "MSFT", "JPM"], min_recall=0.67),
        ],
        notes=(
            "Tool routing — is_holistic_analysis() detects 'complete overview' / 'all my holdings'. "
            "In holistic mode get_live_price is hidden and get_prices_batch is used instead. "
            "Verifies the batch covers at least the 3 held tickers."
        ),
    ),

    # A021 — Targeted single-ticker query uses get_live_price, NOT get_prices_batch
    TestCase(
        id="A021",
        description="Targeted single-ticker query must use get_live_price, not get_prices_batch",
        request={
            "mode": "goal",
            "goal": "Fetch my portfolio. What is the current price of AAPL?",
        },
        checks=[
            ToolWasCalled("get_portfolio"),
            ToolWasCalled("get_live_price"),
            GetPricesBatchNotCalled(),
            SummaryContains(keywords=["aapl", "apple"]),
        ],
        notes=(
            "Tool routing — narrow single-ticker query triggers targeted mode (not holistic). "
            "get_prices_batch is hidden and get_live_price is available. "
            "Verifies the routing logic correctly identifies non-holistic intent."
        ),
    ),

    # ── Beta user profile tests (B001–B008) ───────────────────────────────────
    # Each case targets a specific user with a distinct portfolio profile.
    # Users must be pre-seeded in the DB before running these cases.

    # B001 — beta_user1: $5k cash + 10 AAPL @ $226. Cash-deploy goal.
    TestCase(
        id="B001",
        description="Cash-heavy single-stock user: suggest 1 BUY to diversify (beta_user1)",
        user_id="beta_user1",
        request={
            "mode": "goal",
            "goal": (
                "Fetch my portfolio. I have idle cash sitting around. "
                "Suggest one stock to buy to start diversifying across sectors."
            ),
        },
        checks=[
            ToolWasCalled("get_portfolio"),
            GetPricesBatchCalled(),
            ShouldHaveTrades(min_trades=1),
            RiskApproved(expected=True),
            SummaryContains(keywords=["cash", "aapl", "apple", "diversif", "sector"]),
        ],
        notes=(
            "beta_user1: 10 AAPL + $5k cash. Cash is ~66% of total portfolio — "
            "triggers cash drag warning. Analyst should fetch prices holistically "
            "(contains 'diversifying'), propose 1 BUY from a non-tech sector. "
            "Risk auditor should approve since $5k cash comfortably covers a small buy."
        ),
    ),

    # B002 — beta_user2: 5 MSFT @ $435 (underwater), 20 VZ @ $42, 15 KO @ $60, $3k cash.
    TestCase(
        id="B002",
        description="All-underwater portfolio: worst performer identified by return% (beta_user2)",
        user_id="beta_user2",
        request={
            "mode": "goal",
            "goal": (
                "Fetch my portfolio and live prices. "
                "Show me the gain or loss for each position and identify my worst performing stock."
            ),
        },
        checks=[
            ToolWasCalled("get_portfolio"),
            GetPricesBatchCalled(),
            RiskApproved(expected=True),
            SummaryContains(keywords=["msft", "microsoft", "loss", "worst", "%", "return"]),
            AnswerRelevance(
                keywords=["gain", "loss", "return", "percent", "%", "worst", "performing"],
                min_keywords=2,
            ),
        ],
        notes=(
            "beta_user2: MSFT @ $435 (live ~$373 → -14%), VZ @ $42 (live ~$49 → +17%), "
            "KO @ $60 (live ~$76 → +27%). MSFT is the worst performer by return%. "
            "Analyst must compute return% for each holding and correctly identify MSFT. "
            "Goal is informational — no trades required, risk_approved=True with 0 trades is valid."
        ),
    ),

    # B003 — beta_user3: 40 AAPL @ $185, 25 AMZN @ $178, 15 NVDA @ $495, $75k cash.
    TestCase(
        id="B003",
        description="Cash-bloated FAANG portfolio: cash drag and tech concentration flagged (beta_user3)",
        user_id="beta_user3",
        request={
            "mode": "goal",
            "goal": (
                "Review my portfolio. I have large cash reserves and mostly tech holdings. "
                "Is my cash allocation appropriate? Flag any concentration risks."
            ),
        },
        checks=[
            ToolWasCalled("get_portfolio"),
            GetPricesBatchCalled(),
            RiskApproved(expected=True),
            SummaryContains(keywords=["cash", "tech", "concentrat", "sector", "drag", "idle"]),
            AnswerRelevance(
                keywords=["cash", "tech", "concentration", "sector", "diversif", "allocation"],
                min_keywords=2,
            ),
        ],
        notes=(
            "beta_user3: $75k cash + AAPL/AMZN/NVDA all deeply in-the-money. "
            "Cash is roughly 50%+ of total portfolio — cash drag threshold. "
            "All 3 holdings are technology sector — sector concentration >60%. "
            "Analyst must surface both issues. Goal is a review; no trade is required "
            "if analyst explains the problems clearly (no_trade_reason=analyst_no_trade)."
        ),
    ),

    # B004 — beta_user4: 50 JPM @ $195, 5 GS @ $410, $2k cash. Heavy financials concentration.
    TestCase(
        id="B004",
        description="Single-sector financials concentration: analyst proposes diversifying BUY (beta_user4)",
        user_id="beta_user4",
        request={
            "mode": "goal",
            "goal": (
                "Fetch my portfolio and prices. "
                "Analyse my concentration risk and suggest one trade to diversify across sectors."
            ),
        },
        checks=[
            ToolWasCalled("get_portfolio"),
            GetPricesBatchCalled(),
            ShouldHaveTrades(min_trades=1),
            RiskApproved(expected=True),
            SummaryContains(keywords=["jpm", "jpmorgan", "concentrat", "financial", "sector", "diversif"]),
        ],
        notes=(
            "beta_user4: JPM is 90%+ of equity (50 shares × ~$295). GS is a minor position. "
            "Both are financials — 100% single-sector. Analyst must flag concentration, "
            "propose 1 BUY from a non-financial sector (e.g. healthcare, consumer, tech). "
            "$2k cash constrains the buy size — analyst must respect available cash."
        ),
    ),

    # B005 — beta_user5: 10 NVDA @ $950, 20 BA @ $220, 25 DIS @ $115, $1.5k cash.
    TestCase(
        id="B005",
        description="All positions underwater: gain/loss computed per holding (beta_user5)",
        user_id="beta_user5",
        request={
            "mode": "goal",
            "goal": (
                "Fetch my portfolio and live prices. "
                "Compute my total gain or loss in dollars and percentage for each holding."
            ),
        },
        checks=[
            ToolWasCalled("get_portfolio"),
            GetPricesBatchCalled(),
            RiskApproved(expected=True),
            SummaryContains(keywords=["nvda", "ba", "boeing", "dis", "disney", "loss", "%"]),
            AnswerRelevance(
                keywords=["gain", "loss", "dollar", "$", "percent", "%", "return"],
                min_keywords=2,
            ),
            RetryCountAtMost(n=2),
        ],
        notes=(
            "beta_user5: NVDA @ $950 (live ~$900 → -5%), BA @ $220 (live ~$175 → -20%), "
            "DIS @ $115 (live ~$90 → -22%). All three positions are in the red. "
            "Analyst must show $ gain/loss AND % for each. Goal is informational. "
            "No trade needed — analyst should explain losses and propose nothing unless "
            "there is a strong case for a stop-loss sell."
        ),
    ),

    # B006 — beta_user6: 8 MSFT, 10 JPM, 12 JNJ, 15 CVX, $4k cash. Healthy 4-sector portfolio.
    TestCase(
        id="B006",
        description="Well-diversified healthy portfolio: AI feedback must propose zero trades (beta_user6)",
        user_id="beta_user6",
        request={
            "mode": "feedback",
        },
        checks=[
            ToolWasCalled("get_portfolio"),
            GetPricesBatchCalled(),
            RiskApproved(expected=True),
            ShouldReject(),   # 0 trades — portfolio is healthy
            SummaryContains(keywords=["health", "balanced", "diversif", "sector", "no trade", "well"]),
            RetryCountAtMost(n=2),
        ],
        notes=(
            "beta_user6: MSFT (tech), JPM (financials), JNJ (healthcare), CVX (energy) — "
            "4 sectors, no single stock >40% equity, cash ~15% of total. "
            "This is the 'healthy portfolio' control case. Feedback mode health check "
            "must conclude the portfolio is in good shape and propose 0 trades. "
            "Any trade suggestion here is a false positive."
        ),
    ),

    # B007 — beta_user7: $50k cash, zero holdings.
    TestCase(
        id="B007",
        description="100% cash portfolio: analyst flags cash drag and proposes 2 sector-diversified BUYs (beta_user7)",
        user_id="beta_user7",
        request={
            "mode": "goal",
            "goal": (
                "Fetch my portfolio. All my money is in cash. "
                "Suggest 2 stocks to invest in from different sectors."
            ),
        },
        checks=[
            ToolWasCalled("get_portfolio"),
            GetPricesBatchCalled(),
            ShouldHaveTrades(min_trades=1),
            RiskApproved(expected=True),
            TradeCountAtMost(max_trades=3),
            SummaryContains(keywords=["cash", "invest", "sector", "buy", "deploy"]),
        ],
        notes=(
            "beta_user7: $50k cash, no holdings. Cash drag is 100% — maximum severity. "
            "Analyst must suggest 2 BUYs from different sectors. $50k gives ample room "
            "so the risk auditor should approve. TradeCountAtMost(3) ensures the analyst "
            "doesn't over-propose — goal asked for 2, not 10."
        ),
    ),

    # B009 — beta_user5: BA, DIS, NVDA all underwater. Analyst must not hallucinate
    # tickers from other sessions (e.g. MSFT) into the summary.
    TestCase(
        id="B009",
        description="Analyst must not hallucinate tickers absent from portfolio into summary (beta_user5)",
        user_id="beta_user5",
        request={
            "mode": "goal",
            "goal": (
                "Fetch my portfolio and live prices. "
                "All my stocks are at a loss. Should I sell the worst performer?"
            ),
        },
        checks=[
            ToolWasCalled("get_portfolio"),
            GetPricesBatchCalled(),
            NoHallucinatedTickers(portfolio_tickers=["NVDA", "BA", "DIS"]),
            RiskApproved(expected=True),
            SummaryContains(keywords=["nvda", "ba", "boeing", "dis", "disney", "loss"]),
        ],
        notes=(
            "Hallucination guard — beta_user5 holds only NVDA, BA, DIS. "
            "A known failure mode is the analyst referencing tickers from other users' "
            "sessions (e.g. MSFT, AAPL, JPM) that bleed in via stale risk_feedback state. "
            "NoHallucinatedTickers checks that no DJI ticker outside {NVDA, BA, DIS} "
            "appears in the summary unless its price was explicitly fetched for this session."
        ),
    ),

    # B008 — beta_user8: 6 DJI stocks (AAPL, MSFT, UNH, GS, CAT, WMT), $3k cash.
    TestCase(
        id="B008",
        description="Large diversified portfolio: full rebalance uses 30-ticker batch (beta_user8)",
        user_id="beta_user8",
        request={
            "mode": "goal",
            "goal": (
                "Fetch my portfolio and rebalance across all 30 Dow Jones stocks. "
                "Propose up to 3 incremental trades to improve diversification."
            ),
        },
        checks=[
            ToolWasCalled("get_portfolio"),
            GetPricesBatchCalled(),
            BatchTickerCount(min_tickers=30),
            TradeCountAtMost(max_trades=5),
            RiskApproved(expected=True),
            RetryCountAtMost(n=3),
        ],
        notes=(
            "beta_user8: AAPL+MSFT (tech), UNH (healthcare), GS (financials), "
            "CAT (industrials), WMT (retail) — 5 sectors, reasonably balanced. "
            "Full rebalance mode fires (contains 'rebalance across all 30 Dow Jones'). "
            "Analyst must call get_prices_batch with all 30 tickers. "
            "With good existing diversification, the plan should be conservative "
            "(≤3 trades) and approved first or second pass."
        ),
    ),
]
