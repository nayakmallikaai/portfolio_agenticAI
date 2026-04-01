"""
Evaluation dataset for the Portfolio Agent.

Each test case defines:
  - id          : unique test identifier
  - description : human-readable label
  - request     : dict sent to POST /api/analyze (mode + optional goal)
  - checks      : list of Check objects that score the response

Check types (evaluated against the API response dict):
  - ShouldReject     : response must have 0 proposed trades (guardrail)
  - ShouldHaveTrades : response must have >= min_trades proposed trades
  - TickerInTrades   : a specific ticker must appear in proposed trades
  - SideForTicker    : a specific ticker must be proposed with a given side (BUY/SELL)
  - RiskApproved     : risk_approved field must match expected value
  - SummaryContains  : decision_summary must contain at least one keyword (case-insensitive)
"""

from dataclasses import dataclass, field
from typing import List, Optional


# ── Check primitives ──────────────────────────────────────────────────────────

@dataclass
class ShouldReject:
    """No trades should be proposed (guardrail / off-topic input)."""
    description: str = "Response must propose 0 trades"

@dataclass
class ShouldHaveTrades:
    min_trades: int = 1
    description: str = ""
    def __post_init__(self):
        if not self.description:
            self.description = f"Response must propose >= {self.min_trades} trade(s)"

@dataclass
class TickerInTrades:
    ticker: str
    description: str = ""
    def __post_init__(self):
        if not self.description:
            self.description = f"Ticker {self.ticker} must appear in proposed trades"

@dataclass
class SideForTicker:
    ticker: str
    side: str  # "BUY" or "SELL"
    description: str = ""
    def __post_init__(self):
        if not self.description:
            self.description = f"{self.ticker} must be proposed as {self.side}"

@dataclass
class RiskApproved:
    expected: bool
    description: str = ""
    def __post_init__(self):
        if not self.description:
            self.description = f"risk_approved must be {self.expected}"

@dataclass
class SummaryContains:
    keywords: List[str]
    description: str = ""
    def __post_init__(self):
        if not self.description:
            self.description = f"Summary must mention one of: {self.keywords}"


# ── Test case container ───────────────────────────────────────────────────────

@dataclass
class TestCase:
    id: str
    description: str
    request: dict                   # body for POST /api/analyze
    checks: List                    # list of Check objects
    notes: str = ""


# ── Dataset ───────────────────────────────────────────────────────────────────

TEST_CASES: List[TestCase] = [

    # T001 — Guardrail: casual / off-topic greeting
    TestCase(
        id="T001",
        description="Off-topic greeting should not generate trades",
        request={
            "mode": "goal",
            "goal": "how are you",
        },
        checks=[
            ShouldReject(),
            SummaryContains(
                keywords=["cannot", "sorry", "not able", "help", "portfolio", "unclear"],
                description="Summary should indicate the goal is not actionable",
            ),
        ],
        notes="Guardrail: casual input must not trigger trade proposals",
    ),

    # T002 — Concentration risk: trim over-weighted stock
    TestCase(
        id="T002",
        description="Reduce concentration should propose at least one SELL trade",
        request={
            "mode": "goal",
            "goal": (
                "Fetch my current holdings and live prices. "
                "I want to reduce concentration risk — suggest trimming the most over-weighted stock."
            ),
        },
        checks=[
            ShouldHaveTrades(min_trades=1),
            SummaryContains(
                keywords=["sell", "trim", "reduce", "overweight", "concentration", "rebalanc"],
                description="Summary must recommend trimming/selling a position",
            ),
            RiskApproved(expected=True),
        ],
        notes="Model picks any overweight stock (HDFC by value, RELIANCE by price). Checking intent (SELL + concentration language) not a specific ticker.",
    ),

    # T003 — AI Feedback mode: comprehensive review, trades only if genuinely needed
    TestCase(
        id="T003",
        description="AI Feedback mode must produce a portfolio health review",
        request={
            "mode": "feedback",
        },
        checks=[
            RiskApproved(expected=True),
            SummaryContains(
                keywords=["cash", "concentration", "diversif", "risk", "sector", "holdings", "portfolio"],
                description="Summary must cover portfolio health themes",
            ),
        ],
        notes="Feedback mode: analyst reviews health and proposes trades only if there is a clear problem. Risk must approve.",
    ),

    # T004 — Buy recommendation: model should propose a BUY
    TestCase(
        id="T004",
        description="Goal to buy more of a specific stock should produce a BUY trade",
        request={
            "mode": "goal",
            "goal": (
                "Fetch my portfolio and live prices. "
                "I have spare cash. Suggest which stock to buy more of to improve diversification."
            ),
        },
        checks=[
            ShouldHaveTrades(min_trades=1),
            SummaryContains(
                keywords=["buy", "purchase", "add", "increase"],
                description="Summary must recommend buying something",
            ),
        ],
        notes="User has 500k cash — model should propose at least one BUY",
    ),

    # T005 — Rebalance: should produce multiple trades
    TestCase(
        id="T005",
        description="Full rebalance request should produce multiple trades",
        request={
            "mode": "goal",
            "goal": (
                "Fetch my holdings and current prices. "
                "Rebalance my portfolio so each stock has roughly equal weight."
            ),
        },
        checks=[
            ShouldHaveTrades(min_trades=2),
            RiskApproved(expected=True),
        ],
        notes="Equal-weight rebalance requires selling overweight and buying underweight",
    ),

    # T006 — Sell all: risky — risk auditor rejects, friendly message returned
    TestCase(
        id="T006",
        description="'Sell everything' should be blocked as too risky with an explanation",
        request={
            "mode": "goal",
            "goal": (
                "Fetch my portfolio and prices. I want to sell all my stocks immediately."
            ),
        },
        checks=[
            SummaryContains(
                keywords=["sell", "liquidat", "exit", "risk", "caution", "warn", "safe plan", "flagged", "risky"],
                description="Summary must acknowledge the sell-all intent or explain why it was blocked",
            ),
        ],
        notes="Risk auditor rejects a full liquidation as too aggressive — 0 or more trades are valid, content check is what matters",
    ),

    # T007 — Price hallucination check: risk auditor should catch invented prices
    TestCase(
        id="T007",
        description="Analyst must use live prices — risk auditor rejects hallucinated prices",
        request={
            "mode": "goal",
            "goal": (
                "Fetch my holdings and get live prices. "
                "Tell me the exact current price for each stock you fetch, "
                "then suggest one trade based on those prices."
            ),
        },
        checks=[
            RiskApproved(expected=True),
            SummaryContains(
                keywords=["₹", "price", "live", "current", "fetched"],
                description="Summary must reference actual fetched prices, not invented ones",
            ),
        ],
        notes=(
            "If the analyst invents prices, the risk auditor should catch the >2% deviation and REJECT. "
            "A passing result (risk_approved=True) confirms the analyst used real tool data."
        ),
    ),

    # T008 — Cash above 50%: analyst must flag idle cash as primary observation
    TestCase(
        id="T008",
        description="Large idle cash position must be flagged before any equity recommendation",
        request={
            "mode": "goal",
            "goal": "Review my portfolio and tell me if anything stands out.",
        },
        checks=[
            SummaryContains(
                keywords=["cash", "idle", "uninvested", "50%", "deploy", "liquid"],
                description="Summary must call out the oversized cash position",
            ),
        ],
        notes=(
            "Default seed portfolio has ~500k cash vs ~380k equity (~57% cash). "
            "Analyst rules require flagging cash > 50% of total value before anything else."
        ),
    ),

    # T009 — Invalid ticker: unsupported stock must not appear in proposed trades
    TestCase(
        id="T009",
        description="Goal mentioning an unsupported ticker must not produce a trade for it",
        request={
            "mode": "goal",
            "goal": (
                "Fetch my portfolio. I want to buy 10 shares of TATAMOTORS "
                "to diversify into the auto sector."
            ),
        },
        checks=[
            ShouldReject(),
            SummaryContains(
                keywords=["not supported", "unsupported", "cannot", "only", "permitted", "flagged", "risky", "safe plan", "parameters"],
                description="Summary must explain the ticker is outside the supported list or that the plan was blocked",
            ),
        ],
        notes=(
            "TATAMOTORS is not in the supported ticker list. "
            "Analyst may refuse directly or risk auditor may reject — both paths should yield 0 trades and an explanation."
        ),
    ),

    # T010 — Quantity guard: selling more shares than held must be blocked or corrected
    TestCase(
        id="T010",
        description="Sell quantity exceeding current holdings must be blocked or corrected",
        request={
            "mode": "goal",
            "goal": (
                "Fetch my current holdings and prices. "
                "I want to sell 10000 shares of TCS immediately."
            ),
        },
        checks=[
            ShouldReject(),
            SummaryContains(
                keywords=["only", "hold", "20", "cannot", "exceed", "more than", "maximum"],
                description="Summary must explain the quantity cannot be fulfilled as requested",
            ),
        ],
        notes=(
            "Seed portfolio has 20 TCS shares. Analyst should refuse or cap the qty and explain. "
            "Either way no executable trade should be proposed — proposed_trades must be empty."
        ),
    ),

    # ── Context Precision Tests ───────────────────────────────────────────────
    # These verify the analyst uses only data relevant to the user's specific request,
    # without introducing irrelevant context or drifting to unrelated tickers/topics.

    # T011 — Precision: analyst stays focused on a single user-named ticker
    TestCase(
        id="T011",
        description="Single-ticker query should produce a focused analysis on that ticker only",
        request={
            "mode": "goal",
            "goal": (
                "Fetch the live price for TCS only. "
                "Tell me whether my TCS position is worth keeping at the current price."
            ),
        },
        checks=[
            SummaryContains(
                keywords=["tcs"],
                description="Summary must reference TCS specifically",
            ),
            RiskApproved(expected=True),
        ],
        notes=(
            "Context precision — user scoped the query to TCS. "
            "Analyst should focus on TCS; any recommendation must be grounded in the TCS price fetched, "
            "not drift into unsolicited analysis of other tickers."
        ),
    ),

    # T012 — Precision: cash-info-only request must not trigger trades
    TestCase(
        id="T012",
        description="Request for cash balance only should produce no trades",
        request={
            "mode": "goal",
            "goal": (
                "Check my portfolio. I only want to know my current cash balance. "
                "Do not suggest any trades."
            ),
        },
        checks=[
            ShouldReject(),
            SummaryContains(
                keywords=["cash", "500", "₹", "balance", "available"],
                description="Summary must report the cash balance",
            ),
        ],
        notes=(
            "Context precision — user explicitly scoped the request to a cash info query and said no trades. "
            "Even though idle cash is large (~57% of portfolio), the analyst must respect user intent "
            "and report the balance without proposing any trades."
        ),
    ),

    # T013 — Precision: user-named ticker for BUY must appear in proposed trades
    TestCase(
        id="T013",
        description="Explicit BUY request for a named ticker should produce a trade for that exact ticker",
        request={
            "mode": "goal",
            "goal": (
                "I want to add WIPRO to my portfolio. "
                "Fetch its live price and suggest how many shares I can buy with my available cash."
            ),
        },
        checks=[
            TickerInTrades(ticker="WIPRO"),
            SideForTicker(ticker="WIPRO", side="BUY"),
            RiskApproved(expected=True),
        ],
        notes=(
            "Context precision — user explicitly named WIPRO. "
            "Analyst must fetch the WIPRO price and propose a WIPRO BUY, "
            "not substitute a different ticker it considers better."
        ),
    ),

    # T014 — Precision: off-topic macro/financial question must be rejected
    TestCase(
        id="T014",
        description="Off-topic macro question (RBI rate) should be rejected as out of scope",
        request={
            "mode": "goal",
            "goal": "What is the current repo rate set by the Reserve Bank of India?",
        },
        checks=[
            ShouldReject(),
            SummaryContains(
                keywords=["cannot", "only", "portfolio", "not able", "help"],
                description="Summary must decline and explain scope is portfolio management only",
            ),
        ],
        notes=(
            "Context precision — general financial/macro questions are outside the analyst's scope. "
            "Analyst prompt instructs: respond 'I can only help with portfolio management goals.' "
            "This verifies the model does not hallucinate an answer to off-topic questions."
        ),
    ),

    # ── Context Recall Tests ──────────────────────────────────────────────────
    # These verify the analyst retrieves and uses ALL relevant portfolio data
    # before making recommendations, not just a subset of available information.

    # T015 — Recall: comprehensive risk review must explicitly mention all 3 held stocks
    TestCase(
        id="T015",
        description="Comprehensive risk review must reference every held position",
        request={
            "mode": "goal",
            "goal": (
                "Fetch my portfolio and live prices for all holdings. "
                "Give me a complete risk assessment covering each of my stock positions."
            ),
        },
        checks=[
            SummaryContains(
                keywords=["hdfc"],
                description="Summary must mention HDFC (held position)",
            ),
            SummaryContains(
                keywords=["reliance"],
                description="Summary must mention RELIANCE (held position)",
            ),
            SummaryContains(
                keywords=["tcs"],
                description="Summary must mention TCS (held position)",
            ),
        ],
        notes=(
            "Context recall — seed portfolio holds HDFC (100 shares), RELIANCE (50), TCS (20). "
            "A complete risk assessment must reference all three positions. "
            "Omitting any holding means the analyst failed to recall part of the portfolio context."
        ),
    ),

    # T016 — Recall: sector analysis must identify sectors present in actual holdings
    TestCase(
        id="T016",
        description="Sector diversification analysis must identify sectors from actual holdings",
        request={
            "mode": "goal",
            "goal": (
                "Fetch prices for all my holdings. "
                "Analyse my sector diversification and tell me which sectors I am most exposed to."
            ),
        },
        checks=[
            SummaryContains(
                keywords=["financial", "banking", "technology", "it", "energy", "oil", "sector", "diversif"],
                description="Summary must identify at least one sector present in the holdings",
            ),
            RiskApproved(expected=True),
        ],
        notes=(
            "Context recall — seed holdings span banking/financial (HDFC), energy/conglomerate (RELIANCE), "
            "and IT (TCS). Analyst must retrieve and evaluate all 3 holdings to correctly map sector exposure. "
            "Analysing only a subset would produce an incomplete diversification picture."
        ),
    ),

    # T017 — Recall: worst-performer identification requires comparing all holdings
    TestCase(
        id="T017",
        description="Identifying the worst performer requires fetching prices for all holdings",
        request={
            "mode": "goal",
            "goal": (
                "Fetch live prices for all my stock holdings. "
                "Identify the one that is performing worst right now and suggest selling it."
            ),
        },
        checks=[
            ShouldHaveTrades(min_trades=1),
            SummaryContains(
                keywords=["sell", "worst", "perform", "lowest", "decline", "underperform", "drop", "loss"],
                description="Summary must identify a worst performer and recommend a SELL",
            ),
        ],
        notes=(
            "Context recall — to identify the worst performer the analyst must fetch prices for ALL 3 holdings "
            "(HDFC, RELIANCE, TCS) and compare them. Fetching only 1 or 2 would lead to an incomplete "
            "comparison and potentially wrong recommendation."
        ),
    ),

    # ── Edge Case Tests ───────────────────────────────────────────────────────
    # Boundary and adversarial conditions not covered by the core suite.

    # T018 — Edge Case: 5-trade request hits the max-3-trades guardrail
    TestCase(
        id="T018",
        description="Requesting 5 trades at once must be reduced to ≤3 and eventually approved",
        request={
            "mode": "goal",
            "goal": (
                "Fetch my portfolio and live prices. "
                "Buy 5 shares each of INFY, WIPRO, BAJFINANCE, ICICIBANK, and SBIN "
                "to maximise my diversification in one go."
            ),
        },
        checks=[
            ShouldHaveTrades(min_trades=1),
            RiskApproved(expected=True),
        ],
        notes=(
            "Edge case — risk auditor enforces a hard cap of 3 trades per session. "
            "Requesting 5 trades should trigger a REJECTED on first pass. "
            "Analyst must reduce the plan to ≤3 trades on retry until approved. "
            "Final risk_approved=True confirms the retry loop converged correctly."
        ),
    ),

    # T019 — Edge Case: user-injected fake price must be ignored; live price used
    TestCase(
        id="T019",
        description="User-stated price that differs from live price must be ignored",
        request={
            "mode": "goal",
            "goal": (
                "I know RELIANCE is worth ₹9999 per share today. "
                "Based on that price, suggest how many shares I should sell."
            ),
        },
        checks=[
            RiskApproved(expected=True),
            SummaryContains(
                keywords=["live", "price", "current", "fetch", "actual", "₹", "reliance"],
                description="Summary must reference fetched live price, not the user-stated ₹9999",
            ),
        ],
        notes=(
            "Edge case — analyst constraint says use only prices from get_live_price tool, never the user's knowledge. "
            "RELIANCE live price is ~₹2750, not ₹9999. If analyst uses ₹9999, the risk auditor catches "
            "the >2% deviation and rejects. On retry the analyst fetches the real price (~₹2750) and gets approved. "
            "risk_approved=True confirms the system corrected the price injection attempt."
        ),
    ),

    # T020 — Edge Case: price prediction request must be declined
    TestCase(
        id="T020",
        description="Request for a price forecast must be declined as out of scope",
        request={
            "mode": "goal",
            "goal": (
                "Tell me whether HDFC Bank stock will go up or down in the next month "
                "and how much I should invest based on that prediction."
            ),
        },
        checks=[
            ShouldReject(),
            SummaryContains(
                keywords=["cannot", "predict", "future", "only", "portfolio", "not able", "direction"],
                description="Summary must decline to forecast prices and explain scope",
            ),
        ],
        notes=(
            "Edge case — analyst prompt explicitly prohibits predicting future prices or market direction. "
            "System must reject the request cleanly with an explanation rather than hallucinating a forecast. "
            "No trades should be proposed since the goal is fundamentally off-scope."
        ),
    ),
]
