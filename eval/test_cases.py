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

    # T003 — AI Feedback mode: must propose at least one trade
    TestCase(
        id="T003",
        description="AI Feedback mode must produce a full review with trades",
        request={
            "mode": "feedback",
        },
        checks=[
            ShouldHaveTrades(min_trades=1),
            SummaryContains(
                keywords=["concentration", "diversif", "risk", "sector", "rebalanc", "cash"],
                description="Summary must cover portfolio health themes",
            ),
        ],
        notes="Feedback mode uses a fixed system prompt — should always yield analysis",
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
]
