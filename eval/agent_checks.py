"""
Agent-capability check primitives.

These operate on the full API response dict which now includes:
  - tool_calls_log : list of {name, args, result, order, ticker?}
  - proposed_trades: list of trade dicts
  - decision_summary: str
  - risk_approved  : bool
  - retry_count    : int
"""
from dataclasses import dataclass, field
from typing import List


# ── Tool call checks ─────────────────────────────────────────────────────────

@dataclass
class ToolWasCalled:
    """Assert that a specific tool was called at least once."""
    tool_name: str
    description: str = ""
    def __post_init__(self):
        if not self.description:
            self.description = f"Tool '{self.tool_name}' must be called"


@dataclass
class ToolNotCalled:
    """Assert that a specific tool was NOT called."""
    tool_name: str
    description: str = ""
    def __post_init__(self):
        if not self.description:
            self.description = f"Tool '{self.tool_name}' must NOT be called"


@dataclass
class ToolCallCount:
    """Assert total number of tool calls is within [min_calls, max_calls]."""
    min_calls: int = 0
    max_calls: int = 999
    description: str = ""
    def __post_init__(self):
        if not self.description:
            self.description = f"Total tool calls must be in [{self.min_calls}, {self.max_calls}]"


@dataclass
class PortfolioFetchedFirst:
    """Assert get_portfolio is called before any get_live_price call."""
    description: str = "get_portfolio must be called before get_live_price"


@dataclass
class SpecificTickerFetched:
    """Assert get_live_price was called for a specific ticker."""
    ticker: str
    description: str = ""
    def __post_init__(self):
        if not self.description:
            self.description = f"get_live_price must be called for ticker '{self.ticker}'"


@dataclass
class AllHoldingsFetched:
    """Assert get_live_price was called for every ticker in the seed portfolio."""
    seed_tickers: List[str] = field(default_factory=lambda: ["AAPL", "MSFT", "JPM"])
    description: str = ""
    def __post_init__(self):
        if not self.description:
            self.description = f"Live price must be fetched for all holdings: {self.seed_tickers}"


@dataclass
class NoToolsOnOffTopic:
    """Assert zero tool calls were made (off-topic request should abort before tools)."""
    description: str = "No tools must be called for off-topic requests"


@dataclass
class ToolCallOrder:
    """Assert tool A appears before tool B in the call log (by order index)."""
    first: str   # tool name that must come first
    second: str  # tool name that must come after
    description: str = ""
    def __post_init__(self):
        if not self.description:
            self.description = f"'{self.first}' must be called before '{self.second}'"


# ── Retry checks ─────────────────────────────────────────────────────────────

@dataclass
class RetryOccurred:
    """Assert at least one retry happened (retry_count >= 1)."""
    description: str = "At least one retry must have occurred"


@dataclass
class RetryConverged:
    """Assert retries eventually led to risk_approved=True."""
    description: str = "Retry loop must converge to risk_approved=True"


@dataclass
class MaxRetriesHit:
    """Assert retry_count reached MAX_RETRIES (3) without approval."""
    max_retries: int = 3
    description: str = ""
    def __post_init__(self):
        if not self.description:
            self.description = f"retry_count must equal {self.max_retries} (exhausted)"


@dataclass
class RetryCountAtMost:
    """Assert retry_count <= n (converged quickly)."""
    n: int
    description: str = ""
    def __post_init__(self):
        if not self.description:
            self.description = f"retry_count must be <= {self.n}"


# ── Price grounding checks ────────────────────────────────────────────────────

@dataclass
class PriceGrounded:
    """
    Assert proposed trade price is within `tolerance` of the fetched live price.
    Grounding precision = fraction of trades whose price is within tolerance.
    Pass threshold: >= min_grounded_fraction of trades must be grounded.
    """
    tolerance: float = 0.02      # 2 % — mirrors risk auditor rule
    min_grounded_fraction: float = 1.0   # all trades must be grounded by default
    description: str = ""
    def __post_init__(self):
        if not self.description:
            self.description = (
                f"All proposed trade prices must be within {int(self.tolerance*100)}% "
                f"of fetched live price (grounding >= {int(self.min_grounded_fraction*100)}%)"
            )


# ── LLM quality checks (precision / recall) ──────────────────────────────────

@dataclass
class ContextPrecision:
    """
    Precision = tickers_used_in_recommendation / tickers_fetched_via_tool.
    Measures whether the analyst fetched only what it needed.
    Pass threshold: >= min_precision.
    """
    min_precision: float = 0.5
    description: str = ""
    def __post_init__(self):
        if not self.description:
            self.description = (
                f"Context precision must be >= {self.min_precision:.0%} "
                "(tickers used in summary / tickers fetched via tool)"
            )


@dataclass
class ContextRecall:
    """
    Recall = tickers_mentioned_in_summary / tickers_in_portfolio.
    Measures whether the analyst covered every holding in its analysis.
    Pass threshold: >= min_recall.
    """
    portfolio_tickers: List[str] = field(default_factory=lambda: ["AAPL", "MSFT", "JPM"])
    min_recall: float = 0.67   # must mention at least 2 of 3 holdings
    description: str = ""
    def __post_init__(self):
        if not self.description:
            self.description = (
                f"Context recall must be >= {self.min_recall:.0%} "
                "(holdings mentioned in summary / total holdings)"
            )


@dataclass
class AnswerRelevance:
    """
    Soft relevance: summary must contain at least `min_keywords` from the
    provided keyword list (goal-specific signals).
    """
    keywords: List[str]
    min_keywords: int = 1
    description: str = ""
    def __post_init__(self):
        if not self.description:
            self.description = (
                f"Summary must contain >= {self.min_keywords} of: {self.keywords}"
            )


@dataclass
class Faithfulness:
    """
    Faithfulness: every ticker mentioned in proposed trades must have had
    its live price fetched via tool (no hallucinated tickers in trades).
    """
    description: str = (
        "Every ticker in proposed trades must have a corresponding price fetch call"
    )


# ── Batch pricing checks (DJI 30 / full rebalance) ───────────────────────────

@dataclass
class GetPricesBatchCalled:
    """Assert get_prices_batch was called at least once."""
    description: str = "get_prices_batch must be called (holistic / full rebalance)"


@dataclass
class GetPricesBatchNotCalled:
    """Assert get_prices_batch was NOT called (targeted single-ticker queries)."""
    description: str = "get_prices_batch must NOT be called for targeted queries"


@dataclass
class BatchTickerCount:
    """
    Assert the get_prices_batch call included at least `min_tickers` tickers.
    For full rebalance this should be 30 (all DJI tickers).
    """
    min_tickers: int
    description: str = ""
    def __post_init__(self):
        if not self.description:
            self.description = f"get_prices_batch must be called with >= {self.min_tickers} tickers"


# ── Hallucination checks ─────────────────────────────────────────────────────

@dataclass
class NoHallucinatedTickers:
    """
    Assert the decision_summary does not mention any ticker that is NOT in the
    user's portfolio (as reflected by the tool_calls_log get_portfolio result)
    AND was not fetched via a price tool.

    portfolio_tickers: the known holdings for this user. Any ticker in the
    summary that is absent from this list AND was not fetched is a hallucination.
    """
    portfolio_tickers: List[str]
    description: str = ""
    def __post_init__(self):
        if not self.description:
            self.description = (
                f"Summary must not mention tickers outside the portfolio "
                f"{self.portfolio_tickers} unless their price was fetched"
            )


# ── Trade count checks ────────────────────────────────────────────────────────

@dataclass
class TradeCountAtMost:
    """Assert the proposed_trades list contains at most `max_trades` entries."""
    max_trades: int
    description: str = ""
    def __post_init__(self):
        if not self.description:
            self.description = f"Proposed trade count must be <= {self.max_trades}"

