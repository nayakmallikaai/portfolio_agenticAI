"""
LangGraph multi-agent workflow.
All LLM and graph logic lives here; no HTTP or DB imports.
"""
import json
import operator
import re
from typing import Annotated, TypedDict, Dict, Any, Optional

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import ToolMessage, SystemMessage, HumanMessage, AIMessage
from mcp import ClientSession

from agent.parsing import parse_proposed_trades, extract_trades_via_llm

MAX_RETRIES = 3

# Full DJI 30 ticker list — passed to get_prices_batch for full rebalance mode
_DJI_30_TICKERS = [
    "AAPL", "MSFT", "AMZN", "IBM",  "CSCO", "CRM",  "NVDA",
    "JPM",  "GS",   "AXP",  "V",    "TRV",
    "JNJ",  "MRK",  "UNH",  "AMGN",
    "BA",   "CAT",  "HON",  "MMM",  "CVX",
    "KO",   "WMT",  "MCD",  "PG",   "NKE",
    "DIS",  "VZ",   "HD",   "SHW",
]


# ── Analysis type detectors ────────────────────────────────────────────────────

def is_full_rebalance(mode: str, goal: str) -> bool:
    """
    True when the user explicitly requests a full portfolio rebalance across
    the entire DJI 30 universe. Triggers:
      - get_prices_batch called with ALL 30 tickers
      - relaxed risk auditor (up to 5 trades, total notional ≤ 20% portfolio)
    """
    phrases = [
        "entire portfolio", "full rebalance", "complete rebalance",
        "rebalance entire", "rebalance my entire", "rebalance everything",
        "rebalance all", "full portfolio rebalance", "restructure my portfolio",
        "overhaul", "rebalance to dow", "rebalance toward", "rebalance across",
        "consider all stocks", "all 30", "across all",
    ]
    goal_lower = goal.lower()
    return any(p in goal_lower for p in phrases)


def is_holistic_analysis(mode: str, goal: str) -> bool:
    """
    True when the request needs prices for all currently HELD tickers
    (but not the full DJI universe). Uses get_prices_batch over holdings.
    """
    if mode == "feedback":
        return True
    keywords = [
        "all", "entire", "whole", "complete", "overall", "everything", "rebalanc", "diversif", "review", "assess", "health",
        "overview", "worst", "best performer", "sector", "allocation",
        "weight", "each", "every", "compare",
    ]
    return any(kw in goal.lower() for kw in keywords)


# ── Cached prompt constants ────────────────────────────────────────────────────

_ANALYST_STATIC = """\
SUPPORTED UNIVERSE: DOW JONES INDUSTRIAL AVERAGE (30 STOCKS)
You may only trade tickers from the Dow Jones 30 listed below.
All portfolio values and trade prices are in US Dollars (USD).
Any goal referencing a ticker outside this list must be declined.

TICKER  | SECTOR                   | COMPANY
--------|--------------------------|-----------------------------------
AAPL    | Technology               | Apple
MSFT    | Technology / Cloud       | Microsoft
AMZN    | E-commerce / Cloud       | Amazon
IBM     | IT Services / Cloud      | IBM
CSCO    | Networking               | Cisco Systems
CRM     | Enterprise Software      | Salesforce
NVDA    | Semiconductors / AI      | Nvidia
JPM     | Banking                  | JPMorgan Chase
GS      | Investment Banking       | Goldman Sachs
AXP     | Financial Services       | American Express
V       | Payments                 | Visa
TRV     | Insurance                | Travelers Companies
JNJ     | Healthcare / Pharma      | Johnson & Johnson
MRK     | Pharmaceuticals          | Merck
UNH     | Healthcare Insurance     | UnitedHealth Group
AMGN    | Biotechnology            | Amgen
BA      | Aerospace / Defense      | Boeing
CAT     | Industrial Machinery     | Caterpillar
HON     | Industrial Conglomerate  | Honeywell
MMM     | Diversified Industrial   | 3M
CVX     | Energy / Oil & Gas       | Chevron
KO      | Beverages / FMCG         | Coca-Cola
WMT     | Retail                   | Walmart
MCD     | Fast Food / Restaurants  | McDonald's
PG      | Consumer Goods           | Procter & Gamble
NKE     | Apparel / Consumer       | Nike
DIS     | Entertainment / Media    | Walt Disney
VZ      | Telecommunications       | Verizon
HD      | Home Improvement Retail  | Home Depot
SHW     | Paints / Coatings        | Sherwin-Williams

PORTFOLIO HEALTH GUIDELINES:
  Concentration risk  : A single stock exceeding 40% of total equity value is over-concentrated. Flag it.
  Cash drag           : Cash exceeding 50% of total portfolio value (equity + cash) is excessive idle capital.
  Sector exposure     : A portfolio with more than 60% in a single sector has sector concentration risk.
  Diversification     : A healthy portfolio holds at least 3 distinct sectors.
  Trade sizing        : A single trade should not exceed 20% of total portfolio value.
  Minimum trade size  : Do not propose trades smaller than 1 share or of negligible monetary value.

FULL PORTFOLIO REBALANCE RULES (only when explicitly requested):
  - You will receive prices for all 30 Dow Jones tickers. Evaluate the full universe.
  - Propose at most 5 trades. Prioritise the highest-impact changes only.
  - The combined notional value of ALL proposed trades must not exceed 20% of total portfolio value.
    Formula: sum(qty × price for every trade) / total_portfolio_value ≤ 0.20
  - Focus on: reducing single-stock concentration, adding missing sectors, trimming overweight positions.
  - Do not churn the portfolio — only suggest changes with clear, specific justification.

ANALYSIS APPROACH:
- Read the goal first. Only act if the portfolio genuinely needs to change.
- Propose nothing if the goal is already met or the user only asks for information.
- When trades are needed, keep them small, specific, and data-grounded.
- Always fetch live prices before quoting any value or proposing any trade.
- If cash exceeds 50% of portfolio, flag this prominently before any other observation.

CONSTRAINTS:
- Only use prices from live_prices state or price tools — never your own knowledge.
- Only trade the tickers listed above. Reject any goal naming an unsupported ticker.
- Never sell more shares than the user currently holds.
- Never propose a BUY whose total cost exceeds available cash.
- Do not predict future prices, earnings, or market direction under any circumstances.
- If a price is unavailable for a ticker, note it and continue with the remaining data.
- If the user asks anything unrelated to their portfolio (macro questions, jokes, general finance),
  respond only with: "I can only help with portfolio management goals."

RESPONSE FORMAT — NON NEGOTIABLE:
- Plain conversational text only. No markdown, no tables, no headers, no bullet points, no emojis, no bold.
- Maximum 80 words across a maximum of 2 paragraphs.
- End every response with this exact JSON block (no text after it):
{"trades": [{"ticker": "X", "side": "BUY/SELL", "qty": 0, "price": 0.0}]}
If no trades are needed: {"trades": []}"""


_RISK_AUDITOR_STATIC = """\
You are a risk auditor for a US equity portfolio management system.
Your sole job is to audit a proposed trade plan against strict rules and return a structured verdict.

SUPPORTED TICKERS WHITELIST (Dow Jones 30):
  AAPL, MSFT, AMZN, IBM, CSCO, CRM, NVDA, JPM, GS, AXP, V, TRV,
  JNJ, MRK, UNH, AMGN, BA, CAT, HON, MMM, CVX, KO, WMT, MCD,
  PG, NKE, DIS, VZ, HD, SHW
Any ticker outside this list must be rejected immediately.

HARD REJECTION RULES — reject if ANY of the following are true:
  1. PRICE DEVIATION   : Any proposed trade price deviates more than 2% from the fetched live price.
                         Formula: abs(proposed_price - live_price) / live_price > 0.02
  2. OVERSELL          : A SELL order quantity exceeds the number of shares currently held.
  3. OVERCASH          : The total cost of all BUY orders exceeds the user's available cash balance.
  4. UNSUPPORTED TICKER: Any ticker in proposed_trades is not in the whitelist above.
  5. TRADE COUNT       : More than 3 trades are proposed in a single plan.
  6. TRADE SIZE        : Any single trade's notional value (qty × price) exceeds 30% of total portfolio value.
  7. UNSUPPORTED CLAIMS: The analyst's reasoning references prices, holdings, or facts not in the provided data.
  8. AGGRESSION        : The plan is disproportionately aggressive relative to the stated goal.

APPROVAL CONDITIONS — approve only when ALL hold:
  - All proposed prices within 2% of live prices.
  - All SELL quantities within current holdings.
  - Total BUY cost within available cash.
  - All tickers on the whitelist.
  - Three or fewer trades proposed.
  - No single trade exceeds 30% of portfolio value.
  - Plan is proportionate and conservative relative to the user's goal.
  - Empty trades array is always valid.

FEEDBACK GUIDANCE — when rejecting, your FEEDBACK must be specific and actionable:
  - Name the exact rule violated and the exact ticker or value that caused it.
  - Tell the analyst precisely what to change.
  - Do not give vague feedback like "reconsider the plan".

OUTPUT FORMAT (total response under 60 words):
DECISION: APPROVED or REJECTED
REASON: one sentence explaining the decision
FEEDBACK: (only if REJECTED) one specific, actionable fix for the analyst"""


_RISK_AUDITOR_REBALANCE_STATIC = """\
You are a risk auditor reviewing a FULL PORTFOLIO REBALANCE plan.
The analyst evaluated the complete Dow Jones 30 universe and must propose
conservative incremental changes — not a full reconstruction. Apply the rules below.
The trade count cap is 5 (relaxed from 3) but a cumulative notional cap is added.

SUPPORTED TICKERS WHITELIST (Dow Jones 30):
  AAPL, MSFT, AMZN, IBM, CSCO, CRM, NVDA, JPM, GS, AXP, V, TRV,
  JNJ, MRK, UNH, AMGN, BA, CAT, HON, MMM, CVX, KO, WMT, MCD,
  PG, NKE, DIS, VZ, HD, SHW
Any ticker outside this list must be rejected immediately.

HARD REJECTION RULES — reject if ANY of the following are true:
  1. PRICE DEVIATION   : Any proposed trade price deviates more than 2% from the fetched live price.
  2. OVERSELL          : A SELL quantity exceeds the number of shares currently held for that ticker.
  3. OVERCASH          : Total cost of all BUY orders exceeds the user's available cash balance.
  4. UNSUPPORTED TICKER: Any ticker not in the whitelist above.
  5. TRADE COUNT       : More than 5 trades proposed.
  6. TRADE SIZE        : Any single trade's notional value (qty × price) exceeds 20% of total portfolio value.
  7. TOTAL IMPACT      : The combined notional value of ALL trades exceeds 20% of total portfolio value.
                         Formula: sum(qty × price for all trades) / total_portfolio_value > 0.20
  8. UNSUPPORTED CLAIMS: Analyst references prices, holdings, or facts not in the provided data.
  9. AGGRESSION        : Plan attempts a major reconstruction rather than incremental improvement.
                         Example: liquidating all holdings when asked for a balanced rebalance.

APPROVAL CONDITIONS — approve only when ALL hold:
  - All prices within 2% of live prices.
  - No oversells; total BUY cost within cash.
  - All tickers on the whitelist.
  - 5 or fewer trades.
  - No single trade exceeds 20% of portfolio value.
  - Combined notional of all trades ≤ 20% of total portfolio value.
  - Plan is incremental — adds diversification or reduces concentration, does not overhaul.
  - Empty trades array is always valid if no changes are warranted.

FEEDBACK GUIDANCE — when rejecting, be specific:
  - For TOTAL IMPACT: state the actual combined notional and the 20% maximum.
  - For TRADE COUNT: name which trades to drop (lowest-impact ones first).
  - For TRADE SIZE: name the offending ticker and the corrected quantity.

OUTPUT FORMAT (total response under 80 words):
DECISION: APPROVED or REJECTED
REASON: one sentence explaining the decision
FEEDBACK: (only if REJECTED) one specific, actionable fix for the analyst"""


# ── Shared graph state ─────────────────────────────────────────────────────────
class PortfolioState(TypedDict):
    messages: Annotated[list, add_messages]
    portfolio_snapshot: dict
    live_prices: dict
    risk_approved: bool
    risk_feedback: str
    risk_note: str
    retry_count: int
    tool_calls_log: Annotated[list, operator.add]


# ── Main entry point ───────────────────────────────────────────────────────────
async def run_analysis(
    goal: str,
    user_id: str,
    mcp_holder: Dict[str, Any],
    mode: str = "goal",
) -> Dict[str, Any]:
    """
    Run the full analyst → tools → risk-auditor loop.
    Returns decision_summary, risk_approved, proposed_trades, retry_count.
    """
    session: ClientSession = mcp_holder["session"]

    # ── Determine analysis type (deterministic, before graph runs) ─────────────
    full_rebalance = is_full_rebalance(mode, goal)
    holistic       = full_rebalance or is_holistic_analysis(mode, goal)

    # ── Filter tools exposed to the LLM ───────────────────────────────────────
    # record_trade is always hidden (human approval gate).
    # get_live_price hidden for holistic/rebalance — batch tool is available.
    # get_prices_batch hidden for targeted — sequential tool is sufficient.
    mcp_tools_resp = await session.list_tools()
    read_tools = []
    for t in mcp_tools_resp.tools:
        if t.name == "record_trade":
            continue
        if holistic and t.name == "get_live_price":
            continue
        if not holistic and t.name == "get_prices_batch":
            continue
        schema = dict(t.inputSchema)
        props    = {k: v for k, v in schema.get("properties", {}).items() if k != "user_id"}
        required = [r for r in schema.get("required", []) if r != "user_id"]
        read_tools.append({
            "name": t.name,
            "description": t.description,
            "input_schema": {**schema, "properties": props, "required": required},
        })

    llm_with_tools = ChatAnthropic(model="claude-sonnet-4-6", temperature=0).bind_tools(read_tools)
    llm_no_tools   = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)

    # ── NODE 1: Analyst ────────────────────────────────────────────────────────
    def analyst_node(state: PortfolioState):
        print(f"\n{'='*60}\n[ANALYST NODE] full_rebalance={full_rebalance} holistic={holistic}")

        has_portfolio = bool(state.get("portfolio_snapshot"))
        has_prices    = bool(state.get("live_prices"))

        if has_portfolio and has_prices:
            data_section = (
                f"Portfolio: {json.dumps(state['portfolio_snapshot'])}\n"
                f"Live prices: {json.dumps(state['live_prices'])}"
            )
        elif has_portfolio:
            held = list(state["portfolio_snapshot"].get("holdings", {}).keys())
            if full_rebalance:
                data_section = (
                    f"Portfolio already fetched — do NOT call get_portfolio again.\n"
                    f"Portfolio: {json.dumps(state['portfolio_snapshot'])}\n"
                    f"This is a FULL PORTFOLIO REBALANCE. Your ONLY next action is to call "
                    f"get_prices_batch with ALL 30 Dow Jones tickers: {json.dumps(_DJI_30_TICKERS)}"
                )
            else:
                data_section = (
                    f"Portfolio already fetched — do NOT call get_portfolio again.\n"
                    f"Portfolio: {json.dumps(state['portfolio_snapshot'])}\n"
                    f"Your ONLY next action is to call get_prices_batch with these tickers: {json.dumps(held)}"
                )
        else:
            if full_rebalance:
                data_section = (
                    "Portfolio and prices not yet fetched. "
                    f"Call get_portfolio first, then get_prices_batch with ALL 30 Dow Jones "
                    f"tickers: {json.dumps(_DJI_30_TICKERS)}"
                )
            elif holistic:
                data_section = (
                    "Portfolio and prices not yet fetched. "
                    "Call get_portfolio first, then use get_prices_batch with all your holding tickers."
                )
            else:
                data_section = (
                    "Portfolio and prices not yet fetched. "
                    "Use your tools to gather the data you need before making recommendations."
                )

        feedback_section = ""
        if state.get("risk_feedback"):
            feedback_section = (
                f"\nPrevious plan was REJECTED by the risk auditor. "
                f"Fix these issues before proposing again:\n{state['risk_feedback']}"
            )

        rebalance_flag = "\nANALYSIS MODE: FULL_PORTFOLIO_REBALANCE\n" if full_rebalance else ""

        sys_msg = SystemMessage(content=[
            {
                "type": "text",
                "text": _ANALYST_STATIC,
                "cache_control": {"type": "ephemeral"},
            },
            {
                "type": "text",
                "text": (
                    f"{rebalance_flag}"
                    f"You are a portfolio analyst for user '{user_id}'. "
                    f"Analyse the portfolio and address the user's goal.\n\n"
                    f"{data_section}{feedback_section}"
                ),
            },
        ])

        clean_history = [
            m for m in state["messages"]
            if isinstance(m, HumanMessage)
            or (isinstance(m, AIMessage) and not getattr(m, "tool_calls", None))
        ]
        # Claude does not support assistant prefill — strip trailing AIMessages
        # so the conversation always ends with a HumanMessage.
        while clean_history and isinstance(clean_history[-1], AIMessage):
            clean_history.pop()

        active_llm = llm_no_tools if (has_portfolio and has_prices) else llm_with_tools
        response = active_llm.invoke([sys_msg] + clean_history)
        print(f"[ANALYST] content: {str(response.content)[:300]}")
        print(f"[ANALYST] tool_calls: {[tc['name'] for tc in response.tool_calls] if response.tool_calls else 'NONE'}")
        return {"messages": [response]}

    # ── NODE 2: Tool execution ─────────────────────────────────────────────────
    async def tool_node(state: PortfolioState):
        last_msg = state["messages"][-1]
        print(f"\n[TOOL NODE] {len(last_msg.tool_calls)} call(s)")

        tool_messages      = []
        portfolio_snapshot = state.get("portfolio_snapshot", {})
        live_prices        = state.get("live_prices", {})
        call_order         = len(state.get("tool_calls_log", []))
        new_log_entries    = []

        for tc in last_msg.tool_calls:
            args = {**tc["args"], "user_id": user_id}
            print(f"  {tc['name']} args={args}")
            result = await session.call_tool(tc["name"], args)
            raw    = result.content[0].text
            print(f"  → {raw[:200]}")

            log_entry = {
                "name": tc["name"],
                "args": tc["args"],
                "result": raw[:500],
                "order": call_order,
            }

            if tc["name"] == "get_portfolio":
                try:
                    portfolio_snapshot = json.loads(raw)
                except Exception:
                    portfolio_snapshot = {"raw": raw}
                tool_messages.append(ToolMessage(
                    tool_call_id=tc["id"],
                    content="Portfolio snapshot updated in state.",
                ))

            elif tc["name"] == "get_live_price":
                ticker = tc["args"].get("ticker", "UNKNOWN").upper()
                log_entry["ticker"] = ticker
                try:
                    live_prices[ticker] = float(raw)
                except ValueError:
                    live_prices[ticker] = raw
                tool_messages.append(ToolMessage(
                    tool_call_id=tc["id"],
                    content=f"Live price for {ticker} updated in state: {raw}",
                ))

            elif tc["name"] == "get_prices_batch":
                log_entry["tickers"] = [t.upper() for t in tc["args"].get("tickers", [])]
                try:
                    batch = json.loads(raw)
                    for ticker_key, price_val in batch.items():
                        if price_val is not None:
                            live_prices[ticker_key.upper()] = price_val
                    fetched = sum(1 for v in batch.values() if v is not None)
                    tool_messages.append(ToolMessage(
                        tool_call_id=tc["id"],
                        content=f"Batch prices updated in state: {fetched}/{len(batch)} tickers fetched.",
                    ))
                except Exception:
                    tool_messages.append(ToolMessage(
                        tool_call_id=tc["id"],
                        content=raw,
                    ))

            else:
                tool_messages.append(ToolMessage(
                    tool_call_id=tc["id"],
                    content=raw,
                ))

            new_log_entries.append(log_entry)
            call_order += 1

        return {
            "messages": tool_messages,
            "portfolio_snapshot": portfolio_snapshot,
            "live_prices": live_prices,
            "tool_calls_log": new_log_entries,
        }

    # ── NODE 3: Risk auditor ───────────────────────────────────────────────────
    def risk_node(state: PortfolioState):
        print(f"\n[RISK NODE] full_rebalance={full_rebalance}")

        portfolio = json.dumps(state.get("portfolio_snapshot", {}))
        prices    = json.dumps(state.get("live_prices", {}))

        last_analyst = ""
        for msg in reversed(state["messages"]):
            if getattr(msg, "type", None) == "ai" and isinstance(msg.content, str) and msg.content.strip():
                last_analyst = msg.content
                break

        trades_block = ""
        match = re.search(r'\{"trades":\s*\[.*?\]\s*\}', last_analyst, re.DOTALL)
        if match:
            trades_block = match.group(0)

        conversation_text = (
            f"Portfolio: {portfolio}\n"
            f"Live prices: {prices}\n"
            f"Proposed trades: {trades_block or 'none'}"
        )
        print(f"[RISK] Sending:\n{conversation_text[:400]}")

        # Select the appropriate auditor prompt based on analysis type
        auditor_prompt = _RISK_AUDITOR_REBALANCE_STATIC if full_rebalance else _RISK_AUDITOR_STATIC

        audit_llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)
        response = audit_llm.invoke([
            SystemMessage(content=[
                {
                    "type": "text",
                    "text": auditor_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ]),
            HumanMessage(content=conversation_text),
        ])
        print(f"[RISK] Response: {response.content}")

        approved     = "APPROVED" in response.content.upper()
        retry_count  = state.get("retry_count", 0)
        risk_note    = ""
        risk_feedback_text = ""

        for line in response.content.strip().splitlines():
            line = line.strip()
            if line.upper().startswith("REASON:"):
                risk_note = line.split(":", 1)[-1].strip()
            elif line.upper().startswith("FEEDBACK:"):
                risk_feedback_text = line.split(":", 1)[-1].strip()

        result: Dict[str, Any] = {
            "risk_approved": approved,
            "risk_note":     risk_note,
            "retry_count":   retry_count + 1,
        }
        if not approved:
            result["risk_feedback"] = risk_feedback_text or risk_note

        return result

    # ── Routing ────────────────────────────────────────────────────────────────
    def route_analyst(state: PortfolioState):
        has_tools = bool(getattr(state["messages"][-1], "tool_calls", []))
        print(f"\n[ROUTE] analyst → {'execute_tools' if has_tools else 'audit_risk'}")
        return "execute_tools" if has_tools else "audit_risk"

    def route_risk(state: PortfolioState):
        if state.get("risk_approved", False):
            print("[ROUTE] risk → END (approved)")
            return END
        if state.get("retry_count", 0) >= MAX_RETRIES:
            print(f"[ROUTE] risk → END (hard limit {MAX_RETRIES})")
            return END
        print(f"[ROUTE] risk → analyst (retry {state.get('retry_count', 0)})")
        return "analyst"

    # ── Graph assembly ─────────────────────────────────────────────────────────
    workflow = StateGraph(PortfolioState)
    workflow.add_node("analyst",       analyst_node)
    workflow.add_node("execute_tools", tool_node)
    workflow.add_node("audit_risk",    risk_node)
    workflow.set_entry_point("analyst")
    workflow.add_conditional_edges("analyst", route_analyst)
    workflow.add_edge("execute_tools", "analyst")
    workflow.add_conditional_edges("audit_risk", route_risk, {"analyst": "analyst", END: END})
    graph = workflow.compile()

    # ── Run ────────────────────────────────────────────────────────────────────
    final_state = await graph.ainvoke({
        "messages":           [("user", goal)],
        "portfolio_snapshot": {},
        "live_prices":        {},
        "retry_count":        0,
        "risk_approved":      False,
        "risk_feedback":      "",
        "risk_note":          "",
        "tool_calls_log":     [],
    })

    last_analyst_content = ""
    for msg in reversed(final_state["messages"]):
        if getattr(msg, "type", None) == "ai" and isinstance(msg.content, str) and msg.content.strip():
            last_analyst_content = msg.content
            break

    proposed_trades, block_found = parse_proposed_trades(last_analyst_content)
    if not block_found:
        proposed_trades = extract_trades_via_llm(last_analyst_content)

    risk_approved = final_state.get("risk_approved", False)

    no_trade_reason = None
    if not proposed_trades:
        if risk_approved:
            no_trade_reason = "analyst_no_trade"
        else:
            no_trade_reason = "retries_exhausted"

    return {
        "decision_summary": last_analyst_content,
        "risk_approved":    risk_approved,
        "risk_note":        final_state.get("risk_note", ""),
        "proposed_trades":  proposed_trades,
        "retry_count":      final_state.get("retry_count", 0),
        "no_trade_reason":  no_trade_reason,
        "tool_calls_log":   final_state.get("tool_calls_log", []),
    }


# ── Helper ─────────────────────────────────────────────────────────────────────
def extract_rejection_reasons(risk_response: str) -> list[str]:
    reasons = []
    for line in risk_response.strip().split("\n"):
        line = line.strip()
        if not line or line.upper().startswith("REJECTED") or len(line) < 10:
            continue
        reasons.append(line)
    return reasons[:3]
