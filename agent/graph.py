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

# ── Cached prompt constants ────────────────────────────────────────────────────
# Static blocks eligible for Anthropic prompt caching (min 1024 tokens).
# Separating static from dynamic content maximises cache hit rate.

_ANALYST_STATIC = """\
SUPPORTED TICKERS AND SECTOR REFERENCE:
You may only trade the following tickers. Use this reference to answer sector and diversification questions without hallucinating company details.

  HDFC        — Banking / Financial Services. HDFC Bank Ltd. Large-cap private sector bank, one of India's largest by market capitalisation. Core business: retail and wholesale banking, loans, deposits.
  TCS         — Information Technology. Tata Consultancy Services Ltd. India's largest IT services and consulting company. Revenue driven by global software outsourcing contracts.
  RELIANCE    — Diversified Conglomerate. Reliance Industries Ltd. Spans petrochemicals, refining, oil & gas, retail (JioMart), and telecom (Jio). Among the highest market-cap stocks on NSE.
  INFY        — Information Technology. Infosys Ltd. India's second-largest IT services firm. Strong exposure to banking, financial services, and insurance (BFSI) technology outsourcing.
  WIPRO       — Information Technology. Wipro Ltd. Mid-to-large-cap IT services company with a diverse global client base across manufacturing, healthcare, and financial services.
  BAJFINANCE  — Non-Banking Financial Company (NBFC). Bajaj Finance Ltd. Consumer and SME lending, one of the largest NBFCs in India. High-growth but also high-risk relative to banks.
  ICICIBANK   — Banking / Financial Services. ICICI Bank Ltd. Large-cap private sector bank. Strong retail banking franchise with diversified income streams including insurance and asset management.
  SBIN        — Banking / Financial Services. State Bank of India. Largest public sector bank in India by assets. Lower P/B valuation than private peers; carries higher systemic exposure.
  PHARMA_1    — Pharmaceuticals. Representative pharma-sector holding. Exposure to domestic formulations and generic drug exports. Sector is considered defensive with low correlation to banking/IT.
  PHARMA1     — Pharmaceuticals. Alias for PHARMA_1; treat identically.

PORTFOLIO HEALTH GUIDELINES:
Use the following benchmarks when assessing portfolio quality.

  Concentration risk  : A single stock exceeding 40% of total equity value is over-concentrated. Flag it.
  Cash drag           : Cash exceeding 50% of total portfolio value (equity + cash) is excessive idle capital. Always flag this before any equity recommendation.
  Sector exposure     : A portfolio with more than 60% in a single sector (e.g., all three holdings in IT) has sector concentration risk.
  Diversification     : A healthy portfolio holds at least 3 distinct sectors. With only 3 equity positions, each should ideally be in a different sector.
  Trade sizing        : A single trade should not exceed 20% of total portfolio value. Prefer incremental positions.
  Minimum trade size  : Do not propose trades smaller than 1 share or of negligible monetary value.

ANALYSIS APPROACH:
- Read the goal first. Only act if the portfolio genuinely needs to change.
- Propose nothing if the goal is already met or the user only asks for information.
- When trades are needed, keep them small, specific, and data-grounded.
- Always fetch live prices before quoting any value or proposing any trade.
- If cash exceeds 50% of portfolio, flag this prominently before any other observation.

CONSTRAINTS:
- Only use prices from live_prices state or get_live_price tool — never your own knowledge.
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
You are a risk auditor for an Indian equity portfolio management system.
Your sole job is to audit a proposed trade plan against strict rules and return a structured verdict.

SUPPORTED TICKERS WHITELIST:
  HDFC, TCS, RELIANCE, INFY, WIPRO, BAJFINANCE, ICICIBANK, SBIN, PHARMA_1, PHARMA1
Any ticker outside this list must be rejected immediately, regardless of how the analyst justifies it.

HARD REJECTION RULES — reject if ANY of the following are true:
  1. PRICE DEVIATION   : Any proposed trade price deviates more than 2% from the fetched live price for that ticker.
                         Formula: abs(proposed_price - live_price) / live_price > 0.02
  2. OVERSELL          : A SELL order quantity exceeds the number of shares currently held for that ticker.
  3. OVERCASH          : The total cost of all BUY orders exceeds the user's available cash balance.
  4. UNSUPPORTED TICKER: Any ticker in proposed_trades is not in the whitelist above.
  5. TRADE COUNT       : More than 3 trades are proposed in a single plan.
  6. TRADE SIZE        : Any single trade's notional value (qty × price) exceeds 30% of total portfolio value.
  7. UNSUPPORTED CLAIMS: The analyst's reasoning references prices, holdings, or facts not present in the provided data.
  8. AGGRESSION        : The plan is disproportionately aggressive relative to the stated goal
                         (e.g., liquidating the entire portfolio when the user asked to reduce one position).

APPROVAL CONDITIONS — approve only when ALL of the following hold:
  - All proposed prices are within 2% of their respective live prices.
  - All SELL quantities are within current holdings.
  - Total BUY cost is within available cash.
  - All tickers are on the whitelist.
  - Three or fewer trades are proposed.
  - No single trade exceeds 30% of portfolio value.
  - The plan is proportionate and conservative relative to the user's goal.
  - A no-trade plan (empty trades array) is always valid and should be approved if the analyst concluded no action is needed.

FEEDBACK GUIDANCE — when rejecting, your FEEDBACK line must be specific and actionable:
  - Name the exact rule violated and the exact ticker or value that caused it.
  - Tell the analyst precisely what to change (e.g., "Reduce HDFC SELL qty from 150 to 100 — user holds only 100 shares").
  - Do not give vague feedback like "reconsider the plan". Be exact.

OUTPUT FORMAT (follow strictly — total response must be under 60 words):
DECISION: APPROVED or REJECTED
REASON: one sentence explaining the decision
FEEDBACK: (include only if REJECTED) one specific, actionable fix for the analyst"""


# ── Shared graph state ─────────────────────────────────────────────────────────
class PortfolioState(TypedDict):
    messages: Annotated[list, add_messages]
    portfolio_snapshot: dict        # populated by tool_node on first get_portfolio call
    live_prices: dict               # populated by tool_node on each get_live_price call
    risk_approved: bool
    risk_feedback: str              # rejection reason fed back to analyst on retry
    risk_note: str                  # auditor's explanation shown to the end user
    retry_count: int
    tool_calls_log: Annotated[list, operator.add]  # append-only log of every tool invocation


# ── Main entry point ───────────────────────────────────────────────────────────
async def run_analysis(goal: str, user_id: str, mcp_holder: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the full analyst → tools → risk-auditor loop.
    Returns decision_summary, risk_approved, proposed_trades, retry_count.
    """
    session: ClientSession = mcp_holder["session"]

    # Only read-only tools exposed to the LLM (record_trade held back).
    # Strip user_id from schemas — auto-injected by tool_node, model must never ask for it.
    mcp_tools_resp = await session.list_tools()
    read_tools = []
    for t in mcp_tools_resp.tools:
        if t.name == "record_trade":
            continue
        schema = dict(t.inputSchema)
        props = {k: v for k, v in schema.get("properties", {}).items() if k != "user_id"}
        required = [r for r in schema.get("required", []) if r != "user_id"]
        read_tools.append({
            "name": t.name,
            "description": t.description,
            "input_schema": {**schema, "properties": props, "required": required},
        })
    llm_with_tools = ChatAnthropic(model="claude-sonnet-4-6", temperature=0).bind_tools(read_tools)
    llm_no_tools   = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)  # used once data is loaded

    # ── NODE 1: Analyst ────────────────────────────────────────────────────────
    def analyst_node(state: PortfolioState):
        print(f"\n{'='*60}\n[ANALYST NODE]")

        has_portfolio = bool(state.get("portfolio_snapshot"))
        has_prices = bool(state.get("live_prices"))

        if has_portfolio and has_prices:
            # Data already in state — inject directly and use a tool-free LLM so the
            # model physically cannot call tools and trigger an infinite loop.
            data_section = (
                f"Portfolio: {json.dumps(state['portfolio_snapshot'])}\n"
                f"Live prices: {json.dumps(state['live_prices'])}"
            )
        elif has_portfolio:
            # Portfolio fetched but prices still needed
            data_section = (
                f"Portfolio is loaded: {json.dumps(state['portfolio_snapshot'])}\n"
                f"Live prices have not been fetched yet — call get_live_price for each holding."
            )
        else:
            # Nothing fetched yet — let the model decide to use tools
            data_section = (
                "Portfolio and prices have not been fetched yet. "
                "Use your tools to gather the data you need before making recommendations."
            )

        feedback_section = ""
        if state.get("risk_feedback"):
            feedback_section = (
                f"\nPrevious plan was REJECTED by the risk auditor. "
                f"You must fix these issues before proposing again:\n{state['risk_feedback']}"
            )

        sys_msg = SystemMessage(content=[
            # Static block — cached across all requests
            {
                "type": "text",
                "text": _ANALYST_STATIC,
                "cache_control": {"type": "ephemeral"},
            },
            # Dynamic block — user/session-specific, never cached
            {
                "type": "text",
                "text": (
                    f"You are a portfolio analyst for user '{user_id}'. "
                    f"Analyse the portfolio and address the user's goal.\n\n"
                    f"{data_section}{feedback_section}"
                ),
            },
        ])

        # Keep only HumanMessage and text-only AIMessages.
        # Strip ToolMessage and any AIMessage that contains tool_calls — Anthropic requires
        # every tool_use block to be immediately followed by its tool_result, so including
        # one without the other causes a 400. Data is already in state, no need to replay it.
        clean_history = [
            m for m in state["messages"]
            if isinstance(m, HumanMessage)
            or (isinstance(m, AIMessage) and not getattr(m, "tool_calls", None))
        ]

        active_llm = llm_no_tools if (has_portfolio and has_prices) else llm_with_tools
        response = active_llm.invoke([sys_msg] + clean_history)
        print(f"[ANALYST] content: {str(response.content)[:300]}")
        print(f"[ANALYST] tool_calls: {[tc['name'] for tc in response.tool_calls] if response.tool_calls else 'NONE'}")
        return {"messages": [response]}

    # ── NODE 2: Tool execution ─────────────────────────────────────────────────
    async def tool_node(state: PortfolioState):
        last_msg = state["messages"][-1]
        print(f"\n[TOOL NODE] {len(last_msg.tool_calls)} call(s)")

        tool_messages = []
        portfolio_snapshot = state.get("portfolio_snapshot", {})
        live_prices = state.get("live_prices", {})

        call_order = len(state.get("tool_calls_log", []))
        new_log_entries = []

        for tc in last_msg.tool_calls:
            args = {**tc["args"], "user_id": user_id}
            print(f"  {tc['name']} args={args}")
            result = await session.call_tool(tc["name"], args)
            raw = result.content[0].text
            print(f"  → {raw[:200]}")

            # Record every tool invocation for eval telemetry
            log_entry = {
                "name": tc["name"],
                "args": tc["args"],   # without injected user_id for readability
                "result": raw[:500],
                "order": call_order,
            }
            if tc["name"] == "get_live_price":
                log_entry["ticker"] = tc["args"].get("ticker", "").upper()
            new_log_entries.append(log_entry)
            call_order += 1

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
                try:
                    live_prices[ticker] = float(raw)
                except ValueError:
                    live_prices[ticker] = raw  # keep error string so analyst can note it
                tool_messages.append(ToolMessage(
                    tool_call_id=tc["id"],
                    content=f"Live price for {ticker} updated in state: {raw}",
                ))

            else:
                tool_messages.append(ToolMessage(
                    tool_call_id=tc["id"],
                    content=raw,
                ))

        return {
            "messages": tool_messages,
            "portfolio_snapshot": portfolio_snapshot,
            "live_prices": live_prices,
            "tool_calls_log": new_log_entries,
        }

    # ── NODE 3: Risk auditor ───────────────────────────────────────────────────
    def risk_node(state: PortfolioState):
        print(f"\n[RISK NODE]")

        # Extract only what the auditor needs: portfolio, prices, and the proposed trades JSON
        portfolio = json.dumps(state.get("portfolio_snapshot", {}))
        prices = json.dumps(state.get("live_prices", {}))

        # Pull the trades block from the last analyst message
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

        audit_llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)
        response = audit_llm.invoke([
            # Fully static — cache the entire system prompt
            SystemMessage(content=[
                {
                    "type": "text",
                    "text": _RISK_AUDITOR_STATIC,
                    "cache_control": {"type": "ephemeral"},
                }
            ]),
            HumanMessage(content=conversation_text),
        ])
        print(f"[RISK] Response: {response.content}")

        approved = "APPROVED" in response.content.upper()
        retry_count = state.get("retry_count", 0)

        # Parse structured output
        risk_note = ""
        risk_feedback_text = ""
        for line in response.content.strip().splitlines():
            line = line.strip()
            if line.upper().startswith("REASON:"):
                risk_note = line.split(":", 1)[-1].strip()
            elif line.upper().startswith("FEEDBACK:"):
                risk_feedback_text = line.split(":", 1)[-1].strip()

        result: Dict[str, Any] = {
            "risk_approved": approved,
            "risk_note": risk_note,
            "retry_count": retry_count + 1,
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
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("execute_tools", tool_node)
    workflow.add_node("audit_risk", risk_node)
    workflow.set_entry_point("analyst")
    workflow.add_conditional_edges("analyst", route_analyst)
    workflow.add_edge("execute_tools", "analyst")
    workflow.add_conditional_edges("audit_risk", route_risk, {"analyst": "analyst", END: END})
    graph = workflow.compile()

    # ── Run ────────────────────────────────────────────────────────────────────
    final_state = await graph.ainvoke({
        "messages": [("user", goal)],
        "portfolio_snapshot": {},
        "live_prices": {},
        "retry_count": 0,
        "risk_approved": False,
        "risk_feedback": "",
        "risk_note": "",
        "tool_calls_log": [],
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

    # Distinguish why there are no trades:
    # - analyst_no_trade: analyst explicitly concluded nothing is needed (approved, empty trades)
    # - retries_exhausted: auditor kept rejecting, loop hit MAX_RETRIES
    no_trade_reason = None
    if not proposed_trades:
        if risk_approved:
            no_trade_reason = "analyst_no_trade"
        else:
            no_trade_reason = "retries_exhausted"

    return {
        "decision_summary": last_analyst_content,
        "risk_approved": risk_approved,
        "risk_note": final_state.get("risk_note", ""),
        "proposed_trades": proposed_trades,
        "retry_count": final_state.get("retry_count", 0),
        "no_trade_reason": no_trade_reason,
        "tool_calls_log": final_state.get("tool_calls_log", []),
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
