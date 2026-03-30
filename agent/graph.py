"""
LangGraph multi-agent workflow.
All LLM and graph logic lives here; no HTTP or DB imports.
"""
import json
import re
from typing import Annotated, TypedDict, Dict, Any, Optional

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import ToolMessage, SystemMessage, HumanMessage, AIMessage
from mcp import ClientSession

from agent.parsing import parse_proposed_trades, extract_trades_via_llm

MAX_RETRIES = 3


# ── Shared graph state ─────────────────────────────────────────────────────────
class PortfolioState(TypedDict):
    messages: Annotated[list, add_messages]
    portfolio_snapshot: dict        # populated by tool_node on first get_portfolio call
    live_prices: dict               # populated by tool_node on each get_live_price call
    risk_approved: bool
    risk_feedback: str              # rejection reason fed back to analyst on retry
    risk_note: str                  # auditor's explanation shown to the end user
    retry_count: int


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

        sys_msg = SystemMessage(content=
            f"""You are a portfolio analyst for user '{user_id}'. Analyse the portfolio and address the user's goal.

            {data_section}{feedback_section}

            ANALYSIS APPROACH:
            - Read the goal first. Only act if the portfolio genuinely needs to change.
            - Propose nothing if the goal is already met.
            - When trades are needed, keep them small and data-grounded.

            CONSTRAINTS:
            - Only use prices from live_prices state or get_live_price tool — never your own knowledge.
            - Only trade these tickers: HDFC, TCS, RELIANCE, INFY, WIPRO, BAJFINANCE, ICICIBANK, SBIN, PHARMA_1, PHARMA1.
            - Never sell more shares than held. Never buy more than cash allows.
            - Do not predict future prices or market direction.
            - If a price is unavailable, note it and proceed with remaining data.
            - If asked something unrelated to portfolio management, respond:
              "I can only help with portfolio management goals."

            RESPONSE FORMAT — NON NEGOTIABLE:
            - Plain conversational text only. No markdown, no tables, no headers, no emojis, no bold.
            - Maximum 80 words across a maximum of 2 paragraphs.
            - End every response with this exact JSON block (no text after it):
            {{"trades": [{{"ticker": "X", "side": "BUY/SELL", "qty": 0, "price": 0.0}}]}}
            If no trades: {{"trades": []}}""")

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

        for tc in last_msg.tool_calls:
            args = {**tc["args"], "user_id": user_id}
            print(f"  {tc['name']} args={args}")
            result = await session.call_tool(tc["name"], args)
            raw = result.content[0].text
            print(f"  → {raw[:200]}")

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
            SystemMessage(content="""You are a risk auditor for an Indian stock portfolio.

            REJECT if ANY rule is violated:
            - Price deviates >2% from live price
            - SELL qty exceeds current holdings
            - BUY cost exceeds available cash
            - Ticker not in: HDFC, TCS, RELIANCE, INFY, WIPRO, BAJFINANCE, ICICIBANK, SBIN, PHARMA_1, PHARMA1
            - More than 3 trades proposed
            - Single trade exceeds 30% of total portfolio value
            - Claims unsupported by the data
            - Plan is aggressive relative to the goal

            APPROVE if: conservative, data-grounded, proportionate. No-trade plans are valid.

            OUTPUT FORMAT (strictly follow this, total response under 50 words):
            DECISION: APPROVED or REJECTED
            REASON: one sentence explaining why
            FEEDBACK: (only if REJECTED) one specific fix the analyst must make"""),
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
