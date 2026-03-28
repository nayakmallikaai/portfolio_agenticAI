"""
LangGraph multi-agent workflow.
All LLM and graph logic lives here; no HTTP or DB imports.
"""
import asyncio
from typing import Annotated, TypedDict, Dict, Any, List

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import ToolMessage, SystemMessage, HumanMessage
from mcp import ClientSession

from agent.parsing import parse_proposed_trades, extract_trades_via_llm

MAX_RETRIES = 3


# ── Shared graph state ─────────────────────────────────────────────────────────
class PortfolioState(TypedDict):
    messages: Annotated[list, add_messages]
    risk_approval: bool
    final_plan: str
    retry_count: int
    rejection_list: list


# ── Main entry point ───────────────────────────────────────────────────────────
async def run_analysis(goal: str, user_id: str, mcp_holder: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the full analyst → tools → risk-auditor loop.
    Returns decision_summary, risk_approved, proposed_trades, retry_count.
    """
    session: ClientSession = mcp_holder["session"]

    # Only read-only tools exposed to the LLM (record_trade held back)
    # Strip user_id from schemas — it is auto-injected by the tool node, model must never ask for it
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
    llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0).bind_tools(read_tools)

    # ── NODE 1: Analyst ────────────────────────────────────────────────────────
    def analyst_node(state: PortfolioState):
        print(f"\n{'='*60}\n[ANALYST NODE]")
        sys_msg = SystemMessage(content=
            f"""You are a portfolio analyst for user '{user_id}'. Analyse the portfolio and address the user's goal.

            Use your tools to fetch current holdings and live prices, then reason from that data.

            ANALYSIS APPROACH:
            - Understand what the user is actually asking for before proposing anything.
            - If the portfolio already satisfies the goal, say so clearly and propose nothing.
            - Only propose trades when there is a genuine, specific reason grounded in the data.
            - Prefer small, targeted adjustments over sweeping changes.

            CONSTRAINTS:
            - Only use prices from get_live_price — never your own knowledge.
            - Only trade these tickers: HDFC, TCS, RELIANCE, INFY, WIPRO, BAJFINANCE, ICICIBANK, SBIN, PHARMA_1, PHARMA1.
            - Never sell more shares than currently held. Never buy more than cash allows.
            - Do not predict future prices or market direction.
            - If a price fetch fails for one or more tickers, proceed with available data and note the missing tickers in your response.
            - If asked something unrelated to portfolio management, respond:
              "I can only help with portfolio management goals."

            RESPONSE FORMAT — NON NEGOTIABLE:
            - Plain conversational text only. No markdown, no tables, no headers, no emojis, no bold.
            - Maximum 80 words across a maximum of 2 paragraphs.
            - End every response with this exact JSON block (no text after it):
            {{"trades": [{{"ticker": "X", "side": "BUY/SELL", "qty": 0, "price": 0.0}}]}}
            If no trades: {{"trades": []}}""")
        response = llm.invoke([sys_msg] + state["messages"])
        print(f"[ANALYST] content: {str(response.content)[:300]}")
        print(f"[ANALYST] tool_calls: {[tc['name'] for tc in response.tool_calls] if response.tool_calls else 'NONE'}")
        return {"messages": [response]}

    # ── NODE 2: Tool execution ─────────────────────────────────────────────────
    async def tool_node(state: PortfolioState):
        last_msg = state["messages"][-1]
        print(f"\n[TOOL NODE] {len(last_msg.tool_calls)} call(s)")
        results = []
        for tc in last_msg.tool_calls:
            # Inject user_id so the LLM doesn't have to remember to include it
            args = {**tc["args"], "user_id": user_id}
            print(f"  {tc['name']} args={args}")
            result = await session.call_tool(tc["name"], args)
            text = result.content[0].text
            print(f"  → {text[:200]}")
            results.append(ToolMessage(tool_call_id=tc["id"], content=text))
        return {"messages": results}

    # ── NODE 3: Risk auditor ───────────────────────────────────────────────────
    def risk_node(state: PortfolioState):
        print(f"\n[RISK NODE]")
        last_two = state["messages"][-2:]
        conversation_text = "\n".join(
            f"{type(m).__name__}: {m.content}" for m in last_two if m.content
        )
        print(f"[RISK] Sending:\n{conversation_text[:400]}")
        audit_llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0)
        response = audit_llm.invoke([
            SystemMessage(content="""You are a risk auditor for an Indian stock portfolio.

            REJECT the plan if ANY of the following are true:
            - A recommended price deviates more than 2% from the live price provided
            - A SELL quantity exceeds the user's current holdings for that ticker
            - A BUY quantity would require more cash than the user has available
            - A trade involves a ticker outside the supported list: HDFC, TCS, RELIANCE, INFY, WIPRO, BAJFINANCE, ICICIBANK, SBIN, PHARMA_1, PHARMA1
            - The plan proposes more than 3 trades at once (no sweeping portfolio overhauls)
            - Any single trade exceeds 30% of total portfolio value
            - The analyst made factual claims not supported by the portfolio or price data
            - The plan is unnecessarily aggressive given the user's stated goal

            APPROVE if the plan is conservative, data-grounded, proportionate to the goal,
            and each proposed trade has a clear stated reason.
            A plan with no trades is also valid — APPROVE it if the analyst correctly
            determined no action was needed.

            Start your response with exactly APPROVED or REJECTED, then explain why briefly."""),
            HumanMessage(content=conversation_text),
        ])
        print(f"[RISK] Response: {response.content}")
        approved = "APPROVED" in response.content.upper()
        retry_count = state.get("retry_count", 0)
        # Get accumulated errors from previous retries
        previous_errors = state.get("rejection_list", [])
        result: Dict[str, Any] = {"risk_approval": approved, "retry_count": retry_count + 1}
        print(f"[RISK] Response: {response.content}")

        if not approved:
            # Extract clean reasons from this rejection
            new_reasons = extract_rejection_reasons(response.content)
            
            # Accumulate across retries
            all_reasons = previous_errors + new_reasons
            
            # Store for next retry
            result["rejection_list"] = all_reasons
            
            # Build clean feedback message
            reasons_text = "\n".join(f"- {r}" for r in all_reasons)
            
            result["messages"] = [HumanMessage(
                content=(
                    f"RISK MANAGER FEEDBACK "
                    f"(attempt {retry_count + 1}/{MAX_RETRIES}): REJECTED.\n\n"
                    f"You must fix ALL of these issues:\n"
                    f"{reasons_text}\n\n"
                    f"Revise your trade plan addressing each point above."
                )
            )]
        
        return result

    # ── Routing ────────────────────────────────────────────────────────────────
    def route_analyst(state: PortfolioState):
        has_tools = bool(getattr(state["messages"][-1], "tool_calls", []))
        print(f"\n[ROUTE] analyst → {'execute_tools' if has_tools else 'audit_risk'}")
        return "execute_tools" if has_tools else "audit_risk"

    def route_risk(state: PortfolioState):
        if state.get("risk_approval", False):
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
    final_state = await graph.ainvoke({"messages": [("user", goal)], "retry_count": 0})

    last_analyst_content = ""
    for msg in reversed(final_state["messages"]):
        if getattr(msg, "type", None) == "ai" and isinstance(msg.content, str) and msg.content.strip():
            last_analyst_content = msg.content
            break

    proposed_trades, block_found = parse_proposed_trades(last_analyst_content)
    if not block_found:
        proposed_trades = extract_trades_via_llm(last_analyst_content)

    return {
        "decision_summary": last_analyst_content,
        "risk_approved": final_state.get("risk_approval", False),
        "proposed_trades": proposed_trades,
        "retry_count": final_state.get("retry_count", 0),
    }

# ── Helper methods ─────────────────────────────────────────────────────────
def extract_rejection_reasons(risk_response: str) -> list[str]:
    lines = risk_response.strip().split("\n")
    reasons = []
    
    for line in lines:
        line = line.strip()
        # Skip the REJECTED line itself
        if not line or line.upper().startswith("REJECTED"):
            continue
        # Skip very short lines
        if len(line) < 10:
            continue
        reasons.append(line)
    
    # Return max 3 most important reasons
    return reasons[:3] 