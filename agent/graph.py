"""
LangGraph multi-agent workflow.
All LLM and graph logic lives here; no HTTP or DB imports.
"""
import asyncio
from typing import Annotated, TypedDict, Dict, Any, List

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama
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


# ── Main entry point ───────────────────────────────────────────────────────────
async def run_analysis(goal: str, user_id: str, mcp_holder: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the full analyst → tools → risk-auditor loop.
    Returns decision_summary, risk_approved, proposed_trades, retry_count.
    """
    session: ClientSession = mcp_holder["session"]

    # Only read-only tools exposed to the LLM (record_trade held back)
    mcp_tools_resp = await session.list_tools()
    read_tools = [
        {"name": t.name, "description": t.description, "parameters": t.inputSchema}
        for t in mcp_tools_resp.tools
        if t.name != "record_trade"
    ]
    llm = ChatOllama(model="llama3.1", temperature=0).bind_tools(read_tools)

    # ── NODE 1: Analyst ────────────────────────────────────────────────────────
    def analyst_node(state: PortfolioState):
        print(f"\n{'='*60}\n[ANALYST NODE]")
        sys_msg = SystemMessage(content=(
            "You are a Senior Investment Analyst. "
            "Use get_portfolio and get_live_price to retrieve live data, then propose a trade plan. "
            "When calling get_portfolio, pass user_id as an argument. "
            "At the end of your response output proposed trades in this exact JSON block:\n"
            "```json\n"
            '{"trades": [{"ticker": "X", "side": "BUY or SELL", "qty": 10, "price": 1000.0}]}\n'
            "```"
        ))
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
        audit_llm = ChatOllama(model="llama3.1", temperature=0)
        response = audit_llm.invoke([
            SystemMessage(content="You are a Risk Manager. Review the analyst's trade proposal. Reply with 'APPROVED' or 'REJECTED' plus a specific reason."),
            HumanMessage(content=conversation_text),
        ])
        print(f"[RISK] Response: {response.content}")
        approved = "APPROVED" in response.content.upper()
        retry_count = state.get("retry_count", 0)
        result: Dict[str, Any] = {"risk_approval": approved, "retry_count": retry_count + 1}
        if not approved:
            result["messages"] = [HumanMessage(
                content=(
                    f"RISK MANAGER FEEDBACK (attempt {retry_count + 1}/{MAX_RETRIES}): REJECTED.\n"
                    f"Reason: {response.content}\n"
                    f"Revise your trade plan to address these concerns."
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
        if getattr(msg, "type", None) == "ai" and msg.content:
            last_analyst_content = msg.content
            break

    proposed_trades = parse_proposed_trades(last_analyst_content)
    if not proposed_trades:
        proposed_trades = extract_trades_via_llm(last_analyst_content)

    return {
        "decision_summary": last_analyst_content,
        "risk_approved": final_state.get("risk_approval", False),
        "proposed_trades": proposed_trades,
        "retry_count": final_state.get("retry_count", 0),
    }
