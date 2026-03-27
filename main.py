import json
import re
from contextlib import asynccontextmanager
from typing import Annotated, TypedDict, Dict, Any, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama
from langchain_core.messages import ToolMessage, SystemMessage, HumanMessage

from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters

# ── Constants ──────────────────────────────────────────────────────────────────
MAX_RETRIES = 3
server_params = StdioServerParameters(command="python", args=["tools/market_server_mcp.py"])

# ── Shared state ───────────────────────────────────────────────────────────────
mcp_holder: Dict[str, Any] = {}
sessions: Dict[str, Dict[str, Any]] = {}  # session_id → session data


# ── LangGraph state ────────────────────────────────────────────────────────────
class PortfolioState(TypedDict):
    messages: Annotated[list, add_messages]
    risk_approval: bool
    final_plan: str
    retry_count: int


# ── Pydantic request models ────────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    user_id: str
    session_id: str
    goal: str

class ExecuteRequest(BaseModel):
    user_id: str
    session_id: str
    approved: bool


# ── FastAPI lifespan: keep one MCP session alive ───────────────────────────────
@asynccontextmanager
async def lifespan(_app: FastAPI):
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            mcp_holder["session"] = session
            print("[STARTUP] MCP session ready.")
            yield
    mcp_holder.clear()
    print("[SHUTDOWN] MCP session closed.")


app = FastAPI(title="Portfolio Agent API", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open("static/index.html") as f:
        return f.read()


# ── Helpers ────────────────────────────────────────────────────────────────────
def parse_proposed_trades(text: str) -> List[Dict]:
    """Try to extract a trades JSON block from analyst text. Returns [] if nothing found."""
    # Match ```json ... ``` or just a bare { "trades": [...] }
    patterns = [
        r"```json\s*(\{.*?\})\s*```",   # fenced code block
        r"```\s*(\{.*?\})\s*```",        # fenced block without language tag
        r'(\{\s*"trades"\s*:\s*\[.*?\]\s*\})',  # bare JSON object
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
                trades = data.get("trades", [])
                if trades:
                    print(f"[PARSE] Found {len(trades)} trade(s) via pattern: {pattern[:30]}")
                    return trades
            except json.JSONDecodeError:
                continue
    return []


def extract_trades_via_llm(plan_text: str) -> List[Dict]:
    """Fallback: ask a fresh LLM to extract trades from the plan as strict JSON."""
    print("[PARSE] Regex parse failed — falling back to LLM extraction.")
    extractor = ChatOllama(model="llama3.1", temperature=0)
    prompt = (
        "Extract ALL proposed trades from the text below.\n"
        "Reply with ONLY a JSON object, no explanation, no markdown:\n"
        '{"trades": [{"ticker": "X", "side": "BUY or SELL", "qty": 10, "price": 1000.0}]}\n\n'
        f"TEXT:\n{plan_text}"
    )
    response = extractor.invoke(prompt)
    raw = response.content.strip()
    print(f"[PARSE] LLM extractor raw output: {raw[:300]}")
    # Strip any accidental markdown fences the model adds
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        data = json.loads(raw)
        trades = data.get("trades", [])
        print(f"[PARSE] LLM extracted {len(trades)} trade(s)")
        return trades
    except json.JSONDecodeError as e:
        print(f"[PARSE] LLM extraction JSON parse error: {e}")
        return []


async def run_analysis(goal: str) -> Dict[str, Any]:
    """Run the full multi-agent loop (read-only tools) and return the result."""
    session: ClientSession = mcp_holder["session"]

    # Only expose read-only tools to the analyst — record_trade is held back
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
            "You are a Senior Investment Analyst. Use get_portfolio and get_live_price to check data, "
            "then propose a trade plan. At the end of your response output proposed trades as this exact JSON block:\n"
            "```json\n{\"trades\": [{\"ticker\": \"X\", \"side\": \"BUY or SELL\", \"qty\": 10, \"price\": 1000.0}]}\n```"
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
            print(f"  {tc['name']} args={tc['args']}")
            result = await session.call_tool(tc["name"], tc["args"])
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
        last = state["messages"][-1]
        has_tools = bool(getattr(last, "tool_calls", []))
        print(f"\n[ROUTE_ANALYST] → {'execute_tools' if has_tools else 'audit_risk'}")
        return "execute_tools" if has_tools else "audit_risk"

    def route_risk(state: PortfolioState):
        if state.get("risk_approval", False):
            print("[ROUTE_RISK] APPROVED → END")
            return END
        if state.get("retry_count", 0) >= MAX_RETRIES:
            print(f"[ROUTE_RISK] Hard limit {MAX_RETRIES} reached → END")
            return END
        print(f"[ROUTE_RISK] REJECTED (attempt {state.get('retry_count',0)}) → analyst retry")
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

    final_state = await graph.ainvoke({"messages": [("user", goal)], "retry_count": 0})

    # Pull last AI message that has content (the final plan)
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


# ── API 1: Analyze ─────────────────────────────────────────────────────────────
@app.post("/api/analyze")
async def analyze(req: AnalyzeRequest):
    """Run the multi-agent loop and return the decision summary."""
    result = await run_analysis(req.goal)
    sessions[req.session_id] = {
        "user_id": req.user_id,
        "goal": req.goal,
        "executed": False,
        **result,
    }
    return {
        "session_id": req.session_id,
        "decision_summary": result["decision_summary"],
        "risk_approved": result["risk_approved"],
        "proposed_trades": result["proposed_trades"],
        "retry_count": result["retry_count"],
    }


# ── API 2: Manual approval / execute ──────────────────────────────────────────
@app.post("/api/execute")
async def execute(req: ExecuteRequest):
    """User manually approves or rejects. On approval, execute trades via MCP."""
    data = sessions.get(req.session_id)
    if not data:
        raise HTTPException(status_code=404, detail="Session not found.")
    if data.get("executed"):
        raise HTTPException(status_code=400, detail="Trades already executed for this session.")

    sessions[req.session_id]["executed"] = True

    if not req.approved:
        return {"status": "rejected", "message": "Trade plan rejected by user. Portfolio unchanged."}

    trades: List[Dict] = data.get("proposed_trades", [])
    if not trades:
        raise HTTPException(
            status_code=400,
            detail="No structured trades found in the plan. Cannot execute automatically."
        )

    mcp_session: ClientSession = mcp_holder["session"]
    trade_results = []
    for trade in trades:
        result = await mcp_session.call_tool("record_trade", {
            "ticker": trade["ticker"],
            "side": trade["side"],
            "qty": int(trade["qty"]),
            "price": float(trade["price"]),
        })
        trade_results.append({"trade": trade, "result": result.content[0].text})

    portfolio_result = await mcp_session.call_tool("get_portfolio", {})
    updated_portfolio = json.loads(portfolio_result.content[0].text)

    return {
        "status": "executed",
        "trade_results": trade_results,
        "updated_portfolio": updated_portfolio,
    }
