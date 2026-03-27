import asyncio
import os
from typing import Annotated, TypedDict, List
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama
from langchain_core.messages import ToolMessage, SystemMessage, HumanMessage
from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters

# 1. Define the Shared State
MAX_RETRIES = 3

class PortfolioState(TypedDict):
    messages: Annotated[list, add_messages]
    risk_approval: bool
    final_plan: str
    retry_count: int

# 2. Setup MCP Parameters
server_params = StdioServerParameters(command="python", args=["tools/market_server_mcp.py"])

async def run_system():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # Discovery
            mcp_tools = await session.list_tools()
            tools = [{"name": t.name, "description": t.description, "parameters": t.inputSchema} 
                     for t in mcp_tools.tools]
        
            # Initialize Local LLM
            llm = ChatOllama(model="llama3.1", temperature=0).bind_tools(tools)

            # --- NODE 1: Senior Analyst (The Proposer) ---
            analyst_call_count = [0]
            def analyst_node(state: PortfolioState):
                analyst_call_count[0] += 1
                print(f"\n{'='*60}")
                print(f"[ANALYST NODE] Call #{analyst_call_count[0]}")
                print(f"[ANALYST] Total messages in state: {len(state['messages'])}")
                for i, m in enumerate(state["messages"]):
                    role = getattr(m, 'type', type(m).__name__)
                    content_preview = str(getattr(m, 'content', ''))[:120].replace('\n', ' ')
                    tool_calls = getattr(m, 'tool_calls', [])
                    print(f"  msg[{i}] type={role} | content={content_preview!r}")
                    if tool_calls:
                        print(f"         tool_calls={[tc['name'] for tc in tool_calls]}")
                print(f"[ANALYST] Invoking LLM with goal: {state['messages'][0].content!r}")
                sys_msg = SystemMessage(content="You are a Senior Investment Analyst. Use tools to check portfolio and prices, then propose a trade plan.")
                response = llm.invoke([sys_msg] + state["messages"])
                print(f"[ANALYST] LLM response type: {response.type}")
                print(f"[ANALYST] LLM response content: {str(response.content)[:300]!r}")
                print(f"[ANALYST] LLM tool_calls: {[tc['name'] for tc in response.tool_calls] if response.tool_calls else 'NONE'}")
                return {"messages": [response]}

            # --- NODE 2: Tool Execution (The Action Layer) ---
            async def tool_node(state: PortfolioState):
                last_msg = state["messages"][-1]
                print(f"\n{'='*60}")
                print(f"[TOOL NODE] Executing {len(last_msg.tool_calls)} tool call(s)")
                tool_results = []
                for tc in last_msg.tool_calls:
                    print(f"  Calling tool: {tc['name']} | args: {tc['args']}")
                    result = await session.call_tool(tc['name'], tc['args'])
                    result_text = result.content[0].text
                    print(f"  Result: {result_text[:200]!r}")
                    tool_results.append(ToolMessage(tool_call_id=tc['id'], content=result_text))
                return {"messages": tool_results}

            # --- NODE 3: Risk Auditor (The Guardrail) ---
            def risk_node(state: PortfolioState):
                print(f"\n{'='*60}")
                print(f"[RISK NODE] Auditing last 2 messages")
                last_two = state["messages"][-2:]
                for i, m in enumerate(last_two):
                    print(f"  auditing msg[{i}]: {str(getattr(m, 'content', ''))[:200]!r}")
                audit_llm = ChatOllama(model="llama3.1", temperature=0)
                # Extract plain text content from each message for the audit
                conversation_text = "\n".join(
                    f"{type(m).__name__}: {m.content}" for m in last_two if m.content
                )
                print(f"[RISK] Sending to audit LLM:\n{conversation_text[:500]}")
                audit_messages = [
                    SystemMessage(content="You are a Risk Manager. Review the following analyst output for trade safety. Reply with 'APPROVED' or 'REJECTED' plus a specific reason."),
                    HumanMessage(content=conversation_text),
                ]
                response = audit_llm.invoke(audit_messages)
                print(f"[RISK] Full audit response: {response.content}")
                approved = "APPROVED" in response.content.upper()
                retry_count = state.get("retry_count", 0)
                print(f"[RISK] Parsed approval: {approved} | retry_count: {retry_count}")

                result = {"risk_approval": approved, "retry_count": retry_count + 1}
                if not approved:
                    feedback = HumanMessage(
                        content=f"RISK MANAGER FEEDBACK (attempt {retry_count + 1}/{MAX_RETRIES}): Your previous plan was REJECTED.\nReason: {response.content}\nRevise your trade plan to address these concerns."
                    )
                    print(f"[RISK] Injecting feedback into messages for analyst retry.")
                    result["messages"] = [feedback]
                return result

            # --- ROUTING LOGIC ---
            def route_analyst(state: PortfolioState):
                last = state["messages"][-1]
                has_tool_calls = bool(getattr(last, 'tool_calls', []))
                decision = "execute_tools" if has_tool_calls else "audit_risk"
                print(f"\n[ROUTE_ANALYST] has_tool_calls={has_tool_calls} → routing to '{decision}'")
                return decision

            def route_risk(state: PortfolioState):
                approved = state.get("risk_approval", False)
                retry_count = state.get("retry_count", 0)
                if approved:
                    print(f"\n[ROUTE_RISK] APPROVED → routing to END")
                    return END
                if retry_count >= MAX_RETRIES:
                    print(f"\n[ROUTE_RISK] REJECTED after {MAX_RETRIES} attempts → routing to END (hard limit reached)")
                    return END
                print(f"\n[ROUTE_RISK] REJECTED (attempt {retry_count}/{MAX_RETRIES}) → routing back to analyst")
                return "analyst"

            # --- GRAPH ASSEMBLY ---
            workflow = StateGraph(PortfolioState)
            workflow.add_node("analyst", analyst_node)
            workflow.add_node("execute_tools", tool_node)
            workflow.add_node("audit_risk", risk_node)

            workflow.set_entry_point("analyst")
            workflow.add_conditional_edges("analyst", route_analyst)
            workflow.add_edge("execute_tools", "analyst")
            workflow.add_conditional_edges("audit_risk", route_risk, {"analyst": "analyst", END: END})

            app = workflow.compile()

            # Execute
            #goal = "I want to sell half my HDFC and buy RELIANCE. Check if this is a good move and execute."
            goal = "I want to rebalance. Check my holdings and prices, then tell me the exact trades to make each stock equal weight."
            print(f"\n{'='*60}")
            print(f"[START] Goal: {goal!r}")
            print(f"[START] MCP tools available: {[t['name'] for t in tools]}")
            print(f"{'='*60}\n")
            async for output in app.astream({"messages": [("user", goal)], "retry_count": 0}):
                node_name = list(output.keys())[0]
                print(f"\n>>> Finished node: {node_name}")
            print(f"\n{'='*60}")
            print("[DONE] Graph execution complete.")

if __name__ == "__main__":
    asyncio.run(run_system())