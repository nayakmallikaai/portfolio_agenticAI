# Entry point
import asyncio
from dotenv import load_dotenv
from agents.agents import call_analyst, call_risk_manager

load_dotenv()

async def run_portfolio_engine():
    # Simulated State
    my_portfolio = {"RELIANCE": 100, "CASH": 500000}
    user_goal = "I want to diversify into the Banking sector while keeping 50% in Energy."

    print("🚀 Phase 1: Analyst is reasoning...")
    proposal = await call_analyst(my_portfolio, user_goal)
    
    for trade in proposal.proposed_trades:
        print(f"Proposed: {trade.action} {trade.quantity} {trade.ticker}")

    print("\n🛡️ Phase 2: Risk Manager is auditing...")
    review = await call_risk_manager(proposal)
    
    print(f"Risk Score: {review.risk_score}")
    if review.approved:
        print("✅ STRATEGY APPROVED: " + review.comments)
    else:
        print("❌ STRATEGY REJECTED: " + review.comments)

asyncio.run(run_portfolio_engine()) 