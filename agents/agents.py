import os
import json
from anthropic import AsyncAnthropic
from core.schema import AnalystOutput, RiskReview, Trade

def get_client():
    return AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

async def call_analyst(portfolio, goal):
    system_prompt = f"""You are a Senior Investment Analyst. 
    Analyze the portfolio and suggest trades. 
    Return ONLY a JSON object matching this schema: {AnalystOutput.model_json_schema()}"""
    
    response = await get_client().messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1500,
        system=system_prompt,
        messages=[{"role": "user", "content": f"Portfolio: {portfolio}\nGoal: {goal}"}]
    )
    return AnalystOutput.model_validate_json(response.content[0].text)

async def call_risk_manager(proposal):
    system_prompt = f"""You are a Risk Manager. Review the following trades for safety.
    Return ONLY a JSON object matching this schema: {RiskReview.model_json_schema()}"""
    
    response = await get_client().messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1000,
        system=system_prompt,
        messages=[{"role": "user", "content": f"Review these trades: {proposal.model_dump_json()}"}]
    )
    return RiskReview.model_validate_json(response.content[0].text)