import json
import re
from typing import List, Dict
from langchain_ollama import ChatOllama


def parse_proposed_trades(text: str) -> List[Dict]:
    """Try regex extraction of JSON trades block from analyst text."""
    patterns = [
        r"```json\s*(\{.*?\})\s*```",
        r"```\s*(\{.*?\})\s*```",
        r'(\{\s*"trades"\s*:\s*\[.*?\]\s*\})',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
                trades = data.get("trades", [])
                if trades:
                    print(f"[PARSE] Found {len(trades)} trade(s) via regex.")
                    return trades
            except json.JSONDecodeError:
                continue
    return []


def extract_trades_via_llm(plan_text: str) -> List[Dict]:
    """Fallback: dedicated LLM call to force-extract trades as strict JSON."""
    print("[PARSE] Regex failed — using LLM extraction fallback.")
    extractor = ChatOllama(model="llama3.1", temperature=0)
    prompt = (
        "Extract ALL proposed trades from the text below.\n"
        "Reply with ONLY a JSON object, no explanation, no markdown fences:\n"
        '{"trades": [{"ticker": "X", "side": "BUY or SELL", "qty": 10, "price": 1000.0}]}\n\n'
        f"TEXT:\n{plan_text}"
    )
    response = extractor.invoke(prompt)
    raw = response.content.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    print(f"[PARSE] LLM extractor raw: {raw[:300]}")
    try:
        data = json.loads(raw)
        trades = data.get("trades", [])
        print(f"[PARSE] LLM extracted {len(trades)} trade(s).")
        return trades
    except json.JSONDecodeError as e:
        print(f"[PARSE] LLM extraction failed: {e}")
        return []
