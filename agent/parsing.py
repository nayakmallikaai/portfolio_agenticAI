import json
import re
from typing import List, Dict, Tuple
from anthropic import Anthropic


def parse_proposed_trades(text: str) -> Tuple[List[Dict], bool]:
    """
    Try regex extraction of JSON trades block from analyst text.
    Returns (trades, block_found) — block_found=True even when trades list is empty,
    so the caller knows the model explicitly said no trades vs. no block at all.
    """
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
                print(f"[PARSE] Found trades block with {len(trades)} trade(s) via regex.")
                return trades, True
            except json.JSONDecodeError:
                continue
    return [], False


def extract_trades_via_llm(plan_text: str) -> List[Dict]:
    """Fallback: Claude API call to force-extract trades when no JSON block was found."""
    if not plan_text.strip():
        print("[PARSE] Empty text — skipping LLM extraction.")
        return []
    print("[PARSE] No trades block found — using Claude extraction fallback.")
    client = Anthropic()
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": (
                "Extract ALL proposed trades from the text below.\n"
                "Reply with ONLY a JSON object, no explanation, no markdown:\n"
                '{"trades": [{"ticker": "X", "side": "BUY or SELL", "qty": 10, "price": 1000.0}]}\n'
                "If no trades are mentioned, reply: {\"trades\": []}\n\n"
                f"TEXT:\n{plan_text}"
            ),
        }],
    )
    raw = response.content[0].text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    print(f"[PARSE] Claude extractor raw: {raw[:300]}")
    try:
        data = json.loads(raw)
        trades = data.get("trades", [])
        print(f"[PARSE] Claude extracted {len(trades)} trade(s).")
        return trades
    except json.JSONDecodeError as e:
        print(f"[PARSE] Claude extraction failed: {e}")
        return []
