# Portfolio Agent

A multi-agent AI system for Indian equity portfolio management. An LLM analyst proposes trades, a risk auditor reviews them, and a human approves or rejects before anything touches the portfolio.

---

## What It Does

- Fetches live NSE prices via yfinance (falls back to previous close when market is closed)
- Analyst (Claude Sonnet) calls MCP tools to read the portfolio, then proposes trades grounded in real data
- Risk auditor (Claude Haiku) enforces hard rules — rejects aggressive, oversized, or data-unsupported plans
- Human-in-the-loop: proposed trades are shown in the UI before any execution
- All sessions, trades, and portfolio state persisted in PostgreSQL

---

## Architecture

```
Browser (http://localhost:8000)
        │
        ▼
FastAPI (main.py)
   ├── POST /api/analyze  ──► LangGraph Agent (agent/graph.py)
   │                              ├── Analyst Node  (claude-sonnet-4-6)
   │                              │     └── calls MCP tools
   │                              ├── Tool Node
   │                              │     └── get_portfolio / get_live_price
   │                              └── Risk Auditor  (claude-haiku-4-5)
   │                                    └── APPROVED / REJECTED (max 3 retries)
   │
   ├── POST /api/execute  ──► MCP record_trade ──► PostgreSQL
   └── GET  /api/portfolio/{user_id}
```

```
User Goal
  │
  ▼
[Analyst] ──tool calls──► [Tool Node] ──► back to Analyst
  │ no tools needed
  ▼
[Risk Auditor]
  ├── APPROVED ──► trades saved ──► User approves/rejects in UI ──► /api/execute
  └── REJECTED ──► feedback injected ──► Analyst retries (max 3)
```

MCP server runs as a **subprocess** communicating over stdin/stdout. `record_trade` is hidden from the LLM — it can only be called via explicit user approval through `/api/execute`.

---

## Project Structure

```
portfolio_agenticAI/
├── main.py                     # FastAPI app, lifespan, MCP session init
├── agent/
│   ├── graph.py                # LangGraph workflow: analyst + tools + risk + routing
│   └── parsing.py              # Regex + Claude fallback for JSON trade extraction
├── api/
│   ├── routes.py               # /api/analyze, /api/execute, /api/portfolio handlers
│   └── schemas.py              # Pydantic request models
├── db/
│   ├── engine.py               # SQLAlchemy engine, SessionLocal, migrations
│   ├── models.py               # ORM: User, Portfolio, AnalysisSession, TradeHistory
│   └── crud.py                 # All DB operations (idempotent)
├── tools/
│   └── market_server_mcp.py    # MCP server — get_portfolio, get_live_price, record_trade
├── eval/
│   ├── test_cases.py           # 10 test cases with typed check primitives
│   ├── evaluator.py            # Check runner and result types
│   └── run_eval.py             # CLI: colored report + JSON export
├── static/
│   └── index.html              # Browser UI
├── docker-compose.yml          # PostgreSQL container
├── requirements.txt
└── .env                        # API keys and DB config (git-ignored)
```

---

## Supported Tickers

`HDFC` `TCS` `RELIANCE` `INFY` `WIPRO` `BAJFINANCE` `ICICIBANK` `SBIN` `PHARMA_1` (Sun Pharma)

---

## Setup & Deployment

### Prerequisites

- Python 3.11+
- Docker Desktop
- Anthropic API key
- LangSmith API key (free, for tracing)

### 1. Clone and install

```bash
git clone <repo>
cd portfolio_agenticAI
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure `.env`

```env
DATABASE_URL=postgresql://portfolio_user:portfolio_pass@localhost:5432/portfolio_db

ANTHROPIC_API_KEY=your_anthropic_api_key

LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=portfolio-agent
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

### 3. Start PostgreSQL

```bash
docker compose up -d
```

### 4. Start the server

```bash
uvicorn main:app --reload --port 8000
```

DB tables are created on first startup. First request for a new `user_id` seeds the portfolio:
- Holdings: HDFC ×100, RELIANCE ×50, TCS ×20
- Cash: ₹5,00,000

Open **http://localhost:8000** to use the UI.

---

## Running the Eval Suite

```bash
# Run all 10 tests against the live server
python -m eval.run_eval

# Run specific tests
python -m eval.run_eval --ids T001 T002 T009

# Save JSON report
python -m eval.run_eval --out eval/results.json
```

Tests cover: guardrails, concentration risk, feedback mode, buy/sell intent, rebalancing, price hallucination, cash flagging, invalid tickers, quantity validation.

Current score: **10/10 (100%)**

---

## API Reference

### `POST /api/analyze`

```json
{
  "user_id": "mallika_01",
  "session_id": "sess_abc123",
  "mode": "goal",
  "goal": "Suggest which stock to trim to reduce concentration risk."
}
```

Set `"mode": "feedback"` to run a full automated portfolio health review (no `goal` needed).

**Response:**
```json
{
  "session_id": "sess_abc123",
  "decision_summary": "HDFC is your largest position at 40% of equity...",
  "risk_approved": true,
  "retry_count": 1,
  "proposed_trades": [
    { "trade_id": 7, "ticker": "HDFC", "side": "SELL", "qty": 20, "proposed_price": 756.2, "accepted": null }
  ]
}
```

### `POST /api/execute`

```json
{ "user_id": "mallika_01", "session_id": "sess_abc123", "approved": true }
```

On approval: fetches live price per trade, updates portfolio, returns execution results.
On rejection: marks all trades `accepted=false`, no portfolio change.

### `GET /api/portfolio/{user_id}`

Returns current holdings with live prices and total portfolio value.

---

## Observability

**LangSmith** — https://smith.langchain.com → Projects → `portfolio-agent`
Full node-by-node trace, tool calls, token usage, retry loops.

**Swagger UI** — http://localhost:8000/docs

**Server logs** — analyst tool calls, risk decisions, MCP tool responses printed to stdout.

---

## Key Design Decisions

| Decision | Reason |
|---|---|
| `record_trade` hidden from LLM | Prevents autonomous execution — human approval is the only trigger |
| `user_id` injected at tool node | LLMs unreliable at passing IDs consistently; injection is deterministic |
| Two-phase pricing | `proposed_price` (analysis) vs `executed_price` (execution) captures real slippage |
| Retry loop with accumulated feedback | Risk rejection reasons injected back so analyst can revise the plan |
| MCP subprocess | Decouples tool implementation from agent; standard protocol |
| Sync SQLAlchemy | MCP subprocess cannot use async drivers; keeps DB layer uniform |

---

## TODO

- **Memory-based responses** — persist per-user conversation history and past analysis sessions using LangGraph's `MemorySaver` or a vector store, so the analyst can reference prior recommendations and portfolio evolution over time
- **RAGAS evaluation** — integrate [RAGAS](https://docs.ragas.io) metrics (faithfulness, answer relevance, context precision) to score analyst responses against retrieved portfolio data, replacing keyword-based summary checks with semantic evaluation
- **Agent workflow evaluation framework** — adopt a structured agentic eval framework (e.g. LangSmith evaluators or a custom harness) to score tool-call correctness, multi-step reasoning chains, and risk auditor decision quality across a larger synthetic portfolio dataset
