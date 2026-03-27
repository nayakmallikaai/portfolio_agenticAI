# Portfolio Agent — Agentic AI Trading System

A multi-agent portfolio management system built with **LangGraph**, **FastAPI**, **MCP (Model Context Protocol)**, and **PostgreSQL**. An LLM-powered analyst proposes trades, a risk auditor reviews them, and a human approves or rejects before anything touches the portfolio.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Browser (UI)                         │
│          http://localhost:8000                          │
└────────────────────┬────────────────────────────────────┘
                     │ HTTP
┌────────────────────▼────────────────────────────────────┐
│                FastAPI (main.py)                        │
│   POST /api/analyze       POST /api/execute             │
│   api/routes.py           api/routes.py                 │
└──────────┬──────────────────────────┬───────────────────┘
           │ run_analysis()           │ call_tool()
┌──────────▼──────────┐   ┌──────────▼──────────────────┐
│  LangGraph Agent    │   │  MCP Server (subprocess)    │
│  agent/graph.py     │   │  tools/market_server_mcp.py │
│                     │   │                             │
│  ┌─────────────┐    │   │  get_portfolio(user_id)     │
│  │  Analyst    │◄───┼───┤  get_live_price(ticker)     │
│  │  (llama3.1) │    │   │  record_trade(...)          │
│  └──────┬──────┘    │   └──────────────┬──────────────┘
│         │           │                  │
│  ┌──────▼──────┐    │   ┌──────────────▼──────────────┐
│  │ Risk Auditor│    │   │     PostgreSQL (Docker)     │
│  │ (llama3.1)  │    │   │                             │
│  └──────┬──────┘    │   │  users                      │
│         │           │   │  portfolios                 │
│      retry          │   │  analysis_sessions          │
│      (max 3)        │   │  trade_history              │
└─────────────────────┘   └─────────────────────────────┘
```

### Agent Flow

```
User Goal
    │
    ▼
[Analyst Node] ──── calls get_portfolio, get_live_price via MCP
    │
  tool calls? ──yes──► [Tool Node] ──► back to Analyst
    │ no
    ▼
[Risk Auditor] ──── reviews proposed plan
    │
  APPROVED ──────────────────────────────► END
    │                                        │
  REJECTED + retries < 3                     │
    │                                    proposed trades
  feedback injected ──► back to Analyst  saved to DB
                                             │
                                         User sees UI
                                             │
                                      Approve / Reject
                                             │
                                  /api/execute fetches live price
                                             │
                                  record_trade updates portfolio
```

---

## Project Structure

```
portfolio_agenticAI/
│
├── main.py                     # FastAPI app, lifespan, DB init, MCP session
│
├── agent/
│   ├── graph.py                # LangGraph workflow — analyst + risk + routing
│   └── parsing.py              # Regex + LLM fallback trade extraction
│
├── api/
│   ├── routes.py               # /api/analyze and /api/execute handlers
│   └── schemas.py              # Pydantic request models
│
├── db/
│   ├── engine.py               # SQLAlchemy engine + SessionLocal
│   ├── models.py               # ORM models: User, Portfolio, AnalysisSession, TradeHistory
│   └── crud.py                 # All DB operations (idempotent)
│
├── tools/
│   └── market_server_mcp.py    # MCP server — runs as subprocess, 3 tools
│
├── static/
│   └── index.html              # Browser UI
│
├── docker-compose.yml          # PostgreSQL container
├── requirements.txt
└── .env                        # API keys and config (git ignored)
```

---

## Database Schema

### `users`
| Column | Type | Description |
|---|---|---|
| user_id | String PK | Unique user identifier |
| cash | Float | Available cash balance (seeded at 500,000) |
| created_at | DateTime | Account creation time |

### `portfolios`
| Column | Type | Description |
|---|---|---|
| id | Integer PK | Auto-increment |
| user_id | FK → users | Owner |
| ticker | String(20) | Stock ticker (e.g. HDFC, RELIANCE) |
| quantity | Integer | Current holding quantity |
| updated_at | DateTime | Last trade time |

Unique constraint: `(user_id, ticker)`

### `analysis_sessions`
| Column | Type | Description |
|---|---|---|
| session_id | String PK | Client-generated session ID |
| user_id | FK → users | Owner |
| goal | Text | User's original goal string |
| decision_summary | Text | Analyst's final recommendation |
| risk_approved | Boolean | Final risk auditor decision |
| proposed_trades | JSON | List of proposed trade dicts |
| retry_count | Integer | Number of risk retry loops used (max 3) |
| executed | Boolean | Whether user has acted on this session |
| created_at | DateTime | Analysis time |

### `trade_history`
| Column | Type | Description |
|---|---|---|
| id | Integer PK | Unique trade row ID returned to UI |
| session_id | FK → analysis_sessions | Linked session |
| user_id | FK → users | Owner |
| ticker | String(20) | Stock ticker |
| side | String(4) | BUY or SELL |
| qty | Integer | Quantity |
| proposed_price | Float (nullable) | LLM estimated price at analysis time |
| executed_price | Float (nullable) | Real-time yfinance price at execution |
| total_value | Float (nullable) | executed_price x qty, set at execution |
| proposed | Boolean | Always true |
| accepted | Boolean (nullable) | null=pending, true=accepted, false=rejected |
| created_at | DateTime | Row creation time |
| executed_at | DateTime (nullable) | Execution timestamp |

Unique constraint: `(session_id, ticker, side)` — idempotency key

**Trade lifecycle:**

| Phase | `accepted` | `proposed_price` | `executed_price` |
|---|---|---|---|
| After `/api/analyze` | `null` (pending) | LLM estimate | — |
| After `/api/execute` approve | `true` | LLM estimate | live yfinance price |
| After `/api/execute` reject | `false` | LLM estimate | — |

---

## MCP Server Tools

The MCP server runs as a **separate subprocess** and communicates with the FastAPI app over stdin/stdout pipes. Each tool call opens its own DB session.

| Tool | Exposed to LLM | Description |
|---|---|---|
| `get_portfolio(user_id)` | Yes | Fetches holdings + cash from DB |
| `get_live_price(ticker, user_id)` | Yes | Real-time NSE price via yfinance; falls back to hardcoded prices if market is closed |
| `record_trade(user_id, session_id, ticker, side, qty, price)` | No | Updates portfolio + DB; only called by `/api/execute` after user approval |

Supported NSE tickers: `HDFC`, `RELIANCE`, `TCS`, `INFY`, `WIPRO`, `BAJFINANCE`, `ICICIBANK`, `SBIN`

---

## API Reference

### `POST /api/analyze`

Runs the multi-agent loop. Saves proposed trades immediately to `trade_history` with `accepted=null`. Returns a `trade_id` per trade for tracking.

**Request:**
```json
{
  "user_id": "mallika_01",
  "session_id": "sess_abc123",
  "goal": "Check my holdings and prices, then tell me which stock to trim."
}
```

**Response:**
```json
{
  "session_id": "sess_abc123",
  "decision_summary": "Based on your portfolio...",
  "risk_approved": true,
  "retry_count": 1,
  "proposed_trades": [
    {
      "trade_id": 7,
      "ticker": "HDFC",
      "side": "SELL",
      "qty": 20,
      "proposed_price": 165000.0,
      "accepted": null
    }
  ]
}
```

---

### `POST /api/execute`

Manual approval gate. On approval, fetches real-time price per trade and updates the portfolio. On rejection, marks all trades `accepted=false`.

**Request:**
```json
{
  "user_id": "mallika_01",
  "session_id": "sess_abc123",
  "approved": true
}
```

**Response (approved):**
```json
{
  "status": "executed",
  "trade_results": [
    {
      "trade_id": 7,
      "ticker": "HDFC",
      "side": "SELL",
      "qty": 20,
      "proposed_price": 165000.0,
      "executed_price": 164820.5,
      "total_value": 3296410.0,
      "status": "SUCCESS"
    }
  ],
  "updated_portfolio": {
    "holdings": {"HDFC": 80, "RELIANCE": 50, "TCS": 20},
    "cash": 3796410.0
  }
}
```

**Response (rejected):**
```json
{
  "status": "rejected",
  "message": "Trade plan rejected. All proposed trades marked as rejected."
}
```

---

## Setup

### Prerequisites

- Python 3.11+
- Docker Desktop (for PostgreSQL)
- [Ollama](https://ollama.ai) with `llama3.1` pulled
- LangSmith account (free) for tracing

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install sqlalchemy psycopg2-binary uvicorn python-multipart
```

### 2. Configure `.env`

```env
DATABASE_URL=postgresql://portfolio_user:portfolio_pass@localhost:5432/portfolio_db

# LangSmith — get key from https://smith.langchain.com
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=portfolio-agent
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

### 3. Start PostgreSQL

```bash
docker compose up -d
```

### 4. Pull the LLM

```bash
ollama pull llama3.1
```

### 5. Start the server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

DB tables are created automatically on first startup. First request for a new `user_id` seeds the portfolio with:
- Holdings: HDFC x100, RELIANCE x50, TCS x20
- Cash: 500,000

---

## Using the UI

Open **http://localhost:8000**

1. Enter a **User ID**
2. Session ID is auto-generated (or enter your own)
3. Enter a **Goal** — be specific to force tool calls:
   - *"Check my holdings and prices, then tell me which stock to trim to reduce concentration risk."*
   - *"Fetch my portfolio and suggest which stock to buy more of given current prices."*
   - *"I want to rebalance. Check holdings and prices, then give me exact trades to make each stock equal weight."*
4. Click **Run Analysis** — takes 30–60s with a local LLM
5. Review the **Decision Summary**, **Risk status**, **Proposed Trades** table with IDs
6. Click **Approve & Execute** or **Reject**

---

## Observability

### LangSmith (recommended)

**https://smith.langchain.com** → Projects → `portfolio-agent`

Each run shows:
- Full LLM inputs/outputs for analyst and risk auditor nodes
- Tool calls and MCP tool results
- LangGraph node-by-node execution trace
- Token usage and latency per step
- Retry loops with rejection feedback

### Server console

```
[STARTUP] LangSmith tracing=ON project='portfolio-agent'
[ANALYST NODE] Call #1
[ANALYST] tool_calls: ['get_portfolio', 'get_live_price']
[TOOL NODE] 2 call(s)
  get_portfolio args={'user_id': 'mallika_01'}
  → {"holdings": {"HDFC": 100, ...}, "cash": 500000.0}
[RISK NODE]
[RISK] Response: APPROVED — the plan is conservative...
[EXECUTE] HDFC live price: 164820.5
[MCP] record_trade user=mallika_01 SELL 20xHDFC@164820.5
```

### Swagger UI

**http://localhost:8000/docs** — interactive API explorer

### Direct DB queries

```bash
# Portfolio state
docker exec portfolio_agenticai-postgres-1 psql -U portfolio_user -d portfolio_db \
  -c "SELECT user_id, cash FROM users; SELECT * FROM portfolios;"

# Trade history
docker exec portfolio_agenticai-postgres-1 psql -U portfolio_user -d portfolio_db \
  -c "SELECT id, ticker, side, qty, proposed_price, executed_price, accepted FROM trade_history;"

# All sessions
docker exec portfolio_agenticai-postgres-1 psql -U portfolio_user -d portfolio_db \
  -c "SELECT session_id, goal, risk_approved, executed FROM analysis_sessions;"
```

---

## Idempotency

Protection at two levels:

**Session level** — `analysis_sessions.executed` is set to `true` before any trade runs. A second call to `/api/execute` with the same `session_id` returns HTTP 400.

**Trade level** — `trade_history` has a unique constraint on `(session_id, ticker, side)`. `crud.record_trade` checks `accepted=True` before executing; duplicate calls return `ALREADY_EXECUTED` with the original `trade_id`.

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **PostgreSQL** | ACID guarantees for financial data; JSON column for flexible trade storage |
| **Sync SQLAlchemy** | MCP server is a subprocess and cannot use async drivers; keeps DB layer consistent across both contexts |
| **MCP subprocess** | Standard LLM tool protocol; decouples tool implementation from agent logic |
| **`record_trade` hidden from LLM** | Prevents the LLM from executing trades autonomously; execution only via explicit user approval |
| **`user_id` injected in `tool_node`** | LLMs unreliable at consistently passing IDs; injection at infrastructure layer is deterministic |
| **Two-phase pricing** | `proposed_price` (analysis time) vs `executed_price` (execution time) captures real price slippage |
| **Retry loop with feedback** | Risk rejection reason injected back as a message so the analyst can revise the plan |
| **LangSmith tracing** | Zero-code observability — LangGraph auto-instruments when env vars are set |
