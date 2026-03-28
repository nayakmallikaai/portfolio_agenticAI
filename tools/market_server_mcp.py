"""
MCP server — runs as a subprocess.
Tools: get_portfolio, get_live_price (yfinance), record_trade (DB + idempotency).
"""
import sys
import os
import json

# Ensure project root is on sys.path so db.* imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server.fastmcp import FastMCP
import yfinance as yf

from db.engine import SessionLocal
import db.crud as crud

mcp = FastMCP("WallStreetEngine")

# Map short tickers → Yahoo Finance NSE symbols
_NSE_MAP = {
    "HDFC":       "HDFCBANK.NS",
    "RELIANCE":   "RELIANCE.NS",
    "TCS":        "TCS.NS",
    "INFY":       "INFY.NS",
    "WIPRO":      "WIPRO.NS",
    "BAJFINANCE": "BAJFINANCE.NS",
    "ICICIBANK":  "ICICIBANK.NS",
    "SBIN":       "SBIN.NS",
    "PHARMA_1":   "SUNPHARMA.NS",
    "PHARMA1":    "SUNPHARMA.NS",
}



# ── Tool 1: get_portfolio ─────────────────────────────────────────────────────
@mcp.tool()
def get_portfolio(user_id: str) -> str:
    """Returns the current stock holdings and cash balance for a user from the database."""
    print(f"[MCP] get_portfolio user_id={user_id}")
    db = SessionLocal()
    try:
        portfolio = crud.get_portfolio(db, user_id)
    finally:
        db.close()
    return json.dumps(portfolio)


# ── Tool 2: get_live_price ────────────────────────────────────────────────────
@mcp.tool()
def get_live_price(ticker: str, user_id: str = "") -> str:
    """
    Fetches the current market price for a stock ticker from NSE via Yahoo Finance.
    Returns the live price if the market is open, otherwise the previous day's closing price.
    user_id is accepted for uniform tool signature but not used for price lookup.
    """
    upper = ticker.upper()
    yf_symbol = _NSE_MAP.get(upper, upper + ".NS")
    print(f"[MCP] get_live_price ticker={ticker} → {yf_symbol}")
    try:
        stock = yf.Ticker(yf_symbol)

        # Try live price first (works when market is open)
        price = stock.fast_info.last_price
        if price and price > 0:
            print(f"[MCP] Live price for {ticker}: {price}")
            return str(round(float(price), 2))

        # Market closed — fall back to previous close from daily history
        hist = stock.history(period="5d")
        if not hist.empty:
            prev_close = hist["Close"].iloc[-1]
            print(f"[MCP] Market closed, using prev close for {ticker}: {prev_close}")
            return str(round(float(prev_close), 2))

    except Exception as e:
        print(f"[MCP] yfinance error for {yf_symbol}: {e}")

    return f"ERROR: price unavailable for {ticker}"


# ── Tool 3: record_trade ──────────────────────────────────────────────────────
@mcp.tool()
def record_trade(user_id: str, session_id: str, ticker: str, side: str, qty: int, price: float) -> str:
    """
    Executes and records a trade in the database. Updates the user portfolio.
    Idempotent: calling twice with the same (session_id, ticker, side) is safe.
    """
    print(f"[MCP] record_trade user={user_id} session={session_id} {side} {qty}x{ticker}@{price}")
    db = SessionLocal()
    try:
        result = crud.record_trade(db, session_id, user_id, ticker, side, qty, price)
    finally:
        db.close()
    return json.dumps(result)


if __name__ == "__main__":
    mcp.run()
