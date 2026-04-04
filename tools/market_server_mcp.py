"""
MCP server — runs as a subprocess.
Tools: get_portfolio, get_live_price, get_prices_batch, record_trade.
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server.fastmcp import FastMCP
import yfinance as yf

from db.engine import SessionLocal
import db.crud as crud

mcp = FastMCP("WallStreetEngine")

# Dow Jones Industrial Average 30 — ticker → Yahoo Finance symbol.
# US tickers map directly (no suffix). Kept explicit for whitelist validation.
_TICKER_MAP = {
    "AAPL": "AAPL",  "MSFT": "MSFT",  "AMZN": "AMZN",  "IBM":  "IBM",
    "CSCO": "CSCO",  "CRM":  "CRM",   "NVDA": "NVDA",
    "JPM":  "JPM",   "GS":   "GS",    "AXP":  "AXP",   "V":    "V",
    "TRV":  "TRV",   "JNJ":  "JNJ",   "MRK":  "MRK",   "UNH":  "UNH",
    "AMGN": "AMGN",  "BA":   "BA",    "CAT":  "CAT",   "HON":  "HON",
    "MMM":  "MMM",   "CVX":  "CVX",   "KO":   "KO",    "WMT":  "WMT",
    "MCD":  "MCD",   "PG":   "PG",    "NKE":  "NKE",   "DIS":  "DIS",
    "VZ":   "VZ",    "HD":   "HD",    "SHW":  "SHW",
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
    Fetches the current market price for a single stock ticker via Yahoo Finance.
    Returns live price if market is open, otherwise previous day's closing price.
    Use for targeted single-ticker queries only.
    For holistic or full-rebalance analysis use get_prices_batch instead.
    """
    upper = ticker.upper()
    symbol = _TICKER_MAP.get(upper, upper)
    print(f"[MCP] get_live_price ticker={ticker} → {symbol}")
    try:
        stock = yf.Ticker(symbol)
        price = stock.fast_info.last_price
        if price and price > 0:
            print(f"[MCP] Live price for {ticker}: {price}")
            return str(round(float(price), 2))

        hist = stock.history(period="5d")
        if not hist.empty:
            prev_close = hist["Close"].iloc[-1]
            print(f"[MCP] Market closed, using prev close for {ticker}: {prev_close}")
            return str(round(float(prev_close), 2))

    except Exception as e:
        print(f"[MCP] yfinance error for {symbol}: {e}")

    return f"ERROR: price unavailable for {ticker}"


# ── Tool 3: get_prices_batch ──────────────────────────────────────────────────
@mcp.tool()
def get_prices_batch(tickers: list[str], user_id: str = "") -> str:
    """
    Fetches current market prices for multiple tickers in a single batch call.
    Returns JSON dict {ticker: price}. Use for holistic portfolio analysis
    (all held tickers) or full portfolio rebalance (all 30 DJI tickers).
    Significantly faster than calling get_live_price once per ticker.
    """
    if not tickers:
        return json.dumps({})

    upper_tickers = [t.upper() for t in tickers]
    symbol_map = {t: _TICKER_MAP.get(t, t) for t in upper_tickers}
    symbols = list(symbol_map.values())
    print(f"[MCP] get_prices_batch {len(symbols)} tickers: {upper_tickers}")

    results = {}
    try:
        data = yf.download(symbols, period="2d", progress=False, auto_adjust=True)

        for ticker_key, symbol in symbol_map.items():
            try:
                if len(symbols) == 1:
                    price = data["Close"].dropna().iloc[-1]
                else:
                    price = data["Close"][symbol].dropna().iloc[-1]
                results[ticker_key] = round(float(price), 2)
            except Exception as e:
                print(f"[MCP] get_prices_batch: failed for {ticker_key}: {e}")
                results[ticker_key] = None

    except Exception as e:
        print(f"[MCP] get_prices_batch error: {e}")
        for t in upper_tickers:
            results[t] = None

    available = sum(1 for v in results.values() if v is not None)
    print(f"[MCP] get_prices_batch complete: {available}/{len(tickers)} prices fetched")
    return json.dumps(results)


# ── Tool 4: record_trade ──────────────────────────────────────────────────────
@mcp.tool()
def record_trade(user_id: str, session_id: str, ticker: str, side: str, qty: int, price: float) -> str:
    """
    Executes and records a trade in the database. Updates the user portfolio.
    Idempotent: calling twice with the same (session_id, ticker, side) is safe.
    """
    print(f"[MCP] record_trade user={user_id} session={session_id} {side} {qty}x{ticker}@${price}")
    db = SessionLocal()
    try:
        result = crud.record_trade(db, session_id, user_id, ticker, side, qty, price)
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
    return json.dumps(result)


if __name__ == "__main__":
    mcp.run()
