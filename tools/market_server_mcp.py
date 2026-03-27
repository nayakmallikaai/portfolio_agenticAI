from mcp.server.fastmcp import FastMCP
import json

mcp = FastMCP("WallStreetEngine")

# State-persistent "Database"
DB = {
    "holdings": {"HDFC": 100, "RELIANCE": 5000, "TCS": 20},
    "cash": 500000.0,
    "trade_history": []
}

@mcp.tool()
def get_portfolio() -> str:
    """Returns the current stock holdings and cash balance."""
    print("DEBUG: get_portfolio received request!")
    return json.dumps(DB)

@mcp.tool()
def get_live_price(ticker: str) -> float:
    """Fetches real-time price for Indian stocks (Simulated)."""
    prices = {"HDFC": 165000.0, "RELIANCE": 290.0, "TCS": 3800.0}
    print("DEBUG: get_live_price received request!")
    return prices.get(ticker.upper(), 1200.0)

@mcp.tool()
def record_trade(ticker: str, side: str, qty: int, price: float) -> str:
    """Executes and logs a trade. Updates portfolio state."""
    total = qty * price
    if side.upper() == "BUY" and DB["cash"] < total:
        return "ERROR: Insufficient funds."
    
    if side.upper() == "BUY":
        DB["cash"] -= total
        DB["holdings"][ticker] = DB["holdings"].get(ticker, 0) + qty
    else:
        DB["cash"] += total
        DB["holdings"][ticker] = DB["holdings"].get(ticker, 0) - qty
        
    DB["trade_history"].append({"ticker": ticker, "side": side, "qty": qty})
    return f"SUCCESS: {side} {qty} {ticker} at {price}. Current Cash: {DB['cash']}"

if __name__ == "__main__":
    mcp.run()