from dotenv import load_dotenv
load_dotenv()  # must run before any langchain/langgraph import

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters

from db.engine import engine, migrate_db
from db.models import Base
from api.routes import router, set_mcp_holder

server_params = StdioServerParameters(command="python", args=["tools/market_server_mcp.py"])


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Create all DB tables on startup (idempotent), then run incremental migrations
    Base.metadata.create_all(bind=engine)
    migrate_db()

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            set_mcp_holder({"session": session})
            import os
            project = os.getenv("LANGCHAIN_PROJECT", "default")
            tracing = os.getenv("LANGCHAIN_TRACING_V2", "false")
            print(f"[STARTUP] DB tables ready. MCP session ready.")
            print(f"[STARTUP] LangSmith tracing={'ON' if tracing == 'true' else 'OFF'} project='{project}'")
            yield

    set_mcp_holder({})
    print("[SHUTDOWN] MCP session closed.")


app = FastAPI(title="Portfolio Agent API", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.include_router(router)


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open("static/index.html") as f:
        return f.read()
