"""Authenticated single-worker service. Start with uvicorn api:app."""
import hashlib
import logging
import os
import secrets
import sqlite3
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from settings import load_settings, server_key

load_settings()

LOG = logging.getLogger(__name__)
HERE = Path(__file__).resolve().parent


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


class ChatResponse(BaseModel):
    session_id: UUID
    answer: str
    elapsed_seconds: float


class SessionResponse(BaseModel):
    session_id: UUID


def create_app(runtime_factory=None, db_path=None, api_key=None):
    key = api_key if api_key is not None else os.getenv("CHAT_API_KEY", "")
    database = str(db_path or os.getenv("CHAT_DB_PATH", str(HERE / "data" / "sessions.sqlite3")))
    capacity = threading.BoundedSemaphore(int(os.getenv("CHAT_MAX_CONCURRENT", "2")))
    locks = [threading.Lock() for _ in range(64)]

    @contextmanager
    def connect():
        conn = sqlite3.connect(database, timeout=10)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    @asynccontextmanager
    async def lifespan(app):
        nonlocal key
        if not key:
            key = server_key()
        Path(database).parent.mkdir(parents=True, exist_ok=True)
        with connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, history TEXT NOT NULL)")
        if runtime_factory:
            app.state.respond = runtime_factory()
        else:
            from geo_chat import create_runtime
            app.state.respond = create_runtime()
        yield

    app = FastAPI(title="GeoJSON Chat API", version="0.3.0", lifespan=lifespan)
    header = APIKeyHeader(name="X-API-Key", auto_error=False)

    def authorize(value: str | None = Depends(header)):
        if not value or not secrets.compare_digest(value.encode(), key.encode()):
            raise HTTPException(401, "Invalid API key")

    @app.get("/health")
    def health():
        return {"status": "ok", "dataset": "Workspace GeoJSON layers"}

    @app.get("/ready")
    def ready():
        try:
            with connect() as conn:
                conn.execute("SELECT 1 FROM sessions LIMIT 1")
            if os.getenv("DATABASE_URL"):
                from postgres_store import connect as pg_connect
                with pg_connect() as conn:
                    conn.execute("SELECT 1 FROM geo_features LIMIT 1")
            if not callable(getattr(app.state, "respond", None)):
                raise RuntimeError("Runtime not initialized")
        except Exception:
            raise HTTPException(503, "Service is not ready")
        return {"status": "ready"}

    @app.get("/layers", dependencies=[Depends(authorize)])
    def list_layers():
        from geo_data import CATALOG
        import json
        if not CATALOG.exists():
            raise HTTPException(503, "Layer catalog has not been built")
        return json.loads(CATALOG.read_text(encoding="utf-8"))["layers"]

    @app.post("/sessions", response_model=SessionResponse, dependencies=[Depends(authorize)])
    def create_session():
        session_id = uuid4()
        with connect() as conn:
            conn.execute("INSERT INTO sessions VALUES (?, ?)", (str(session_id), "[]"))
        return {"session_id": session_id}

    @app.post("/sessions/{session_id}/chat", response_model=ChatResponse, dependencies=[Depends(authorize)])
    def chat(session_id: UUID, body: ChatRequest):
        import json
        from langchain_core.messages import AIMessage, HumanMessage
        if not body.message.strip():
            raise HTTPException(422, "Message cannot be blank")
        lock_index = int(hashlib.sha256(str(session_id).encode()).hexdigest(), 16) % len(locks)
        lock = locks[lock_index]
        if not lock.acquire(blocking=False):
            raise HTTPException(409, "Session is processing another request; retry shortly")
        try:
            with connect() as conn:
                row = conn.execute("SELECT history FROM sessions WHERE id=?", (str(session_id),)).fetchone()
            if row is None:
                raise HTTPException(404, "Session not found")
            if not capacity.acquire(blocking=False):
                raise HTTPException(429, "Service busy; retry shortly", headers={"Retry-After": "5"})
            try:
                stored = json.loads(row[0])
                history = [(HumanMessage if item["role"] == "user" else AIMessage)(content=item["content"])
                           for item in stored]
                started = time.monotonic()
                response = app.state.respond(body.message, history)
                stored.extend([{"role": "user", "content": body.message},
                               {"role": "assistant", "content": response}])
                # Bound both message count and the size of persisted context.
                stored = stored[-8:]
                for item in stored:
                    item["content"] = item["content"][:8000]
                with connect() as conn:
                    conn.execute("UPDATE sessions SET history=? WHERE id=?",
                                 (json.dumps(stored), str(session_id)))
                return {"session_id": session_id, "answer": response,
                        "elapsed_seconds": round(time.monotonic() - started, 3)}
            except Exception:
                LOG.exception("Chat request failed")
                raise HTTPException(502, "Chat processing failed; check server logs and provider configuration")
            finally:
                capacity.release()
        finally:
            lock.release()

    @app.delete("/sessions/{session_id}", dependencies=[Depends(authorize)], status_code=204)
    def delete_session(session_id: UUID):
        lock_index = int(hashlib.sha256(str(session_id).encode()).hexdigest(), 16) % len(locks)
        lock = locks[lock_index]
        if not lock.acquire(blocking=False):
            raise HTTPException(409, "Session is processing another request")
        try:
            with connect() as conn:
                count = conn.execute("DELETE FROM sessions WHERE id=?", (str(session_id),)).rowcount
            if not count:
                raise HTTPException(404, "Session not found")
        finally:
            lock.release()

    return app


app = create_app()
