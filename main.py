"""
MentorAI Backend v3.2
Scalable SaaS Production Architecture
Render + Vercel Compatible
"""

from __future__ import annotations

# ─────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────

import os
import re
import json
import time
import base64
import asyncio
import logging
from typing import Any, Optional
from collections import defaultdict
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field, validator

from openai import OpenAI

import firebase_admin
from firebase_admin import credentials, auth

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("mentorai")

# ─────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    raise RuntimeError("NVIDIA_API_KEY not set")

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

FRONTEND_ORIGINS = [
    "https://mentorai-blush.vercel.app",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
]

# ─────────────────────────────────────────────
# Firebase Init
# ─────────────────────────────────────────────

if not firebase_admin._apps:
    firebase_json = os.getenv("FIREBASE_ADMIN_JSON")
    if not firebase_json:
        raise RuntimeError("FIREBASE_ADMIN_JSON not set")

    cred_dict = json.loads(firebase_json)
    cred = credentials.Certificate(cred_dict)
    firebase_admin.initialize_app(cred)
    log.info("Firebase initialized")

# ─────────────────────────────────────────────
# OpenAI Client (Singleton)
# ─────────────────────────────────────────────

_client: Optional[OpenAI] = None

def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=NVIDIA_BASE_URL,
            api_key=NVIDIA_API_KEY
        )
        log.info("NVIDIA client ready")
    return _client

# ─────────────────────────────────────────────
# Plan + Rate System
# ─────────────────────────────────────────────

RATE_WINDOW = 60
RATE_FREE = 30
RATE_PRO = 120
DAILY_FREE_LIMIT = 20

_rates: dict[str, list[float]] = defaultdict(list)
_daily_usage: dict[str, tuple[int, int]] = {}

def rate_check(uid: str, plan: str):
    now = time.monotonic()
    limit = RATE_PRO if plan == "pro" else RATE_FREE
    cutoff = now - RATE_WINDOW

    _rates[uid] = [t for t in _rates[uid] if t > cutoff]
    if len(_rates[uid]) >= limit:
        raise HTTPException(429, "Rate limit exceeded")

    _rates[uid].append(now)

    if plan == "free":
        day = int(time.time() // 86400)
        count, stored = _daily_usage.get(uid, (0, day))
        if stored != day:
            count = 0
        if count >= DAILY_FREE_LIMIT:
            raise HTTPException(403, "Daily free limit reached")
        _daily_usage[uid] = (count + 1, day)

# ─────────────────────────────────────────────
# Injection Guard
# ─────────────────────────────────────────────

_INJECT = re.compile(
    r"(ignore\s+previous|override\s+instructions|developer\s+mode)",
    re.IGNORECASE
)

def guard(text: str) -> str:
    if _INJECT.search(text):
        raise HTTPException(400, "Prompt injection detected")
    return text.strip()

# ─────────────────────────────────────────────
# Auth Dependency
# ─────────────────────────────────────────────

async def get_current_user(request: Request):
    header = request.headers.get("Authorization")
    if not header or not header.startswith("Bearer "):
        raise HTTPException(401, "Missing token")

    token = header.split(" ")[1]

    try:
        decoded = auth.verify_id_token(token)
        uid = decoded["uid"]
        plan = decoded.get("plan", "free")
        return uid, plan
    except Exception:
        raise HTTPException(401, "Invalid token")

# ─────────────────────────────────────────────
# Pydantic Models
# ─────────────────────────────────────────────

class HistMsg(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str

class ChatReq(BaseModel):
    message: str = ""
    history: list[HistMsg] = []

# ─────────────────────────────────────────────
# Streaming Engine
# ─────────────────────────────────────────────

async def sse_stream(messages: list[dict]):
    client = get_client()
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    def run():
        try:
            completion = client.chat.completions.create(
                model="meta/llama-3.1-8b-instruct",
                messages=messages,
                stream=True,
                temperature=0.2
            )
            for chunk in completion:
                token = chunk.choices[0].delta.content
                if token:
                    payload = json.dumps({"choices":[{"delta":{"content":token}}]})
                    loop.call_soon_threadsafe(queue.put_nowait, f"data: {payload}\n\n")
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    loop.run_in_executor(None, run)

    while True:
        item = await queue.get()
        if item is None:
            break
        yield item

    yield "data: [DONE]\n\n"

# ─────────────────────────────────────────────
# FastAPI App (CRITICAL ORDER)
# ─────────────────────────────────────────────

app = FastAPI(title="MentorAI API", version="3.2.0")

# ✅ Correct CORS placement
app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "service": "MentorAI API",
        "status": "online",
        "auth": "firebase_required"
    }

@app.get("/health")
async def health():
    return {"ok": True}

@app.post("/api/chat")
async def chat(req: ChatReq, user_data: tuple = Depends(get_current_user)):
    uid, plan = user_data

    rate_check(uid, plan)

    safe_message = guard(req.message)

    messages = [
        {"role": "system", "content": "You are MentorAI, a professional AI assistant."}
    ]

    for h in req.history:
        messages.append({"role": h.role, "content": guard(h.content)})

    messages.append({"role": "user", "content": safe_message})

    return StreamingResponse(
        sse_stream(messages),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

# ─────────────────────────────────────────────
# Global Error Handler
# ─────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_error(request: Request, exc: Exception):
    log.error(f"Unhandled error: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
