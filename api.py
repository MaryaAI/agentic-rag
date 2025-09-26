# api.py
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional, Any, Dict
import os
import httpx
import logging
import asyncio
# Add / ensure these imports near the top of api.py
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import httpx
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi import status


import time

from typing import Any, Dict, List, Optional

from fastapi import status
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

# --------------------
# Config
# --------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "512"))
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0.0"))
OPENAI_TIMEOUT_S = float(os.getenv("OPENAI_TIMEOUT_S", "60"))
OPENAI_RETRIES = int(os.getenv("OPENAI_RETRIES", "3"))
OPENAI_BACKOFF_BASE = float(os.getenv("OPENAI_BACKOFF_BASE", "0.6"))  # seconds

if not OPENAI_API_KEY:
    # Don't crash on import; endpoints will return clear error
    logging.warning("OPENAI_API_KEY not set. Chat endpoints will return an error until set.")

# --------------------
# Logging
# --------------------
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("api")

# --------------------
# FastAPI app + client
# --------------------
app = FastAPI(title="OpenAI proxy - minimal, robust")

# Create one shared AsyncClient for connection pooling
_http_client: Optional[httpx.AsyncClient] = None


@app.on_event("startup")
async def startup_event():
    global _http_client
    if _http_client is None:
        limits = httpx.Limits(max_keepalive_connections=10, max_connections=50)
        timeout = httpx.Timeout(OPENAI_TIMEOUT_S)
        _http_client = httpx.AsyncClient(limits=limits, timeout=timeout)
        logger.info("httpx AsyncClient initialized (pooling enabled)")


@app.on_event("shutdown")
async def shutdown_event():
    global _http_client
    if _http_client:
        await _http_client.aclose()
        _http_client = None
        logger.info("httpx AsyncClient closed")


# --------------------
# Request models
# --------------------
class ChatMessage(BaseModel):
    role: str = Field(..., description="role, e.g., 'user'|'assistant'|'system'")
    content: str = Field(..., description="message content")


class ChatRequest(BaseModel):
    model: Optional[str] = Field(None, description="OpenAI model id")
    messages: List[ChatMessage] = Field(..., min_items=1)
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stream: Optional[bool] = False


# --------------------
# Helpers
# --------------------
async def _call_openai_with_retries(
    payload: Dict[str, Any],
    retries: int = OPENAI_RETRIES,
) -> httpx.Response:
    """
    Call OpenAI's chat completions with retries on transient errors (5xx, 429, network).
    """
    assert _http_client is not None, "HTTP client not initialized"

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    backoff = OPENAI_BACKOFF_BASE
    for attempt in range(1, retries + 1):
        try:
            resp = await _http_client.post(url, json=payload, headers=headers)
            # On 4xx (client errors) return immediately since likely a bad request or auth problem
            if 400 <= resp.status_code < 500:
                return resp
            # On 5xx or too many requests, either return when last attempt or retry
            if resp.status_code >= 500 or resp.status_code == 429:
                logger.warning(
                    "OpenAI transient status %s (attempt %s/%s). Retrying after %.2fs",
                    resp.status_code,
                    attempt,
                    retries,
                    backoff,
                )
                if attempt == retries:
                    return resp
                await asyncio.sleep(backoff)
                backoff *= 2
                continue
            # success or other handled status
            return resp
        except (httpx.RequestError, httpx.TimeoutException) as e:
            logger.exception("Network error calling OpenAI (attempt %s/%s): %s", attempt, retries, e)
            if attempt == retries:
                # Build a synthetic Response-like object to indicate failure
                raise
            await asyncio.sleep(backoff)
            backoff *= 2

    # fallback: should not reach here
    raise RuntimeError("OpenAI call exhausted retries unexpectedly")


# --------------------
# Endpoints
# --------------------
@app.get("/healthz", response_class=PlainTextResponse)
async def healthz():
    return "ok"


@app.get("/v1/models")
async def list_models():
    if not OPENAI_API_KEY:
        return JSONResponse(status_code=500, content={"error": "OPENAI_API_KEY not set"})
    try:
        resp = await _call_openai_with_retries({"model": "list"})
        # Usually OpenAI returns JSON
        try:
            return JSONResponse(status_code=resp.status_code, content=resp.json())
        except Exception:
            return JSONResponse(status_code=resp.status_code, content={"raw": resp.text})
    except Exception as e:
        logger.exception("Failed to list OpenAI models: %s", e)
        return JSONResponse(status_code=502, content={"error": "OpenAI unreachable", "detail": str(e)})




# Pydantic request model used in handler
class ChatRequest(BaseModel):
    model: Optional[str] = "gpt-3.5-turbo"
    messages: Optional[List[Dict[str, Any]]] = []
    stream: Optional[bool] = False
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.0
    # accept extra fields if swagger posts them
    class Config:
        extra = "allow"

async def _call_openai_with_retries(payload: Dict[str, Any], headers: Dict[str, str], *, max_attempts: int = 2, timeout: float = 60.0):
    """Post to OpenAI with simple retry/backoff for transient errors."""
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers)
                resp.raise_for_status()
                return resp
        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code if e.response is not None else None
            # For non-transient client errors (except 429) raise immediately
            if status_code and 400 <= status_code < 500 and status_code != 429:
                raise
            last_exc = e
        except Exception as e:
            last_exc = e
        # backoff
        await asyncio.sleep(min(2.0, 0.5 * attempt))
    # exhausted attempts
    raise last_exc

async def _run_crewai_in_thread(user_message: str):
    """Run the create_rag_crew + kickoff in a thread to avoid blocking the loop."""
    def sync_runner(msg: str):
        # create_rag_crew is assumed to be available in global scope
        rag_crew = create_rag_crew(msg)
        return rag_crew.kickoff()
    return await asyncio.to_thread(sync_runner, user_message)

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    # --- extract a last user message robustly (keeps prior behavior) ---
    user_message = None
    try:
        for msg in reversed(request.messages or []):
            if not isinstance(msg, dict):
                continue
            role = msg.get("role") or msg.get("type") or msg.get("sender")
            content = msg.get("content") or msg.get("text") or msg.get("message")
            if role == "user" and content:
                user_message = content
                break
        if not user_message:
            # fallback: last content
            for msg in reversed(request.messages or []):
                if isinstance(msg, dict):
                    content = msg.get("content") or msg.get("text") or msg.get("message")
                    if content:
                        user_message = content
                        break
    except Exception:
        try:
            user_message = (request.messages or [])[-1].get("content")
        except Exception:
            user_message = None

    if not user_message:
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"error": "No user message found"})

    # --- choose OpenAI if configured ---
    openai_key = os.getenv("OPENAI_API_KEY")
    use_openai_flag = os.getenv("USE_OPENAI", "").strip()

    if openai_key or use_openai_flag == "1":
        model = request.model or os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        payload = {
            "model": model,
            "messages": request.messages,
            "max_tokens": int(request.max_tokens or os.getenv("OPENAI_MAX_TOKENS", 512)),
            "temperature": float(request.temperature or os.getenv("OPENAI_TEMPERATURE", 0.0)),
        }
        headers = {"Authorization": f"Bearer {openai_key}", "Content-Type": "application/json"}

        try:
            # Stream mode: proxy OpenAI streaming SSE
            if request.stream:
                async with httpx.AsyncClient(timeout=None) as client:
                    async with client.stream("POST", "https://api.openai.com/v1/chat/completions", json={**payload, "stream": True}, headers=headers) as resp:
                        # surface errors from OpenAI
                        if resp.status_code >= 400:
                            try:
                                detail = await resp.json()
                            except Exception:
                                detail = await resp.aread()
                            return JSONResponse(status_code=resp.status_code, content={"error": "OpenAI error", "detail": detail})

                        async def generator():
                            try:
                                async for line in resp.aiter_lines():
                                    if not line:
                                        continue
                                    # OpenAI SSE lines come prefixed with "data: ". Pass through.
                                    yield f"{line}\n"
                            finally:
                                try:
                                    await resp.aclose()
                                except Exception:
                                    pass

                        return StreamingResponse(generator(), media_type="text/event-stream")

            # Non-stream: call OpenAI, with retries for transient errors
            resp = await _call_openai_with_retries(payload, headers, max_attempts=2, timeout=60.0)
            # Return OpenAI response JSON (keeps swagger shape)
            return JSONResponse(status_code=resp.status_code, content=resp.json())

        except httpx.HTTPStatusError as e:
            logging.exception("OpenAI HTTP error: %s", e)
            try:
                detail = e.response.json()
            except Exception:
                detail = str(e)
            return JSONResponse(status_code=e.response.status_code if e.response is not None else 500, content={"error": "OpenAI error", "detail": detail})
        except Exception as e:
            logging.exception("OpenAI request failed; falling back to CrewAI. Error: %s", e)
            # fall through to fallback path

    # --- Fallback: run existing Crew RAG flow inside a thread (won't block) ---
    try:
        result = await _run_crewai_in_thread(user_message)

        # Normalize response into OpenAI-like structure for non-stream case
        assistant_text = None
        if isinstance(result, dict):
            assistant_text = result.get("text") or result.get("output") or result.get("answer")
        if assistant_text is None:
            assistant_text = str(result)

        return {
            "id": "chatcmpl-crewai",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "crewai-rag-fallback",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": assistant_text}, "finish_reason": "stop"}],
        }
    except Exception as e:
        logging.exception("Crew fallback failed: %s", e)
        return JSONResponse(status_code=500, content={"error": "internal server error", "detail": str(e)})
# ------------ END: OpenAI-compatible async chat handler ------------


    payload = {
        "model": chat_req.model or OPENAI_DEFAULT_MODEL,
        "messages": [m.dict() for m in chat_req.messages],
        "max_tokens": chat_req.max_tokens or OPENAI_MAX_TOKENS,
        "temperature": chat_req.temperature if chat_req.temperature is not None else OPENAI_TEMPERATURE,
    }

    try:
        resp = await _call_openai_with_retries(payload)
    except Exception as e:
        logger.exception("OpenAI call failed: %s", e)
        return JSONResponse(status_code=502, content={"error": "OpenAI request failed", "detail": str(e)})

    # 4xx client errors (bad request, auth) forwarded with body
    if 400 <= resp.status_code < 500:
        try:
            body = resp.json()
        except Exception:
            body = resp.text
        return JSONResponse(status_code=resp.status_code, content={"error": "OpenAI error", "detail": body})

    # 5xx or 200 -> forward parsed JSON or raw text
    try:
        return JSONResponse(status_code=resp.status_code, content=resp.json())
    except Exception:
        return JSONResponse(status_code=resp.status_code, content={"raw": resp.text})
