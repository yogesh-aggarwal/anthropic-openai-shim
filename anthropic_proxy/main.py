import asyncio
import json
import os
import uuid
from typing import Any, Dict, Iterable, List, Optional, Tuple

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from config import DROP_UNSUPPORTED_PARAMS, MAX_RETRIES, REQUEST_TIMEOUT_SECONDS


app = FastAPI(title="Anthropic Adapter")

LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL", "http://litellm:4000/v1").rstrip("/")
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY", "")
ADAPTER_API_KEY = os.getenv("ANTHROPIC_PROXY_API_KEY", "")


class UpstreamError(Exception):
    def __init__(self, status_code: int, body: Any):
        self.status_code = status_code
        self.body = body
        super().__init__(f"Upstream error {status_code}")


def _resolve_model(inbound_model: str) -> str:
    if inbound_model:
        return inbound_model
    raise HTTPException(status_code=400, detail="No model provided")


def _extract_bearer_token(auth_header: Optional[str]) -> Optional[str]:
    if not auth_header:
        return None
    parts = auth_header.strip().split(" ", 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return None


def _validate_adapter_auth(x_api_key: Optional[str], authorization: Optional[str]) -> None:
    if not ADAPTER_API_KEY:
        return
    candidate = x_api_key or _extract_bearer_token(authorization)
    if candidate != ADAPTER_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


def _extract_error_message(body: Any) -> str:
    if isinstance(body, str):
        return body
    if isinstance(body, dict):
        # Common LiteLLM/OpenAI/OpenRouter shapes.
        if isinstance(body.get("error"), dict):
            msg = body["error"].get("message")
            if isinstance(msg, str) and msg:
                return msg
        msg = body.get("message")
        if isinstance(msg, str) and msg:
            return msg
        detail = body.get("detail")
        if isinstance(detail, str) and detail:
            return detail
        if isinstance(detail, dict):
            dmsg = detail.get("message")
            if isinstance(dmsg, str) and dmsg:
                return dmsg
    return "Upstream request failed"


def _error_response(status_code: int, body: Any) -> JSONResponse:
    # OpenRouter-style envelope is well tolerated by Anthropic-compatible clients.
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": _extract_error_message(body), "code": status_code}},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(_request: Request, exc: HTTPException) -> JSONResponse:
    return _error_response(exc.status_code, exc.detail)


# ----------------------------------------------
# LiteLLM translation backend
# ----------------------------------------------

def _extract_text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: List[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            parts.append(block.get("text", ""))
        elif block.get("type") == "thinking":
            parts.append(block.get("thinking", ""))
    return "\n".join([p for p in parts if p])


def _convert_tool_result_content(raw: Any) -> str:
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        text_parts: List[str] = []
        for part in raw:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(part.get("text", ""))
            elif isinstance(part, dict):
                text_parts.append(json.dumps(part, ensure_ascii=True))
            elif isinstance(part, str):
                text_parts.append(part)
        return "\n".join([x for x in text_parts if x])
    return json.dumps(raw, ensure_ascii=True)


def _anthropic_messages_to_openai(req: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    system = req.get("system")
    if isinstance(system, str) and system.strip():
        out.append({"role": "system", "content": system})
    elif isinstance(system, list):
        sys_text = _extract_text_from_content(system)
        if sys_text:
            out.append({"role": "system", "content": sys_text})

    for msg in req.get("messages", []):
        role = msg.get("role")
        content = msg.get("content")

        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue

        if not isinstance(content, list):
            continue

        if role == "assistant":
            text_parts: List[str] = []
            tool_calls: List[Dict[str, Any]] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text":
                    text_parts.append(block.get("text", ""))
                elif btype == "thinking":
                    text_parts.append(block.get("thinking", ""))
                elif btype == "tool_use":
                    call_id = block.get("id") or f"call_{uuid.uuid4().hex[:24]}"
                    tool_calls.append(
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": block.get("name"),
                                "arguments": json.dumps(block.get("input", {}), ensure_ascii=True),
                            },
                        }
                    )

            assistant_msg: Dict[str, Any] = {
                "role": "assistant",
                "content": "\n".join([p for p in text_parts if p]) or None,
            }
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            out.append(assistant_msg)
            continue

        if role == "user":
            text_parts: List[str] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text":
                    text_parts.append(block.get("text", ""))
                elif btype == "tool_result":
                    if text_parts:
                        out.append({"role": "user", "content": "\n".join([p for p in text_parts if p])})
                        text_parts = []
                    out.append(
                        {
                            "role": "tool",
                            "tool_call_id": block.get("tool_use_id"),
                            "content": _convert_tool_result_content(block.get("content", "")),
                        }
                    )
                elif btype == "thinking":
                    text_parts.append(block.get("thinking", ""))

            if text_parts:
                out.append({"role": "user", "content": "\n".join([p for p in text_parts if p])})
            continue

        as_text = _extract_text_from_content(content)
        if as_text:
            out.append({"role": role or "user", "content": as_text})

    return out


def _convert_tools(req: Dict[str, Any]) -> Tuple[Optional[List[Dict[str, Any]]], Optional[Any]]:
    tools = req.get("tools")
    if not isinstance(tools, list) or not tools:
        return None, None

    openai_tools: List[Dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        openai_tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool.get("name"),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
                },
            }
        )

    tool_choice_in = req.get("tool_choice")
    tool_choice_out: Any = None

    if isinstance(tool_choice_in, str):
        if tool_choice_in == "auto":
            tool_choice_out = "auto"
        elif tool_choice_in == "any":
            tool_choice_out = "required"
        elif tool_choice_in == "none":
            tool_choice_out = "none"
    elif isinstance(tool_choice_in, dict):
        t = tool_choice_in.get("type")
        if t == "auto":
            tool_choice_out = "auto"
        elif t == "any":
            tool_choice_out = "required"
        elif t == "none":
            tool_choice_out = "none"
        elif t == "tool":
            tool_choice_out = {
                "type": "function",
                "function": {"name": tool_choice_in.get("name")},
            }

    return openai_tools, tool_choice_out


def _reasoning_effort(req: Dict[str, Any]) -> Optional[str]:
    thinking = req.get("thinking")
    if not isinstance(thinking, dict) or thinking.get("type") != "enabled":
        return None

    budget = int(thinking.get("budget_tokens", 0) or 0)
    if budget >= 4096:
        return "high"
    if budget >= 2048:
        return "medium"
    return "low"


def _finish_reason_to_stop_reason(finish_reason: str, had_tool_calls: bool) -> str:
    if had_tool_calls or finish_reason in {"tool_calls", "function_call"}:
        return "tool_use"
    if finish_reason == "length":
        return "max_tokens"
    return "end_turn"


def _safe_json_loads(raw: str) -> Any:
    if raw is None:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {"raw": raw}


def _encode_redacted_thinking(text: str) -> Dict[str, str]:
    payload = {"text": text, "type": "reasoning.text"}
    packed = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    import base64

    return {"type": "redacted_thinking", "data": f"openrouter.reasoning:{base64.b64encode(packed).decode('ascii')}"}


def _openai_to_anthropic_response(openai_resp: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    choices = openai_resp.get("choices", [])
    if not choices:
        raise HTTPException(status_code=502, detail="Upstream response had no choices")

    msg = choices[0].get("message", {})
    finish_reason = choices[0].get("finish_reason", "stop")
    content_blocks: List[Dict[str, Any]] = []

    reasoning_text = msg.get("reasoning") or msg.get("reasoning_content")
    if isinstance(reasoning_text, str) and reasoning_text.strip():
        content_blocks.append({"type": "thinking", "thinking": reasoning_text, "signature": ""})

    text = msg.get("content")
    if isinstance(text, list):
        joined = []
        for part in text:
            if isinstance(part, dict) and part.get("type") == "text":
                joined.append(part.get("text", ""))
            elif isinstance(part, str):
                joined.append(part)
        text = "\n".join([p for p in joined if p])

    if isinstance(text, str) and text:
        content_blocks.append({"type": "text", "text": text})

    tool_calls = msg.get("tool_calls", []) or []
    for tc in tool_calls:
        fn = tc.get("function", {})
        content_blocks.append(
            {
                "type": "tool_use",
                "id": tc.get("id") or f"call_{uuid.uuid4().hex[:24]}",
                "caller": {"type": "direct"},
                "name": fn.get("name"),
                "input": _safe_json_loads(fn.get("arguments", "{}")),
            }
        )

    if isinstance(reasoning_text, str) and reasoning_text.strip():
        content_blocks.append(_encode_redacted_thinking(reasoning_text))

    usage = openai_resp.get("usage", {}) or {}
    input_tokens = int(usage.get("prompt_tokens", 0) or 0)
    output_tokens = int(usage.get("completion_tokens", 0) or 0)

    return {
        "id": openai_resp.get("id", f"gen-{uuid.uuid4()}"),
        "type": "message",
        "role": "assistant",
        "container": None,
        "content": content_blocks or [{"type": "text", "text": ""}],
        "model": model_name,
        "stop_reason": _finish_reason_to_stop_reason(finish_reason, bool(tool_calls)),
        "stop_sequence": None,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_creation_input_tokens": None,
            "cache_read_input_tokens": 0,
            "cache_creation": None,
            "inference_geo": None,
            "server_tool_use": None,
            "service_tier": None,
            "speed": "standard",
            "cost": 0,
            "is_byok": False,
            "cost_details": {
                "upstream_inference_cost": 0,
                "upstream_inference_prompt_cost": 0,
                "upstream_inference_completions_cost": 0,
            },
        },
        "provider": openai_resp.get("provider", None),
    }


async def _post_to_litellm(payload: Dict[str, Any]) -> Dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if LITELLM_API_KEY:
        headers["Authorization"] = f"Bearer {LITELLM_API_KEY}"

    delay = 0.4
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        for attempt in range(MAX_RETRIES):
            try:
                resp = await client.post(f"{LITELLM_BASE_URL}/chat/completions", headers=headers, json=payload)
                if resp.status_code in {429, 500, 502, 503, 504} and attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(delay)
                    delay *= 2
                    continue
                if resp.status_code >= 400:
                    try:
                        body: Any = resp.json()
                    except Exception:
                        body = {"error": {"message": resp.text, "code": resp.status_code}}
                    raise UpstreamError(resp.status_code, body)
                return resp.json()
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(delay)
                    delay *= 2
                    continue
                raise HTTPException(status_code=502, detail=f"Upstream network error: {exc}") from exc

    raise HTTPException(status_code=502, detail="Upstream request failed")


def _sse(event: str, payload: Dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=True)}\n\n".encode("utf-8")


def _stream_events_from_message(msg: Dict[str, Any]) -> Iterable[bytes]:
    message_start = {
        "type": "message_start",
        "message": {
            "id": msg["id"],
            "type": "message",
            "role": "assistant",
            "container": None,
            "content": [],
            "model": msg.get("model"),
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_creation_input_tokens": None,
                "cache_read_input_tokens": None,
                "cache_creation": None,
                "inference_geo": None,
                "server_tool_use": None,
                "service_tier": None,
                "speed": "standard",
            },
            "provider": msg.get("provider"),
        },
    }
    yield _sse("message_start", message_start)

    for idx, block in enumerate(msg.get("content", [])):
        btype = block.get("type")
        if btype == "thinking":
            yield _sse("content_block_start", {"type": "content_block_start", "index": idx, "content_block": {"type": "thinking", "thinking": "", "signature": ""}})
            yield _sse("content_block_delta", {"type": "content_block_delta", "index": idx, "delta": {"type": "thinking_delta", "thinking": block.get("thinking", "")}})
            yield _sse("content_block_stop", {"type": "content_block_stop", "index": idx})
        elif btype == "text":
            yield _sse("content_block_start", {"type": "content_block_start", "index": idx, "content_block": {"type": "text", "text": ""}})
            yield _sse("content_block_delta", {"type": "content_block_delta", "index": idx, "delta": {"type": "text_delta", "text": block.get("text", "")}})
            yield _sse("content_block_stop", {"type": "content_block_stop", "index": idx})
        elif btype == "tool_use":
            yield _sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": idx,
                    "content_block": {
                        "type": "tool_use",
                        "id": block.get("id"),
                        "caller": {"type": "direct"},
                        "name": block.get("name"),
                        "input": {},
                    },
                },
            )
            yield _sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": idx,
                    "delta": {"type": "input_json_delta", "partial_json": json.dumps(block.get("input", {}), ensure_ascii=True)},
                },
            )
            yield _sse("content_block_stop", {"type": "content_block_stop", "index": idx})
        elif btype == "redacted_thinking":
            yield _sse("content_block_start", {"type": "content_block_start", "index": idx, "content_block": {"type": "redacted_thinking", "data": ""}})
            yield _sse("content_block_delta", {"type": "content_block_delta", "index": idx, "delta": {"type": "signature_delta", "signature": block.get("data", "")}})
            yield _sse("content_block_stop", {"type": "content_block_stop", "index": idx})

    yield _sse(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {
                "container": None,
                "stop_reason": msg.get("stop_reason"),
                "stop_sequence": msg.get("stop_sequence"),
            },
            "usage": msg.get("usage", {}),
        },
    )
    yield _sse("message_stop", {"type": "message_stop"})
    yield b"event: data\ndata: [DONE]\n\n"


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "backend_mode": "litellm"}


@app.post("/v1/messages")
async def messages(
    request: Request,
    x_api_key: Optional[str] = Header(default=None),
    authorization: Optional[str] = Header(default=None),
    anthropic_version: Optional[str] = Header(default=None),
    anthropic_beta: Optional[str] = Header(default=None),
) -> Any:
    # Headers kept for Anthropic compatibility; adapter currently ignores version/beta flags.
    _ = anthropic_version
    _ = anthropic_beta
    _validate_adapter_auth(x_api_key, authorization)

    body = await request.json()
    body["model"] = _resolve_model(body.get("model", ""))
    openai_messages = _anthropic_messages_to_openai(body)
    openai_tools, openai_tool_choice = _convert_tools(body)

    payload: Dict[str, Any] = {
        "model": body["model"],
        "messages": openai_messages,
        "max_tokens": body.get("max_tokens", 1024),
        "temperature": body.get("temperature"),
        "top_p": body.get("top_p"),
        "stop": body.get("stop_sequences"),
        "stream": False,
    }
    if DROP_UNSUPPORTED_PARAMS:
        payload = {k: v for k, v in payload.items() if v is not None}

    if openai_tools:
        payload["tools"] = openai_tools
    if openai_tool_choice is not None:
        payload["tool_choice"] = openai_tool_choice

    reasoning_effort = _reasoning_effort(body)
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort

    try:
        upstream = await _post_to_litellm(payload)
    except UpstreamError as exc:
        return _error_response(exc.status_code, exc.body)

    anthropic_msg = _openai_to_anthropic_response(upstream, body["model"])

    if body.get("stream"):
        return StreamingResponse(_stream_events_from_message(anthropic_msg), media_type="text/event-stream")

    return anthropic_msg
