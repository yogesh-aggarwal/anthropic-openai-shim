import asyncio
import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Iterable, List, Optional, Tuple

import httpx
from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse

from config import DROP_UNSUPPORTED_PARAMS, MAX_RETRIES, REQUEST_TIMEOUT_SECONDS


app = FastAPI(title="Anthropic Adapter")

LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL", "http://litellm:4000/v1").rstrip("/")
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY", "")
ADAPTER_API_KEY = os.getenv("ANTHROPIC_PROXY_API_KEY", "")
STRICT_SIGNED_THINKING = os.getenv("STRICT_SIGNED_THINKING", "true").strip().lower() in {"1", "true", "yes", "on"}
ALLOW_UNSIGNED_THINKING_WHEN_REQUESTED = os.getenv("ALLOW_UNSIGNED_THINKING_WHEN_REQUESTED", "false").strip().lower() in {"1", "true", "yes", "on"}
ALLOW_UNSIGNED_THINKING_USER_AGENTS = [
    token.strip().lower()
    for token in os.getenv("ALLOW_UNSIGNED_THINKING_USER_AGENTS", "").split(",")
    if token.strip()
]
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").strip().upper()
REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_FILE = REPO_ROOT / "models.yaml"
ANTHROPIC_API_VERSION = "2023-06-01"
_HTTP_CLIENT: Optional[httpx.AsyncClient] = None
RETRYABLE_STATUS_CODES = {500, 502, 503, 504}

logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
LOGGER = logging.getLogger("anthropic_proxy")


class UpstreamError(Exception):
    def __init__(self, status_code: int, body: Any):
        self.status_code = status_code
        self.body = body
        super().__init__(f"Upstream error {status_code}")


def _get_http_client() -> httpx.AsyncClient:
    if _HTTP_CLIENT is None:
        raise HTTPException(status_code=500, detail="HTTP client not initialized")
    return _HTTP_CLIENT


@app.on_event("startup")
async def startup_event() -> None:
    global _HTTP_CLIENT
    _HTTP_CLIENT = httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS)


@app.on_event("shutdown")
async def shutdown_event() -> None:
    global _HTTP_CLIENT
    if _HTTP_CLIENT is not None:
        await _HTTP_CLIENT.aclose()
    _HTTP_CLIENT = None


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


def _thinking_requested(body: Dict[str, Any]) -> bool:
    thinking = body.get("thinking")
    return isinstance(thinking, dict) and thinking.get("type") == "enabled"


def _should_require_signed_thinking(body: Dict[str, Any], user_agent: Optional[str]) -> bool:
    if not STRICT_SIGNED_THINKING:
        return False

    if not _thinking_requested(body):
        return True

    if not ALLOW_UNSIGNED_THINKING_WHEN_REQUESTED:
        return True

    if not ALLOW_UNSIGNED_THINKING_USER_AGENTS:
        return False

    ua = (user_agent or "").lower()
    return not any(token in ua for token in ALLOW_UNSIGNED_THINKING_USER_AGENTS)


def _collect_message_content_types(messages: Any) -> List[str]:
    seen: set[str] = set()
    if not isinstance(messages, list):
        return []

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str):
            seen.add("text")
            continue
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, str):
                seen.add("text")
            elif isinstance(block, dict):
                btype = block.get("type")
                if isinstance(btype, str) and btype:
                    seen.add(btype)

    return sorted(seen)


def _extract_error_message(body: Any) -> str:
    if isinstance(body, str):
        return body
    if isinstance(body, dict):
        # Common LiteLLM/OpenAI error shapes.
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
    # Map status codes to Anthropic error types per specification
    error_type = "api_error"
    if status_code in (400, 422):
        error_type = "invalid_request_error"
    elif status_code == 401:
        error_type = "authentication_error"
    elif status_code == 403:
        error_type = "permission_error"
    elif status_code == 404:
        error_type = "not_found_error"
    elif status_code == 429:
        error_type = "overloaded_error"

    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": _extract_error_message(body), "type": error_type}},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(_request: Request, exc: HTTPException) -> JSONResponse:
    return _error_response(exc.status_code, {"error": {"message": exc.detail}})


def _require_anthropic_version(anthropic_version: Optional[str]) -> None:
    if not anthropic_version:
        raise HTTPException(status_code=400, detail="Missing anthropic-version header")
    if anthropic_version != ANTHROPIC_API_VERSION:
        raise HTTPException(status_code=400, detail=f"Unsupported anthropic-version: {anthropic_version}")


def _usage_from_upstream(usage: Any) -> Dict[str, Any]:
    if not isinstance(usage, dict):
        usage = {}

    cache_creation = usage.get("cache_creation_input_tokens")
    cache_read = usage.get("cache_read_input_tokens")

    return {
        "input_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "output_tokens": int(usage.get("completion_tokens", 0) or 0),
        "cache_creation_input_tokens": int(cache_creation or 0) if cache_creation is not None else None,
        "cache_read_input_tokens": int(cache_read or 0) if cache_read is not None else 0,
    }


def _extract_stop_sequence(openai_resp: Dict[str, Any], choice: Dict[str, Any]) -> Optional[str]:
    for candidate in (
        choice.get("stop_sequence"),
        choice.get("stop"),
        openai_resp.get("stop_sequence"),
        openai_resp.get("stop"),
    ):
        if isinstance(candidate, str) and candidate:
            return candidate
    return None


def _approx_token_count(text: str) -> int:
    normalized = text.strip()
    if not normalized:
        return 0
    return max(1, (len(normalized) + 3) // 4)


def _count_tokens_for_content(content: Any) -> int:
    if isinstance(content, str):
        return _approx_token_count(content)

    if isinstance(content, list):
        total = 0
        for block in content:
            if isinstance(block, str):
                total += _approx_token_count(block)
                continue

            if not isinstance(block, dict):
                total += _approx_token_count(str(block))
                continue

            block_type = block.get("type")
            if block_type == "text":
                total += _approx_token_count(str(block.get("text", "")))
            elif block_type == "thinking":
                total += _approx_token_count(str(block.get("thinking", "")))
            elif block_type == "redacted_thinking":
                total += 8
            elif block_type in {"image", "document", "tool_use", "tool_result"}:
                total += _approx_token_count(json.dumps(block, ensure_ascii=True, separators=(",", ":")))
            else:
                total += _approx_token_count(json.dumps(block, ensure_ascii=True, separators=(",", ":")))
        return total

    if isinstance(content, dict):
        return _approx_token_count(json.dumps(content, ensure_ascii=True, separators=(",", ":")))

    if content is None:
        return 0

    return _approx_token_count(str(content))


def _count_tokens_for_request(req: Dict[str, Any]) -> int:
    total = 0

    system = req.get("system")
    if isinstance(system, str):
        total += _count_tokens_for_content(system)
    elif isinstance(system, list):
        total += _count_tokens_for_content(system)

    messages = req.get("messages", [])
    if isinstance(messages, list):
        for msg in messages:
            if not isinstance(msg, dict):
                total += _count_tokens_for_content(msg)
                continue
            total += 4
            total += _count_tokens_for_content(msg.get("content"))

    tools = req.get("tools")
    if isinstance(tools, list):
        total += 2
        total += _count_tokens_for_content(tools)

    total += _count_tokens_for_content(req.get("thinking"))
    total += _count_tokens_for_content(req.get("tool_choice"))
    total += _count_tokens_for_content(req.get("output_config"))

    return total


def _read_model_catalog() -> List[Dict[str, Any]]:
    if not MODELS_FILE.exists():
        return []

    catalog: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    in_model_list = False

    for raw_line in MODELS_FILE.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if stripped == "model_list:":
            in_model_list = True
            continue

        if not in_model_list:
            continue

        if stripped.startswith("- model_name:"):
            if current is not None:
                catalog.append(current)
            current = {"id": stripped.split(":", 1)[1].strip()}
            continue

        if current is None:
            continue

        if stripped.startswith("model:"):
            current["upstream_model"] = stripped.split(":", 1)[1].strip()

    if current is not None:
        catalog.append(current)

    if not catalog:
        return []

    created_at = datetime.fromtimestamp(MODELS_FILE.stat().st_mtime, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out: List[Dict[str, Any]] = []
    for item in catalog:
        model_id = item.get("id")
        if not isinstance(model_id, str) or not model_id:
            continue
        pretty_name = re.sub(r"[:_-]+", " ", model_id).strip().title()
        out.append(
            {
                "id": model_id,
                "type": "model",
                "display_name": pretty_name,
                "created_at": created_at,
                "max_input_tokens": 0,
                "max_tokens": 0,
                "capabilities": {
                    "batch": {"supported": False},
                    "citations": {"supported": False},
                    "code_execution": {"supported": False},
                    "context_management": {"supported": False},
                    "effort": {
                        "supported": True,
                        "low": {"supported": True},
                        "medium": {"supported": True},
                        "high": {"supported": True},
                        "max": {"supported": True},
                    },
                    "image_input": {"supported": True},
                    "pdf_input": {"supported": True},
                    "structured_outputs": {"supported": False},
                    "thinking": {
                        "supported": True,
                        "types": {
                            "enabled": {"supported": True},
                            "adaptive": {"supported": False},
                        },
                    },
                },
            }
        )

    return out


def _get_model_catalog() -> List[Dict[str, Any]]:
    catalog = _read_model_catalog()
    if catalog:
        return catalog

    return [
        {
            "id": "anthropic-proxy-default",
            "type": "model",
            "display_name": "Anthropic Proxy Default",
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "max_input_tokens": 0,
            "max_tokens": 0,
            "capabilities": {
                "batch": {"supported": False},
                "citations": {"supported": False},
                "code_execution": {"supported": False},
                "context_management": {"supported": False},
                "effort": {"supported": True, "low": {"supported": True}, "medium": {"supported": True}, "high": {"supported": True}, "max": {"supported": True}},
                "image_input": {"supported": True},
                "pdf_input": {"supported": True},
                "structured_outputs": {"supported": False},
                "thinking": {"supported": True, "types": {"enabled": {"supported": True}, "adaptive": {"supported": False}}},
            },
        }
    ]


def _find_model(model_id: str) -> Optional[Dict[str, Any]]:
    for model in _get_model_catalog():
        if model.get("id") == model_id:
            return model
    return None


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


def _extract_reasoning_blocks(msg: Dict[str, Any]) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    fallback_signature = msg.get("reasoning_signature")
    if not isinstance(fallback_signature, str):
        fallback_signature = ""

    def append_thinking(text: Any, signature: Any) -> None:
        if not isinstance(text, str) or not text.strip():
            return
        sig = signature if isinstance(signature, str) else ""
        blocks.append({"type": "thinking", "thinking": text, "signature": sig})

    def append_redacted(data: Any) -> None:
        if isinstance(data, str) and data:
            blocks.append({"type": "redacted_thinking", "data": data})

    def consume(raw: Any, default_signature: str) -> None:
        if isinstance(raw, str):
            append_thinking(raw, default_signature)
            return

        if isinstance(raw, dict):
            raw_type = raw.get("type")
            if raw_type in {"redacted_thinking", "reasoning.encrypted", "reasoning.redacted"}:
                append_redacted(raw.get("data") or raw.get("encrypted") or raw.get("ciphertext"))
                return

            text_value = raw.get("thinking")
            if not isinstance(text_value, str):
                text_value = raw.get("text")
            append_thinking(text_value, raw.get("signature") or default_signature)
            return

        if isinstance(raw, list):
            for item in raw:
                consume(item, default_signature)

    consume(msg.get("reasoning"), fallback_signature)
    if not blocks:
        consume(msg.get("reasoning_content"), fallback_signature)

    return blocks


def _openai_to_anthropic_response(
    openai_resp: Dict[str, Any],
    model_name: str,
) -> Dict[str, Any]:
    choices = openai_resp.get("choices", [])
    if not choices:
        raise HTTPException(status_code=502, detail="Upstream response had no choices")

    choice = choices[0] if isinstance(choices[0], dict) else {}
    msg = choice.get("message", {}) if isinstance(choice.get("message", {}), dict) else {}
    finish_reason = choice.get("finish_reason", "stop")
    content_blocks: List[Dict[str, Any]] = []
    found_reasoning = False

    text_parts: List[str] = []
    raw_content = msg.get("content")
    if isinstance(raw_content, list):
        for part in raw_content:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type == "text":
                text_value = part.get("text", "")
                if isinstance(text_value, str) and text_value:
                    text_parts.append(text_value)
            elif part_type == "thinking":
                found_reasoning = True
                content_blocks.append(
                    {
                        "type": "thinking",
                        "thinking": str(part.get("thinking", part.get("text", ""))),
                        "signature": str(part.get("signature", msg.get("reasoning_signature", "")) or ""),
                    }
                )
            elif part_type == "redacted_thinking":
                found_reasoning = True
                data = part.get("data") or part.get("encrypted") or part.get("ciphertext")
                if isinstance(data, str) and data:
                    content_blocks.append({"type": "redacted_thinking", "data": data})
            elif part_type == "tool_use":
                fn = part.get("name")
                if not isinstance(fn, str):
                    fn = "tool"
                content_blocks.append(
                    {
                        "type": "tool_use",
                        "id": part.get("id") or f"call_{uuid.uuid4().hex[:24]}",
                        "caller": {"type": "direct"},
                        "name": fn,
                        "input": _safe_json_loads(part.get("input", {})),
                    }
                )
    elif isinstance(raw_content, str) and raw_content:
        text_parts.append(raw_content)

    if not found_reasoning:
        content_blocks.extend(_extract_reasoning_blocks(msg))

    if text_parts:
        content_blocks.append({"type": "text", "text": "\n".join(text_parts)})

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

    usage = _usage_from_upstream(openai_resp.get("usage", {}))
    stop_sequence = _extract_stop_sequence(openai_resp, choice)
    had_tool_calls = bool(tool_calls) or any(block.get("type") == "tool_use" for block in content_blocks)

    return {
        "id": openai_resp.get("id", f"gen-{uuid.uuid4()}"),
        "type": "message",
        "role": "assistant",
        "content": content_blocks or [{"type": "text", "text": ""}],
        "model": model_name,
        "stop_reason": _finish_reason_to_stop_reason(str(finish_reason or "stop"), had_tool_calls),
        "stop_sequence": stop_sequence,
        "usage": usage,
    }


async def _post_to_litellm(payload: Dict[str, Any]) -> Dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if LITELLM_API_KEY:
        headers["Authorization"] = f"Bearer {LITELLM_API_KEY}"

    delay = 0.4
    client = _get_http_client()
    for attempt in range(MAX_RETRIES):
        try:
            resp = await client.post(f"{LITELLM_BASE_URL}/chat/completions", headers=headers, json=payload)
            if resp.status_code in RETRYABLE_STATUS_CODES and attempt < MAX_RETRIES - 1:
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


def _extract_stream_delta_text(delta: Dict[str, Any]) -> str:
    content = delta.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def _extract_stream_delta_reasoning(delta: Dict[str, Any]) -> Tuple[str, str, str]:
    thinking_parts: List[str] = []
    signature = ""
    redacted_data = ""

    def consume(raw: Any) -> None:
        nonlocal signature, redacted_data

        if isinstance(raw, str):
            if raw:
                thinking_parts.append(raw)
            return

        if isinstance(raw, dict):
            raw_type = raw.get("type")
            if raw_type in {"redacted_thinking", "reasoning.encrypted", "reasoning.redacted"}:
                candidate = raw.get("data") or raw.get("encrypted") or raw.get("ciphertext")
                if isinstance(candidate, str) and candidate and not redacted_data:
                    redacted_data = candidate
                return

            text = raw.get("thinking")
            if not isinstance(text, str):
                text = raw.get("text")
            if isinstance(text, str) and text:
                thinking_parts.append(text)

            sig = raw.get("signature")
            if isinstance(sig, str) and sig and not signature:
                signature = sig
            return

        if isinstance(raw, list):
            for item in raw:
                consume(item)

    consume(delta.get("reasoning"))
    if not thinking_parts and not redacted_data:
        consume(delta.get("reasoning_content"))

    top_sig = delta.get("reasoning_signature")
    if isinstance(top_sig, str) and top_sig and not signature:
        signature = top_sig

    return "".join(thinking_parts), signature, redacted_data


async def _stream_litellm_to_anthropic(
    payload: Dict[str, Any],
    model_name: str,
    require_signed_thinking: bool,
) -> AsyncIterator[bytes]:
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "Accept-Encoding": "identity",
        "Cache-Control": "no-cache",
    }
    if LITELLM_API_KEY:
        headers["Authorization"] = f"Bearer {LITELLM_API_KEY}"

    client = _get_http_client()
    delay = 0.4

    for attempt in range(MAX_RETRIES):
        try:
            async with client.stream(
                "POST",
                f"{LITELLM_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
            ) as resp:
                if resp.status_code in RETRYABLE_STATUS_CODES and attempt < MAX_RETRIES - 1:
                    await resp.aread()
                    await asyncio.sleep(delay)
                    delay *= 2
                    continue

                if resp.status_code >= 400:
                    raw = await resp.aread()
                    try:
                        body: Any = json.loads(raw.decode("utf-8"))
                    except Exception:
                        body = {"error": {"message": raw.decode("utf-8", errors="replace"), "code": resp.status_code}}
                    raise UpstreamError(resp.status_code, body)

                message_id: Optional[str] = None
                stop_reason = "end_turn"
                usage: Dict[str, Any] = {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cache_creation_input_tokens": None,
                    "cache_read_input_tokens": 0,
                }

                next_content_index = 0
                active_text_index: Optional[int] = None
                active_thinking_index: Optional[int] = None
                active_redacted_index: Optional[int] = None
                pending_thinking_text = ""
                tool_blocks: Dict[int, Dict[str, Any]] = {}
                tool_argument_buffers: Dict[int, str] = {}
                had_tool_calls = False
                started = False
                data_lines: List[str] = []
                stop_sequence: Optional[str] = None

                async def close_text_block() -> AsyncIterator[bytes]:
                    nonlocal active_text_index
                    if active_text_index is not None:
                        yield _sse("content_block_stop", {"type": "content_block_stop", "index": active_text_index})
                        active_text_index = None

                async def close_thinking_block() -> AsyncIterator[bytes]:
                    nonlocal active_thinking_index
                    if active_thinking_index is not None:
                        yield _sse("content_block_stop", {"type": "content_block_stop", "index": active_thinking_index})
                        active_thinking_index = None

                async def close_redacted_block() -> AsyncIterator[bytes]:
                    nonlocal active_redacted_index
                    if active_redacted_index is not None:
                        yield _sse("content_block_stop", {"type": "content_block_stop", "index": active_redacted_index})
                        active_redacted_index = None

                async def emit_from_data(raw_data: str) -> AsyncIterator[bytes]:
                    nonlocal message_id, stop_reason, usage
                    nonlocal next_content_index, active_text_index, active_thinking_index, active_redacted_index
                    nonlocal pending_thinking_text
                    nonlocal had_tool_calls, started
                    nonlocal stop_sequence

                    if raw_data == "[DONE]":
                        return

                    try:
                        chunk = json.loads(raw_data)
                    except Exception:
                        return

                    if not started:
                        message_id = chunk.get("id") or f"gen-{uuid.uuid4()}"
                        yield _sse(
                            "message_start",
                            {
                                "type": "message_start",
                                "message": {
                                    "id": message_id,
                                    "type": "message",
                                    "role": "assistant",
                                    "content": [],
                                    "model": model_name,
                                    "stop_reason": None,
                                    "stop_sequence": None,
                                    "usage": {
                                        "input_tokens": 0,
                                        "output_tokens": 0,
                                        "cache_creation_input_tokens": None,
                                        "cache_read_input_tokens": 0,
                                    },
                                },
                            },
                        )
                        started = True

                    chunk_usage = chunk.get("usage")
                    if isinstance(chunk_usage, dict):
                        usage["input_tokens"] = int(chunk_usage.get("prompt_tokens", usage["input_tokens"]) or 0)
                        usage["output_tokens"] = int(chunk_usage.get("completion_tokens", usage["output_tokens"]) or 0)
                        if chunk_usage.get("cache_creation_input_tokens") is not None:
                            usage["cache_creation_input_tokens"] = int(chunk_usage.get("cache_creation_input_tokens") or 0)
                        if chunk_usage.get("cache_read_input_tokens") is not None:
                            usage["cache_read_input_tokens"] = int(chunk_usage.get("cache_read_input_tokens") or 0)

                    choices = chunk.get("choices")
                    if not isinstance(choices, list) or not choices:
                        return

                    choice0 = choices[0] if isinstance(choices[0], dict) else {}
                    finish_reason = choice0.get("finish_reason")
                    if isinstance(finish_reason, str) and finish_reason:
                        stop_reason = _finish_reason_to_stop_reason(finish_reason, had_tool_calls)
                    stop_sequence = _extract_stop_sequence(chunk, choice0) or stop_sequence

                    delta = choice0.get("delta")
                    if not isinstance(delta, dict):
                        return

                    thinking_piece, thinking_signature, redacted_piece = _extract_stream_delta_reasoning(delta)
                    if (
                        active_thinking_index is None
                        and thinking_signature
                        and pending_thinking_text
                    ):
                        async for event in close_text_block():
                            yield event
                        async for event in close_redacted_block():
                            yield event

                        active_thinking_index = next_content_index
                        next_content_index += 1
                        yield _sse(
                            "content_block_start",
                            {
                                "type": "content_block_start",
                                "index": active_thinking_index,
                                "content_block": {
                                    "type": "thinking",
                                    "thinking": "",
                                    "signature": thinking_signature,
                                },
                            },
                        )
                        yield _sse(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": active_thinking_index,
                                "delta": {"type": "thinking_delta", "thinking": pending_thinking_text},
                            },
                        )
                        pending_thinking_text = ""

                    if thinking_piece:
                        async for event in close_text_block():
                            yield event
                        async for event in close_redacted_block():
                            yield event

                        if active_thinking_index is None:
                            active_thinking_index = next_content_index
                            next_content_index += 1
                            yield _sse(
                                "content_block_start",
                                {
                                    "type": "content_block_start",
                                    "index": active_thinking_index,
                                    "content_block": {"type": "thinking", "thinking": "", "signature": thinking_signature},
                                },
                            )

                        merged_thinking_piece = f"{pending_thinking_text}{thinking_piece}" if pending_thinking_text else thinking_piece
                        pending_thinking_text = ""
                        yield _sse(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": active_thinking_index,
                                "delta": {"type": "thinking_delta", "thinking": merged_thinking_piece},
                            },
                        )

                    if redacted_piece and pending_thinking_text and active_thinking_index is None:
                        pending_thinking_text = ""

                    if redacted_piece:
                        async for event in close_text_block():
                            yield event
                        async for event in close_thinking_block():
                            yield event

                        if active_redacted_index is None:
                            active_redacted_index = next_content_index
                            next_content_index += 1
                            yield _sse(
                                "content_block_start",
                                {
                                    "type": "content_block_start",
                                    "index": active_redacted_index,
                                    "content_block": {"type": "redacted_thinking", "data": ""},
                                },
                            )

                        yield _sse(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": active_redacted_index,
                                "delta": {"type": "signature_delta", "signature": redacted_piece},
                            },
                        )

                    text_piece = _extract_stream_delta_text(delta)
                    if text_piece and pending_thinking_text and active_thinking_index is None:
                        pending_thinking_text = ""

                    if text_piece:
                        async for event in close_thinking_block():
                            yield event
                        async for event in close_redacted_block():
                            yield event

                        if active_text_index is None:
                            active_text_index = next_content_index
                            next_content_index += 1
                            yield _sse(
                                "content_block_start",
                                {
                                    "type": "content_block_start",
                                    "index": active_text_index,
                                    "content_block": {"type": "text", "text": ""},
                                },
                            )

                        yield _sse(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": active_text_index,
                                "delta": {"type": "text_delta", "text": text_piece},
                            },
                        )

                    tool_deltas = delta.get("tool_calls")
                    if isinstance(tool_deltas, list) and tool_deltas and pending_thinking_text and active_thinking_index is None:
                        pending_thinking_text = ""

                    if isinstance(tool_deltas, list) and tool_deltas:
                        async for event in close_text_block():
                            yield event
                        async for event in close_thinking_block():
                            yield event
                        async for event in close_redacted_block():
                            yield event

                        had_tool_calls = True
                        stop_reason = "tool_use"
                        for tc in tool_deltas:
                            if not isinstance(tc, dict):
                                continue
                            oai_index = int(tc.get("index", 0) or 0)
                            block = tool_blocks.get(oai_index)

                            raw_function = tc.get("function")
                            function: Dict[str, Any] = raw_function if isinstance(raw_function, dict) else {}

                            if block is None:
                                block = {
                                    "index": next_content_index,
                                    "id": tc.get("id") or f"call_{uuid.uuid4().hex[:24]}",
                                    "name": function.get("name") or "tool",
                                    "input": {},
                                }
                                tool_blocks[oai_index] = block
                                next_content_index += 1
                                tool_argument_buffers.setdefault(oai_index, "")

                                yield _sse(
                                    "content_block_start",
                                    {
                                        "type": "content_block_start",
                                        "index": block["index"],
                                        "content_block": {
                                            "type": "tool_use",
                                            "id": block["id"],
                                            "caller": {"type": "direct"},
                                            "name": block["name"],
                                            "input": {},
                                        },
                                    },
                                )

                            if tc.get("id"):
                                block["id"] = tc["id"]
                            if isinstance(function.get("name"), str) and function.get("name"):
                                block["name"] = function["name"]

                            arguments_piece = function.get("arguments")
                            if isinstance(arguments_piece, str) and arguments_piece:
                                tool_argument_buffers[oai_index] = tool_argument_buffers.get(oai_index, "") + arguments_piece
                                block["input"] = _safe_json_loads(tool_argument_buffers[oai_index])
                                yield _sse(
                                    "content_block_delta",
                                    {
                                        "type": "content_block_delta",
                                        "index": block["index"],
                                        "delta": {"type": "input_json_delta", "partial_json": arguments_piece},
                                    },
                                )

                async for line in resp.aiter_lines():
                    if line is None:
                        continue

                    raw_line = line.rstrip("\r")
                    if raw_line == "":
                        if data_lines:
                            raw_data = "\n".join(data_lines)
                            data_lines = []
                            async for event in emit_from_data(raw_data):
                                yield event
                        continue

                    if raw_line.startswith("data:"):
                        data_lines.append(raw_line[5:].strip())

                if data_lines:
                    raw_data = "\n".join(data_lines)
                    async for event in emit_from_data(raw_data):
                        yield event

                async for event in close_text_block():
                    yield event
                async for event in close_thinking_block():
                    yield event
                async for event in close_redacted_block():
                    yield event

                for block in tool_blocks.values():
                    yield _sse("content_block_stop", {"type": "content_block_stop", "index": block["index"]})

                yield _sse(
                    "message_delta",
                    {
                        "type": "message_delta",
                        "delta": {
                            "stop_reason": stop_reason,
                            "stop_sequence": stop_sequence,
                        },
                        "usage": usage,
                    },
                )
                yield _sse("message_stop", {"type": "message_stop"})
                yield b"event: data\ndata: [DONE]\n\n"
                return
        except UpstreamError:
            raise
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(delay)
                delay *= 2
                continue
            raise HTTPException(status_code=502, detail=f"Upstream network error: {exc}") from exc

    raise HTTPException(status_code=502, detail="Upstream request failed")


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
                "cache_read_input_tokens": 0,
            },
        },
    }
    yield _sse("message_start", message_start)

    for idx, block in enumerate(msg.get("content", [])):
        btype = block.get("type")
        if btype == "thinking":
            signature = block.get("signature") if isinstance(block.get("signature"), str) else ""
            yield _sse("content_block_start", {"type": "content_block_start", "index": idx, "content_block": {"type": "thinking", "thinking": "", "signature": signature}})
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
    user_agent: Optional[str] = Header(default=None),
) -> Any:
    # Headers kept for Anthropic compatibility; adapter currently ignores version/beta flags.
    _require_anthropic_version(anthropic_version)
    _validate_adapter_auth(x_api_key, authorization)

    body = await request.json()
    body["model"] = _resolve_model(body.get("model", ""))

    max_tokens = body.get("max_tokens", 1024)
    try:
        max_tokens_int = int(max_tokens)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="max_tokens must be a positive integer")
    if max_tokens_int <= 0:
        raise HTTPException(status_code=400, detail="max_tokens must be greater than 0")
    body["max_tokens"] = max_tokens_int

    require_signed_thinking = _should_require_signed_thinking(body, user_agent)

    request_id = uuid.uuid4().hex[:12]
    LOGGER.info(
        "messages.request %s",
        json.dumps(
            {
                "request_id": request_id,
                "model": body.get("model"),
                "stream": bool(body.get("stream")),
                "message_count": len(body.get("messages", [])) if isinstance(body.get("messages"), list) else 0,
                "content_types": _collect_message_content_types(body.get("messages")),
                "tools_count": len(body.get("tools", [])) if isinstance(body.get("tools"), list) else 0,
                "anthropic_beta": anthropic_beta,
                "user_agent": (user_agent or "")[:160],
                "require_signed_thinking": require_signed_thinking,
            },
            ensure_ascii=True,
        ),
    )

    openai_messages = _anthropic_messages_to_openai(body)
    openai_tools, openai_tool_choice = _convert_tools(body)

    payload: Dict[str, Any] = {
        "model": body["model"],
        "messages": openai_messages,
        "max_tokens": body.get("max_tokens", 1024),
        "temperature": body.get("temperature"),
        "top_p": body.get("top_p"),
        "stop": body.get("stop_sequences"),
        "stream": bool(body.get("stream")),
    }
    if DROP_UNSUPPORTED_PARAMS:
        payload = {k: v for k, v in payload.items() if v is not None}

    if openai_tools:
        payload["tools"] = openai_tools
    if openai_tool_choice is not None:
        payload["tool_choice"] = openai_tool_choice

    if body.get("thinking"):
        reasoning_effort = _reasoning_effort(body)
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort

    if body.get("stream"):
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
        try:
            return StreamingResponse(
                _stream_litellm_to_anthropic(payload, body["model"], require_signed_thinking),
                media_type="text/event-stream",
            )
        except UpstreamError as exc:
            return _error_response(exc.status_code, exc.body)

    payload["stream"] = False
    try:
        upstream = await _post_to_litellm(payload)
    except UpstreamError as exc:
        return _error_response(exc.status_code, exc.body)

    anthropic_msg = _openai_to_anthropic_response(upstream, body["model"])
    LOGGER.info(
        "messages.response %s",
        json.dumps(
            {
                "request_id": request_id,
                "stream": False,
                "content_types": [
                    block.get("type")
                    for block in anthropic_msg.get("content", [])
                    if isinstance(block, dict)
                ],
                "stop_reason": anthropic_msg.get("stop_reason"),
            },
            ensure_ascii=True,
        ),
    )

    return anthropic_msg


@app.get("/v1/models")
def list_models(
    x_api_key: Optional[str] = Header(default=None),
    authorization: Optional[str] = Header(default=None),
    anthropic_version: Optional[str] = Header(default=None),
    anthropic_beta: Optional[str] = Header(default=None),
    after_id: Optional[str] = Query(default=None),
    before_id: Optional[str] = Query(default=None),
    limit: int = Query(default=20, ge=1, le=1000),
) -> Dict[str, Any]:
    _ = anthropic_beta
    _require_anthropic_version(anthropic_version)
    _validate_adapter_auth(x_api_key, authorization)

    models = _get_model_catalog()
    start = 0
    end = len(models)

    if after_id:
        for index, model in enumerate(models):
            if model.get("id") == after_id:
                start = index + 1
                break

    if before_id:
        for index, model in enumerate(models):
            if model.get("id") == before_id:
                end = index
                break

    if before_id and after_id:
        page = models[start:end][:limit]
    elif before_id:
        page = models[max(0, end - limit):end]
    elif after_id:
        page = models[start:start + limit]
    else:
        page = models[:limit]

    has_more = False
    if page:
        last_index = models.index(page[-1])
        has_more = last_index < len(models) - 1

    return {
        "data": page,
        "first_id": page[0]["id"] if page else "",
        "has_more": has_more,
        "last_id": page[-1]["id"] if page else "",
    }


@app.get("/v1/models/{model_id}")
def get_model(
    model_id: str,
    x_api_key: Optional[str] = Header(default=None),
    authorization: Optional[str] = Header(default=None),
    anthropic_version: Optional[str] = Header(default=None),
    anthropic_beta: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    _ = anthropic_beta
    _require_anthropic_version(anthropic_version)
    _validate_adapter_auth(x_api_key, authorization)

    model = _find_model(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    return model


@app.post("/v1/messages/count_tokens")
async def count_tokens(
    request: Request,
    x_api_key: Optional[str] = Header(default=None),
    authorization: Optional[str] = Header(default=None),
    anthropic_version: Optional[str] = Header(default=None),
    anthropic_beta: Optional[str] = Header(default=None),
) -> Dict[str, int]:
    _ = anthropic_beta
    _require_anthropic_version(anthropic_version)
    _validate_adapter_auth(x_api_key, authorization)

    body = await request.json()
    body["model"] = _resolve_model(body.get("model", ""))

    messages = body.get("messages")
    if not isinstance(messages, list):
        raise HTTPException(status_code=400, detail="messages must be an array")

    return {"input_tokens": _count_tokens_for_request(body)}
