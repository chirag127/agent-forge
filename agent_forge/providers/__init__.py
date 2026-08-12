"""LLM provider — httpx-based, keyless. kilo → pollinations → g4f fallback."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx
import structlog

log = structlog.get_logger(__name__)

# ── Provider configs ──────────────────────────────────────────────────────────

_KILO_URL = "https://api.kilo.ai/api/gateway/v1/chat/completions"
_KILO_MODEL = "kilo-auto/free"

_POLLINATIONS_POST_URL = "https://text.pollinations.ai/openai"
_POLLINATIONS_GET_URL = "https://text.pollinations.ai/{prompt}?model=openai"
_POLLINATIONS_MODEL = "openai"

_DEFAULT_TIMEOUT = 60.0
_DEFAULT_MODEL = "kilo-auto/free"


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class LLMResponse:
    content: str
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0


@dataclass
class LLMConfig:
    model: str = _DEFAULT_MODEL
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: float = _DEFAULT_TIMEOUT


def _msgs_to_list(messages: list[Message]) -> list[dict[str, str]]:
    return [{"role": m.role, "content": m.content} for m in messages]


def _approx_tokens(messages: list[Message], content: str) -> tuple[int, int]:
    in_tok = sum(len(m.content) for m in messages) // 4
    out_tok = len(content) // 4
    return in_tok, out_tok


def _extract_content(body: dict[str, Any]) -> str:
    return body["choices"][0]["message"]["content"]


# ── Provider functions (sync httpx) ──────────────────────────────────────────


def _try_kilo(messages: list[Message], config: LLMConfig) -> str:
    payload = {
        "model": _KILO_MODEL,
        "messages": _msgs_to_list(messages),
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }
    resp = httpx.post(
        _KILO_URL,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=config.timeout,
    )
    resp.raise_for_status()
    return _extract_content(resp.json())


def _try_pollinations_post(messages: list[Message], config: LLMConfig) -> str:
    payload = {
        "model": _POLLINATIONS_MODEL,
        "messages": _msgs_to_list(messages),
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }
    resp = httpx.post(
        _POLLINATIONS_POST_URL,
        json=payload,
        headers={"Content-Type": "application/json", "Referer": "https://oriz.in"},
        timeout=config.timeout,
    )
    resp.raise_for_status()
    return _extract_content(resp.json())


def _try_pollinations_get(messages: list[Message], config: LLMConfig) -> str:
    # Flatten messages into a single prompt string
    prompt = " ".join(f"[{m.role}] {m.content}" for m in messages)
    # URL-encode prompt
    import urllib.parse  # noqa: PLC0415
    encoded = urllib.parse.quote(prompt[:1000])
    url = _POLLINATIONS_GET_URL.format(prompt=encoded)
    resp = httpx.get(
        url,
        headers={"User-Agent": "agent-forge/0.1"},
        timeout=config.timeout,
    )
    resp.raise_for_status()
    # GET endpoint returns plain text (not JSON)
    return resp.text.strip()


def _try_g4f(messages: list[Message], config: LLMConfig) -> str:
    try:
        import g4f  # noqa: PLC0415
    except ImportError as e:
        raise RuntimeError("g4f not installed") from e
    msgs = _msgs_to_list(messages)
    result = g4f.ChatCompletion.create(model="gpt-4o-mini", messages=msgs, stream=False)
    return result if isinstance(result, str) else str(result)


# Ordered provider chain: (name, fn)
_PROVIDERS = [
    ("kilo", _try_kilo),
    ("pollinations_post", _try_pollinations_post),
    ("pollinations_get", _try_pollinations_get),
    ("g4f", _try_g4f),
]


class LLMProvider:
    """Single interface with kilo → pollinations → g4f failover."""

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig()

    def complete(self, messages: list[Message], **_kwargs: Any) -> LLMResponse:
        last_err: Exception | None = None
        for name, fn in _PROVIDERS:
            try:
                t0 = time.monotonic()
                content = fn(messages, self.config)
                latency = (time.monotonic() - t0) * 1000
                in_tok, out_tok = _approx_tokens(messages, content)
                log.debug("llm.complete", provider=name, latency_ms=round(latency, 1))
                return LLMResponse(
                    content=content,
                    provider=name,
                    model=self.config.model,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    latency_ms=round(latency, 1),
                )
            except Exception as e:  # noqa: BLE001
                log.warning("llm.provider_failed", provider=name, error=str(e)[:120])
                last_err = e
        raise RuntimeError(f"All providers failed. Last: {last_err}")

    async def acomplete(self, messages: list[Message], **kwargs: Any) -> LLMResponse:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.complete(messages, **kwargs))

    async def astream(self, messages: list[Message], **_kwargs: Any) -> AsyncIterator[str]:
        resp = await self.acomplete(messages)
        # simulate streaming by yielding whole content
        yield resp.content
