"""Unit tests — LLM provider (mocked httpx)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agent_forge.providers import (
    LLMConfig,
    LLMProvider,
    LLMResponse,
    Message,
    _approx_tokens,
    _msgs_to_list,
    _try_kilo,
    _try_pollinations_get,
    _try_pollinations_post,
)


def _make_openai_resp(content: str) -> dict:
    return {"choices": [{"message": {"content": content}}]}


# ── helpers ───────────────────────────────────────────────────────────────────


def test_msgs_to_list() -> None:
    msgs = [Message("system", "You are an agent."), Message("user", "Hello")]
    result = _msgs_to_list(msgs)
    assert result == [
        {"role": "system", "content": "You are an agent."},
        {"role": "user", "content": "Hello"},
    ]


def test_approx_tokens() -> None:
    msgs = [Message("user", "abcd")]  # 4 chars → 1 token
    in_tok, out_tok = _approx_tokens(msgs, "abcdefgh")  # 8 chars → 2 tokens
    assert in_tok == 1
    assert out_tok == 2


# ── kilo provider ─────────────────────────────────────────────────────────────


def test_try_kilo_success() -> None:
    mock_resp = MagicMock()
    mock_resp.json.return_value = _make_openai_resp("hello from kilo")
    mock_resp.raise_for_status = lambda: None

    with patch("httpx.post", return_value=mock_resp) as mock_post:
        result = _try_kilo([Message("user", "hi")], LLMConfig())

    assert result == "hello from kilo"
    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args
    assert "kilo.ai" in call_kwargs[0][0]


def test_try_kilo_raises_on_http_error() -> None:
    import httpx  # noqa: PLC0415

    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "404", request=MagicMock(), response=MagicMock()
    )

    with patch("httpx.post", return_value=mock_resp):
        with pytest.raises(httpx.HTTPStatusError):
            _try_kilo([Message("user", "hi")], LLMConfig())


# ── pollinations providers ─────────────────────────────────────────────────────


def test_try_pollinations_post_success() -> None:
    mock_resp = MagicMock()
    mock_resp.json.return_value = _make_openai_resp("pollen answer")
    mock_resp.raise_for_status = lambda: None

    with patch("httpx.post", return_value=mock_resp):
        result = _try_pollinations_post([Message("user", "hi")], LLMConfig())

    assert result == "pollen answer"


def test_try_pollinations_get_success() -> None:
    mock_resp = MagicMock()
    mock_resp.text = "  four  "
    mock_resp.raise_for_status = lambda: None

    with patch("httpx.get", return_value=mock_resp):
        result = _try_pollinations_get([Message("user", "What is 2+2?")], LLMConfig())

    assert result == "four"


# ── LLMProvider failover ──────────────────────────────────────────────────────


def test_provider_uses_kilo_first() -> None:
    mock_resp = MagicMock()
    mock_resp.json.return_value = _make_openai_resp("kilo answer")
    mock_resp.raise_for_status = lambda: None

    with patch("httpx.post", return_value=mock_resp):
        provider = LLMProvider(LLMConfig())
        resp = provider.complete([Message("user", "test")])

    assert resp.provider == "kilo"
    assert resp.content == "kilo answer"
    assert isinstance(resp, LLMResponse)


def test_provider_falls_back_to_pollinations_post_on_kilo_fail() -> None:
    import httpx as _httpx  # noqa: PLC0415

    post_call_count = 0

    def mock_post(url: str, **kwargs):  # noqa: ANN001
        nonlocal post_call_count
        post_call_count += 1
        if "kilo.ai" in url:
            raise _httpx.ConnectError("kilo down")
        # pollinations_post
        m = MagicMock()
        m.json.return_value = _make_openai_resp("pollinations answer")
        m.raise_for_status = lambda: None
        return m

    with patch("httpx.post", side_effect=mock_post):
        provider = LLMProvider(LLMConfig())
        resp = provider.complete([Message("user", "test")])

    assert resp.provider == "pollinations_post"
    assert resp.content == "pollinations answer"


def test_provider_falls_back_to_get_when_post_fails() -> None:
    import httpx as _httpx  # noqa: PLC0415

    mock_get = MagicMock()
    mock_get.text = "get answer"
    mock_get.raise_for_status = lambda: None

    with patch("httpx.post", side_effect=_httpx.ConnectError("all posts down")):
        with patch("httpx.get", return_value=mock_get):
            provider = LLMProvider(LLMConfig())
            resp = provider.complete([Message("user", "test")])

    assert resp.provider == "pollinations_get"
    assert resp.content == "get answer"


def test_provider_raises_when_all_fail() -> None:
    import httpx as _httpx  # noqa: PLC0415

    with patch("httpx.post", side_effect=_httpx.ConnectError("down")):
        with patch("httpx.get", side_effect=_httpx.ConnectError("down")):
            with patch.dict("sys.modules", {"g4f": None}):
                provider = LLMProvider(LLMConfig())
                with pytest.raises(RuntimeError, match="All providers failed"):
                    provider.complete([Message("user", "test")])


def test_llm_config_defaults() -> None:
    cfg = LLMConfig()
    assert cfg.temperature == 0.7
    assert cfg.max_tokens == 2048
    assert cfg.timeout == 60.0
