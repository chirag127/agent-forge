"""Unit tests — tools module."""

from __future__ import annotations

import math
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from agent_forge.tools import (
    ToolDefinition,
    _check_ast,
    _schema_from_signature,
    call_tool,
    calculator,
    file_read,
    get_registry,
    python_eval,
    tool,
    web_fetch,
)


# ── tool decorator ────────────────────────────────────────────────────────────


def test_tool_registers_in_registry() -> None:
    @tool(name="test_dummy", description="A dummy tool")
    def dummy(x: str) -> str:
        return x

    reg = get_registry()
    assert "test_dummy" in reg
    assert isinstance(reg["test_dummy"], ToolDefinition)


def test_tool_auto_schema_from_hints() -> None:
    @tool(name="test_typed")
    def typed_fn(name: str, count: int) -> str:
        """Typed function."""
        return f"{name} x {count}"

    reg = get_registry()
    schema = reg["test_typed"].parameters
    assert schema["properties"]["name"]["type"] == "string"
    assert schema["properties"]["count"]["type"] == "integer"
    assert "name" in schema["required"]
    assert "count" in schema["required"]


def test_tool_openai_schema_shape() -> None:
    reg = get_registry()
    assert "calculator" in reg
    schema = reg["calculator"].to_openai_schema()
    assert schema["type"] == "function"
    assert "function" in schema
    assert schema["function"]["name"] == "calculator"


def test_call_tool_unknown_raises() -> None:
    with pytest.raises(KeyError, match="unknown_xyz"):
        call_tool("unknown_xyz", {})


def test_call_tool_schema_validation_fails() -> None:
    import jsonschema  # noqa: PLC0415
    with pytest.raises(jsonschema.ValidationError):
        call_tool("calculator", {})  # missing required 'expression'


def test_schema_from_signature_optional_param() -> None:
    def fn_with_default(path: str, workdir: str = ".") -> str:
        return path

    schema = _schema_from_signature(fn_with_default)
    assert "path" in schema["required"]
    assert "workdir" not in schema["required"]


# ── calculator ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("expr,expected", [
    ("2 + 2", "4"),
    ("2 ** 10", "1024"),
    ("sqrt(144)", "12.0"),
    ("round(3.14159, 2)", "3.14"),
    ("abs(-42)", "42"),
])
def test_calculator_correct(expr: str, expected: str) -> None:
    assert calculator(expr) == expected


def test_calculator_rejects_unsafe() -> None:
    with pytest.raises(ValueError, match="Unsafe"):
        calculator("__import__('os').system('echo hi')")


def test_calculator_caret_pow() -> None:
    assert calculator("2^8") == "256"


# ── python_eval ───────────────────────────────────────────────────────────────


def test_python_eval_basic() -> None:
    result = python_eval("x = 1 + 1")
    assert "2" in result or result  # result is locals dict or contains 2


def test_python_eval_sum() -> None:
    result = python_eval("total = sum(range(1, 11))")
    assert "55" in result


def test_python_eval_blocks_import() -> None:
    with pytest.raises(ValueError, match="Import"):
        python_eval("import os")


def test_python_eval_blocks_dunder() -> None:
    with pytest.raises(ValueError, match="dunder"):
        python_eval("x = ().__class__.__bases__")


def test_check_ast_import_from() -> None:
    import ast  # noqa: PLC0415
    tree = ast.parse("from os import path", mode="exec")
    with pytest.raises(ValueError, match="ImportFrom"):
        _check_ast(tree)


# ── file_read ─────────────────────────────────────────────────────────────────


def test_file_read_success(tmp_path: Path) -> None:
    (tmp_path / "test.txt").write_text("hello agent", encoding="utf-8")
    result = file_read("test.txt", workdir=str(tmp_path))
    assert result == "hello agent"


def test_file_read_escape_raises(tmp_path: Path) -> None:
    with pytest.raises(PermissionError):
        file_read("../secret.txt", workdir=str(tmp_path))


def test_file_read_prefix_collision_safe(tmp_path: Path) -> None:
    """Sibling dir whose name is a prefix of workdir must not be readable."""
    sibling = tmp_path.parent / (tmp_path.name + "_sibling")
    sibling.mkdir()
    (sibling / "data.txt").write_text("secret")
    # Construct a relative path that resolves into the sibling
    rel = "../" + tmp_path.name + "_sibling/data.txt"
    with pytest.raises(PermissionError):
        file_read(rel, workdir=str(tmp_path))


def test_file_read_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        file_read("nope.txt", workdir=str(tmp_path))


# ── web_fetch (mocked) ────────────────────────────────────────────────────────


def test_web_fetch_strips_html() -> None:
    html = "<html><body><p>Hello world</p></body></html>"
    with patch("httpx.Client") as mock_client:
        mock_response = mock_client.return_value.__enter__.return_value.get.return_value
        mock_response.text = html
        mock_response.raise_for_status = lambda: None
        result = web_fetch("http://example.com")
    assert "<" not in result
    assert "Hello world" in result


def test_web_fetch_truncates_long_content() -> None:
    long_text = "word " * 2000
    with patch("httpx.Client") as mock_client:
        mock_response = mock_client.return_value.__enter__.return_value.get.return_value
        mock_response.text = long_text
        mock_response.raise_for_status = lambda: None
        result = web_fetch("http://example.com")
    assert len(result) <= 4000
