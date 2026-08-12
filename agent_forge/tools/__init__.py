"""Tool system — @tool decorator, JSON-schema validation, registry."""

from __future__ import annotations

import ast
import builtins
import inspect
import json
import math
import operator
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import jsonschema
import structlog

log = structlog.get_logger(__name__)

_REGISTRY: dict[str, "ToolDefinition"] = {}


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    fn: Callable[..., Any]

    def to_openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def tool(
    name: str | None = None,
    description: str = "",
    parameters: dict[str, Any] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that registers a callable as an agent tool."""

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        tool_name = name or fn.__name__
        tool_desc = description or (fn.__doc__ or "").strip()

        # Auto-derive JSON schema from type hints if not provided
        schema = parameters or _schema_from_signature(fn)

        defn = ToolDefinition(
            name=tool_name,
            description=tool_desc,
            parameters=schema,
            fn=fn,
        )
        _REGISTRY[tool_name] = defn
        fn._tool_definition = defn  # type: ignore[attr-defined]
        log.debug("tool.registered", name=tool_name)
        return fn

    return decorator


def _schema_from_signature(fn: Callable[..., Any]) -> dict[str, Any]:
    """Derive a simple JSON Schema object from function signature type hints."""
    sig = inspect.signature(fn)
    props: dict[str, Any] = {}
    required: list[str] = []
    _py_to_json = {
        "str": "string",
        "int": "integer",
        "float": "number",
        "bool": "boolean",
        "list": "array",
        "dict": "object",
    }
    for pname, param in sig.parameters.items():
        ann = param.annotation
        type_name = "string"
        if ann != inspect.Parameter.empty:
            raw = ann.__name__ if hasattr(ann, "__name__") else str(ann)
            type_name = _py_to_json.get(raw, "string")
        props[pname] = {"type": type_name}
        if param.default is inspect.Parameter.empty:
            required.append(pname)
    return {"type": "object", "properties": props, "required": required}


def get_registry() -> dict[str, ToolDefinition]:
    return dict(_REGISTRY)


def call_tool(name: str, arguments: str | dict[str, Any]) -> Any:
    """Validate args against schema, then call the tool."""
    if name not in _REGISTRY:
        raise KeyError(f"Tool '{name}' not registered")
    defn = _REGISTRY[name]
    args: dict[str, Any] = json.loads(arguments) if isinstance(arguments, str) else arguments
    jsonschema.validate(args, defn.parameters)
    log.info("tool.call", name=name, args=args)
    result = defn.fn(**args)
    log.info("tool.result", name=name, result=str(result)[:200])
    return result


# ── Built-in tools ────────────────────────────────────────────────────────────


@tool(
    name="web_fetch",
    description="Fetch the text content of a URL via HTTP GET.",
    parameters={
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "URL to fetch"},
            "timeout": {"type": "number", "description": "Timeout in seconds", "default": 15},
        },
        "required": ["url"],
    },
)
def web_fetch(url: str, timeout: float = 15.0) -> str:
    """Fetch URL and return stripped text (truncated at 4 000 chars)."""
    headers = {"User-Agent": "agent-forge/0.1 (+https://github.com/chirag127/agent-forge)"}
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        resp = client.get(url, headers=headers)
        resp.raise_for_status()
        text = resp.text
    # strip HTML tags crudely
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text[:4000]


@dataclass
class _SafeEvalConfig:
    allowed_builtins: set[str] = field(default_factory=lambda: {
        "abs", "all", "any", "bin", "bool", "chr", "dict", "divmod",
        "enumerate", "filter", "float", "format", "frozenset", "hex",
        "int", "isinstance", "issubclass", "len", "list", "map", "max",
        "min", "oct", "ord", "pow", "range", "repr", "reversed",
        "round", "set", "slice", "sorted", "str", "sum", "tuple",
        "type", "zip",
    })
    banned_nodes: tuple[type, ...] = field(default_factory=lambda: (
        ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal,
    ))


_EVAL_CONFIG = _SafeEvalConfig()


def _check_ast(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if isinstance(node, _EVAL_CONFIG.banned_nodes):
            raise ValueError(f"Forbidden AST node: {type(node).__name__}")
        # block attribute access to dunder names
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError(f"Forbidden dunder access: {node.attr}")


@tool(
    name="python_eval",
    description="Safely evaluate a Python expression or short script. No imports allowed.",
    parameters={
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python code to evaluate"},
        },
        "required": ["code"],
    },
)
def python_eval(code: str) -> str:
    """Sandboxed eval — no imports, no dunder access, limited builtins."""
    tree = ast.parse(code, mode="exec")
    _check_ast(tree)
    safe_globals = {
        "__builtins__": {k: getattr(builtins, k) for k in _EVAL_CONFIG.allowed_builtins if hasattr(builtins, k)},
        "math": math,
        "operator": operator,
    }
    local_ns: dict[str, Any] = {}
    exec(compile(tree, "<sandbox>", "exec"), safe_globals, local_ns)  # noqa: S102
    # return last expression value if present
    result = local_ns.get("_result", local_ns)
    return str(result)


@tool(
    name="file_read",
    description="Read a text file within the allowed working directory.",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Relative path inside workdir"},
            "workdir": {"type": "string", "description": "Absolute base directory", "default": "."},
        },
        "required": ["path"],
    },
)
def file_read(path: str, workdir: str = ".") -> str:
    """Read file, restricted to workdir via path resolution check."""
    base = Path(workdir).resolve()
    target = (base / path).resolve()
    # Use is_relative_to (Python 3.9+) to avoid prefix-collision bugs
    try:
        target.relative_to(base)
    except ValueError:
        raise PermissionError(f"Path '{path}' escapes workdir '{workdir}'")
    if not target.exists():
        raise FileNotFoundError(f"File not found: {target}")
    return target.read_text(encoding="utf-8")[:8000]


@tool(
    name="calculator",
    description="Evaluate a safe mathematical expression. Returns a number.",
    parameters={
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "Math expression, e.g. '2 ** 10 + sqrt(16)'"},
        },
        "required": ["expression"],
    },
)
def calculator(expression: str) -> str:
    """Evaluate math expression using ast.literal_eval-safe subset + math module."""
    allowed_names = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
    allowed_names["abs"] = abs
    allowed_names["round"] = round
    # only allow safe chars
    if re.search(r"[^0-9+\-*/().,%^ a-zA-Z_]", expression):
        raise ValueError(f"Unsafe characters in expression: {expression}")
    # replace ^ with ** for convenience
    expr = expression.replace("^", "**")
    result = eval(expr, {"__builtins__": {}}, allowed_names)  # noqa: S307
    return str(result)
