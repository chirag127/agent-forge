"""Observability — structlog setup + OpenTelemetry spans (console exporter)."""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

_configured = False
_tracer_provider: TracerProvider | None = None


def configure(level: str = "INFO", otel: bool = True, json_logs: bool = False) -> None:
    global _configured, _tracer_provider
    if _configured:
        return

    # stdlib logging
    logging.basicConfig(stream=sys.stderr, level=getattr(logging, level.upper(), logging.INFO))

    # structlog
    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]
    if json_logs:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # OpenTelemetry
    if otel:
        resource = Resource.create({"service.name": "agent-forge"})
        _tracer_provider = TracerProvider(resource=resource)
        _tracer_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        trace.set_tracer_provider(_tracer_provider)

    _configured = True


def get_tracer(name: str = "agent-forge") -> trace.Tracer:
    configure()
    return trace.get_tracer(name)


def span(name: str, **attrs: Any) -> Any:
    """Context manager for an OTel span."""
    tracer = get_tracer()
    s = tracer.start_as_current_span(name)
    return s
