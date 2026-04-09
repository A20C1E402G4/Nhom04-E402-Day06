"""Append-only JSONL loggers feeding the Data Flywheel."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from vssa_agent.config import ensure_logs_dir

_AGENT_LOG = "agent.jsonl"
_CORRECTIONS_LOG = "corrections.jsonl"
_FAILED_INTENTS_LOG = "failed_intents.jsonl"
_BOOKINGS_LOG = "bookings.jsonl"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append(filename: str, record: dict[str, Any]) -> None:
    logs_dir = ensure_logs_dir()
    record = {"ts": _now_iso(), **record}
    with (logs_dir / filename).open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def log_node(
    *,
    node_name: str,
    elapsed_ms: int,
    tool_calls: list[str] | None = None,
    tokens_in: int | None = None,
    tokens_out: int | None = None,
) -> None:
    """Record a node-level execution event (latency, tool calls, tokens)."""
    _append(
        _AGENT_LOG,
        {
            "node": node_name,
            "elapsed_ms": elapsed_ms,
            "tool_calls": tool_calls or [],
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
        },
    )


def log_correction(event: dict[str, Any]) -> None:
    """Record a `user_correction` event from the Streamlit UI slider."""
    _append(_CORRECTIONS_LOG, {"event": "user_correction", **event})


def log_failed_intent(query: str, reason: str = "") -> None:
    """Record a query that the agent could not classify into any tool."""
    _append(_FAILED_INTENTS_LOG, {"query": query, "reason": reason})


def log_booking(record: dict[str, Any]) -> None:
    """Persist a confirmed test-drive booking (lead capture)."""
    _append(_BOOKINGS_LOG, {"event": "test_drive_booked", **record})
