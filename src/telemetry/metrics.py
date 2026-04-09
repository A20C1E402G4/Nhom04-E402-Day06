"""In-memory product/business metrics tracker."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from vssa_agent.config import ensure_logs_dir


@dataclass
class Metrics:
    sessions: int = 0
    leads: int = 0
    tool_calls_total: int = 0
    tool_calls_failed: int = 0
    corrections: int = 0
    fallbacks: int = 0
    extra: dict[str, int] = field(default_factory=dict)

    def tool_success_rate(self) -> float:
        if self.tool_calls_total == 0:
            return 1.0
        return 1.0 - self.tool_calls_failed / self.tool_calls_total

    def lead_gen_recall(self) -> float:
        if self.sessions == 0:
            return 0.0
        return self.leads / self.sessions


_METRICS = Metrics()


def get_metrics() -> Metrics:
    return _METRICS


def incr_session() -> None:
    _METRICS.sessions += 1


def incr_lead() -> None:
    _METRICS.leads += 1


def incr_tool_call(failed: bool = False) -> None:
    _METRICS.tool_calls_total += 1
    if failed:
        _METRICS.tool_calls_failed += 1


def incr_correction() -> None:
    _METRICS.corrections += 1


def incr_fallback() -> None:
    _METRICS.fallbacks += 1


def persist() -> None:
    """Write the current metric snapshot to logs/metrics.json."""
    logs_dir = ensure_logs_dir()
    payload = asdict(_METRICS)
    payload["tool_success_rate"] = _METRICS.tool_success_rate()
    payload["lead_gen_recall"] = _METRICS.lead_gen_recall()
    (logs_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
