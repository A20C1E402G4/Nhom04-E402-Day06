"""Centralized configuration: paths, env vars, and cached JSON loaders."""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
LOGS_DIR: Path = PROJECT_ROOT / "logs"
PROMPTS_DIR: Path = Path(__file__).resolve().parent / "prompts"

# LLM configuration
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")
LLM_TEMPERATURE: float = float(os.getenv("VSSA_TEMPERATURE", "0"))

# Agent guardrails
MAX_TOOL_TURNS: int = int(os.getenv("VSSA_MAX_TOOL_TURNS", "3"))

# Checkpointing
CHECKPOINT_DB_PATH: Path = PROJECT_ROOT / "vssa_state.sqlite"


def ensure_logs_dir() -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    return LOGS_DIR


@lru_cache(maxsize=8)
def load_json(filename: str) -> dict[str, Any]:
    """Load a JSON file from the data directory (cached)."""
    path = DATA_DIR / filename
    if not path.exists():
        # Fallback for missing files to prevent crashing the agent
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def load_vehicles() -> dict[str, Any]:
    return load_json("vehicles.json")


def load_finance() -> dict[str, Any]:
    return load_json("finance.json")


def load_showrooms() -> dict[str, Any]:
    return load_json("showrooms.json")


def load_charging_stations() -> dict[str, Any]:
    return load_json("charging_stations.json")


def load_system_prompt() -> str:
    return (PROMPTS_DIR / "system_prompt.txt").read_text(encoding="utf-8")
