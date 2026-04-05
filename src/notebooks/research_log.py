"""Research log — append-only JSONL log for notebook experiment tracking.

Each notebook session appends structured entries recording what was analysed,
what Claude interpreted, and any researcher notes.

Usage:
    from src.notebooks.research_log import ResearchLog
    from src.utils.config import load_config

    log = ResearchLog(load_config())
    log.log_entry("01_data_exploration", "feature_analysis", data, interpretation)
    recent = log.load_recent(5)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_LOG_PATH = "data/processed/research_log.jsonl"


class ResearchLog:
    """Append-only JSONL research log.

    Args:
        config: Project configuration dict (from load_config()).
    """

    def __init__(self, config: dict[str, Any]) -> None:
        nb_config = config.get("notebooks", {})
        self._log_path = Path(nb_config.get("log_path", _DEFAULT_LOG_PATH))
        logger.debug("ResearchLog path: %s", self._log_path)

    def log_entry(
        self,
        notebook: str,
        task: str,
        data_summary: dict[str, Any],
        interpretation: dict[str, Any],
        notes: str = "",
    ) -> None:
        """Append a research log entry.

        Args:
            notebook: Notebook identifier (e.g. '01_data_exploration').
            task: Interpretation task (e.g. 'feature_analysis').
            data_summary: Summary of the quantitative data analysed.
            interpretation: Claude interpretation result dict.
            notes: Optional researcher notes.
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "notebook": notebook,
            "task": task,
            "data_summary": data_summary,
            "interpretation": interpretation,
            "notes": notes,
        }

        # Ensure parent directory exists.
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")

        logger.debug("Logged research entry: notebook=%s, task=%s", notebook, task)

    def load_recent(self, n: int = 10) -> list[dict[str, Any]]:
        """Load the most recent n log entries.

        Args:
            n: Maximum number of entries to return.

        Returns:
            List of log entry dicts, most recent last. Empty list if
            the log file does not exist.
        """
        if not self._log_path.exists():
            return []

        entries: list[dict[str, Any]] = []
        with open(self._log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning("Skipping malformed log line")

        return entries[-n:]

    def summarise_session(self, notebook: str) -> dict[str, Any]:
        """Return all entries for the given notebook from today.

        Args:
            notebook: Notebook identifier to filter by.

        Returns:
            Dict with 'notebook', 'date', 'entries' (list), and 'count'.
        """
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        all_entries = self.load_recent(1000)  # Load a generous window.

        matching = [
            e for e in all_entries
            if e.get("notebook") == notebook
            and e.get("timestamp", "").startswith(today_str)
        ]

        return {
            "notebook": notebook,
            "date": today_str,
            "entries": matching,
            "count": len(matching),
        }
