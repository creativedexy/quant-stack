"""Tests for config loader, including dynamic date resolution."""

from __future__ import annotations

import datetime
import textwrap
from pathlib import Path

import pytest

from src.utils.config import load_config


@pytest.fixture()
def tmp_config(tmp_path: Path):
    """Write a minimal YAML config and return (path, write_helper)."""

    def _write(yaml_text: str) -> Path:
        p = tmp_path / "settings.yaml"
        p.write_text(textwrap.dedent(yaml_text), encoding="utf-8")
        return p

    return _write


class TestDynamicEndDate:
    """data.end_date is resolved dynamically at load time."""

    def test_today_resolved_to_current_date(self, tmp_config) -> None:
        path = tmp_config("""\
            general:
              base_currency: GBP
              timezone: Europe/London
              log_level: INFO
              data_dir: data
              random_seed: 42
            universe:
              name: test
              tickers: [A]
              benchmark: "^FTSE"
            data:
              source: synthetic
              start_date: "2020-01-01"
              end_date: "today"
              interval: "1d"
              fields: [Close]
        """)
        cfg = load_config(path)
        resolved = cfg["data"]["end_date"]
        today = datetime.date.today().isoformat()
        assert resolved == today

    def test_stale_date_replaced(self, tmp_config) -> None:
        stale = (datetime.date.today() - datetime.timedelta(days=60)).isoformat()
        path = tmp_config(f"""\
            general:
              base_currency: GBP
              timezone: Europe/London
              log_level: INFO
              data_dir: data
              random_seed: 42
            universe:
              name: test
              tickers: [A]
              benchmark: "^FTSE"
            data:
              source: synthetic
              start_date: "2020-01-01"
              end_date: "{stale}"
              interval: "1d"
              fields: [Close]
        """)
        cfg = load_config(path)
        resolved = cfg["data"]["end_date"]
        today = datetime.date.today().isoformat()
        assert resolved == today

    def test_recent_date_preserved(self, tmp_config) -> None:
        recent = (datetime.date.today() - datetime.timedelta(days=5)).isoformat()
        path = tmp_config(f"""\
            general:
              base_currency: GBP
              timezone: Europe/London
              log_level: INFO
              data_dir: data
              random_seed: 42
            universe:
              name: test
              tickers: [A]
              benchmark: "^FTSE"
            data:
              source: synthetic
              start_date: "2020-01-01"
              end_date: "{recent}"
              interval: "1d"
              fields: [Close]
        """)
        cfg = load_config(path)
        assert cfg["data"]["end_date"] == recent

    def test_null_end_date_preserved(self, tmp_config) -> None:
        path = tmp_config("""\
            general:
              base_currency: GBP
              timezone: Europe/London
              log_level: INFO
              data_dir: data
              random_seed: 42
            universe:
              name: test
              tickers: [A]
              benchmark: "^FTSE"
            data:
              source: synthetic
              start_date: "2020-01-01"
              end_date:
              interval: "1d"
              fields: [Close]
        """)
        cfg = load_config(path)
        assert cfg["data"]["end_date"] is None

    def test_resolved_date_never_stale(self) -> None:
        """Default config resolves end_date to at most 1 day stale."""
        cfg = load_config()
        end_date = cfg["data"]["end_date"]
        if end_date is None:
            pytest.skip("end_date is null in config")
        parsed = datetime.date.fromisoformat(end_date)
        delta = (datetime.date.today() - parsed).days
        assert delta <= 1, f"end_date is {delta} days stale: {end_date}"
