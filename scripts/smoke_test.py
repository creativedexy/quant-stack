#!/usr/bin/env python3
"""End-to-end smoke test for the Quant Stack pipeline.

Runs the full pipeline from scratch using synthetic data and reports
pass/fail for each step.  Exit code 0 if all steps pass, 1 otherwise.

Usage:
    py -3 scripts/smoke_test.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Configuration ─────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
RAW_DIR = DATA_DIR / "raw"
PYTHON = sys.executable
STREAMLIT_PORT = 8511  # Use non-default port to avoid conflicts
STREAMLIT_TIMEOUT = 15  # Seconds to wait for Streamlit to start
MAX_HEALTH_RETRIES = 10


class StepResult:
    """Container for a single smoke test step result."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.passed = False
        self.duration = 0.0
        self.error: str | None = None

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        msg = f"  [{status}] {self.name} ({self.duration:.1f}s)"
        if self.error:
            msg += f"\n         {self.error}"
        return msg


def _run_step(name: str, func) -> StepResult:
    """Execute a step function and capture timing and errors."""
    result = StepResult(name)
    start = time.monotonic()
    try:
        func()
        result.passed = True
    except Exception as exc:
        result.error = str(exc)
    result.duration = time.monotonic() - start
    return result


# ── Step Functions ────────────────────────────────────────────────

def step_clean_data() -> None:
    """Delete processed/ and synthetic/ directories for a fresh start."""
    for d in [PROCESSED_DIR, SYNTHETIC_DIR]:
        if d.exists():
            shutil.rmtree(d)
    # Also remove raw/ to ensure fully fresh
    if RAW_DIR.exists():
        shutil.rmtree(RAW_DIR)


def step_fetch_synthetic() -> None:
    """Run fetch_data --source synthetic."""
    result = subprocess.run(
        [PYTHON, "-m", "scripts.fetch_data", "--source", "synthetic"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"fetch_data exited with code {result.returncode}\n"
            f"stderr: {result.stderr[-500:]}"
        )
    # Verify files were created
    parquet_files = list(PROCESSED_DIR.glob("*.parquet"))
    if not parquet_files:
        raise RuntimeError("No parquet files found in data/processed/")


def step_feature_pipeline() -> None:
    """Run the feature pipeline for all tickers."""
    from src.utils.config import load_config
    from src.features.pipeline import FeaturePipeline
    from src.data.cleaner import DataCleaner

    config = load_config()
    tickers = config["universe"]["tickers"]

    # Load processed data
    import pandas as pd
    data: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        safe = ticker.replace(".", "_").replace("^", "idx_")
        path = PROCESSED_DIR / f"{safe}.parquet"
        if path.exists():
            data[ticker] = pd.read_parquet(path)

    if not data:
        raise RuntimeError("No processed data files found to generate features")

    pipeline = FeaturePipeline(config=config)
    features = pipeline.generate(data)

    if features.empty:
        raise RuntimeError("Feature pipeline returned empty DataFrame")

    feature_count = len(pipeline.get_feature_names())
    if feature_count < 5:
        raise RuntimeError(
            f"Only {feature_count} features generated (expected >= 5)"
        )


def step_model_training() -> None:
    """Train models for all tickers via the pipeline runner."""
    from src.utils.config import load_config
    from src.scheduler.pipeline import PipelineRunner

    config = load_config()
    runner = PipelineRunner(config)
    result = runner.run_model_retrain()

    if result["status"] == "failed":
        raise RuntimeError(f"Model retrain failed: {result.get('reason', 'unknown')}")


def step_portfolio_optimisation() -> None:
    """Run portfolio optimisation on synthetic returns."""
    from src.utils.config import load_config
    from src.portfolio.optimiser import PortfolioOptimiser
    from src.services.data_service import DataService

    config = load_config()
    data_svc = DataService(config=config)
    returns = data_svc.get_returns(window=60)

    if returns.empty:
        raise RuntimeError("No return data available for optimisation")

    optimiser = PortfolioOptimiser(config=config)
    weights = optimiser.optimise(returns, method="equal_weight")

    if weights.empty:
        raise RuntimeError("Optimiser returned empty weights")

    if abs(weights.sum() - 1.0) > 0.01:
        raise RuntimeError(
            f"Weights do not sum to 1.0 (got {weights.sum():.4f})"
        )


def step_backtest() -> None:
    """Run backtest for the default mean_reversion strategy."""
    result = subprocess.run(
        [PYTHON, "-m", "scripts.run_backtest", "--strategy", "mean_reversion"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"run_backtest exited with code {result.returncode}\n"
            f"stderr: {result.stderr[-500:]}"
        )


def step_streamlit_health() -> None:
    """Start Streamlit, hit /_stcore/health, verify HTTP 200, then kill."""
    proc = subprocess.Popen(
        [
            PYTHON, "-m", "streamlit", "run", "app.py",
            "--server.port", str(STREAMLIT_PORT),
            "--server.headless", "true",
            "--server.address", "127.0.0.1",
        ],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        # Wait for Streamlit to start, then hit health endpoint
        url = f"http://127.0.0.1:{STREAMLIT_PORT}/_stcore/health"
        healthy = False

        for attempt in range(MAX_HEALTH_RETRIES):
            time.sleep(1.5)
            try:
                req = urllib.request.urlopen(url, timeout=5)
                if req.status == 200:
                    healthy = True
                    break
            except (urllib.error.URLError, ConnectionError, OSError):
                continue

        if not healthy:
            raise RuntimeError(
                f"Streamlit did not respond at {url} after "
                f"{MAX_HEALTH_RETRIES} attempts"
            )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


# ── Dashboard Smoke Checklist (documented, not automated) ─────────
DASHBOARD_CHECKLIST = """
Dashboard Smoke Checklist (manual verification with synthetic data):
  [x] All 5 nav pages load without exception (Overview, Strategy, Portfolio, Research, Execution)
  [x] Overview page shows portfolio value and P&L metrics
  [x] Strategy page shows equity curve and metrics for at least one strategy
  [x] Portfolio page shows current vs target weights
  [x] Research page shows feature explorer with at least one feature selectable
  [x] Execution page shows paper trade controls
"""


# ── Main ──────────────────────────────────────────────────────────

def main() -> int:
    """Run all smoke test steps and print summary."""
    print("=" * 60)
    print("  Quant Stack — End-to-End Smoke Test")
    print("=" * 60)
    print()

    total_start = time.monotonic()

    steps = [
        ("1. Clean data directories", step_clean_data),
        ("2. Fetch synthetic data", step_fetch_synthetic),
        ("3. Feature pipeline (all tickers)", step_feature_pipeline),
        ("4. Model training (all tickers)", step_model_training),
        ("5. Portfolio optimisation", step_portfolio_optimisation),
        ("6. Backtest (mean_reversion)", step_backtest),
        ("7. Streamlit health check", step_streamlit_health),
    ]

    results: list[StepResult] = []
    for name, func in steps:
        print(f"  Running: {name} ...", end="", flush=True)
        result = _run_step(name, func)
        results.append(result)
        status = "PASS" if result.passed else "FAIL"
        print(f" [{status}] ({result.duration:.1f}s)")
        if result.error:
            print(f"    Error: {result.error}")

    total_time = time.monotonic() - total_start

    # ── Summary ───────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  Summary")
    print("=" * 60)
    for r in results:
        print(r)

    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)

    print()
    print(f"  Total: {passed} passed, {failed} failed")
    print(f"  Time:  {total_time:.1f}s")
    print()
    print(DASHBOARD_CHECKLIST)

    if failed > 0:
        print("  RESULT: FAIL")
        return 1
    else:
        print("  RESULT: ALL STEPS PASSED")
        return 0


if __name__ == "__main__":
    sys.exit(main())
