#!/usr/bin/env python3
"""Fetch market data for the configured universe.

Usage:
    python -m scripts.fetch_data
    python -m scripts.fetch_data --source synthetic
    python -m scripts.fetch_data --source yfinance --start 2020-01-01
    python -m scripts.fetch_data --tickers AAPL MSFT GOOGL
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.fetcher import create_fetcher
from src.data.cleaner import DataCleaner
from src.utils.config import load_config, get_data_dir
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch market data")
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Data source: synthetic, yfinance (default: from config)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date YYYY-MM-DD (default: from config)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="Override tickers (default: from config universe)",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Skip the cleaning step",
    )
    parser.add_argument(
        "--output-format",
        type=str,
        default=None,
        choices=["parquet", "csv"],
        help="Output format (default: from config)",
    )

    args = parser.parse_args()

    # Load config
    config = load_config()

    # Resolve parameters (CLI > config)
    source = args.source or config["data"]["source"]
    start = args.start or config["data"]["start_date"]
    end = args.end or config["data"].get("end_date")
    tickers = args.tickers or config["universe"]["tickers"]
    output_format = args.output_format or config["data"]["output_format"]

    logger.info(f"Data source: {source}")
    logger.info(f"Universe: {len(tickers)} tickers")
    logger.info(f"Date range: {start} → {end or 'today'}")

    # Fetch
    fetcher = create_fetcher(source, seed=config["general"].get("random_seed", 42))
    raw_data = fetcher.fetch_multiple(tickers, start=start, end=end)

    if not raw_data:
        logger.error("No data fetched. Exiting.")
        sys.exit(1)

    # Save raw
    data_dir = get_data_dir(config)
    raw_paths = fetcher.save(raw_data, data_dir / "raw", fmt=output_format)
    logger.info(f"Saved {len(raw_paths)} raw files to {data_dir / 'raw'}")

    # Clean
    if not args.no_clean:
        cleaner = DataCleaner()
        clean_data = cleaner.clean_multiple(raw_data)
        clean_paths = fetcher.save(clean_data, data_dir / "processed", fmt=output_format)
        logger.info(f"Saved {len(clean_paths)} cleaned files to {data_dir / 'processed'}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
