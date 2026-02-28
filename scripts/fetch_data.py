#!/usr/bin/env python3
"""Fetch market data for the configured universe.

Usage:
    python -m scripts.fetch_data
    python -m scripts.fetch_data --source synthetic
    python -m scripts.fetch_data --source yfinance --start 2020-01-01
    python -m scripts.fetch_data --tickers AAPL MSFT GOOGL
    python -m scripts.fetch_data --source synthetic --features
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
    parser.add_argument(
        "--features",
        action="store_true",
        help="Run feature pipeline after fetch/clean and save feature matrices",
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
    else:
        clean_data = raw_data

    # Feature engineering
    if args.features:
        from src.features.pipeline import FeaturePipeline
        from src.features.visualisation import (
            plot_price_with_bollinger,
            plot_rsi,
            plot_macd,
            plot_feature_correlations,
        )

        pipeline = FeaturePipeline(config=config)
        feature_dir = data_dir / "processed" / "features"

        # Use cleaned data if available, otherwise raw
        input_data = clean_data if not args.no_clean else raw_data

        features = pipeline.generate(input_data)
        feature_path = pipeline.generate_and_save(input_data, feature_dir / "features.parquet")
        logger.info(
            "Feature pipeline: %d features, %d rows → %s",
            len(pipeline.get_feature_names()), len(features), feature_path,
        )

        # Generate visualisation charts for each ticker
        charts_dir = data_dir / "processed" / "charts"
        charts_dir.mkdir(parents=True, exist_ok=True)

        for ticker, df in input_data.items():
            ticker_features = pipeline.generate({ticker: df})
            # Drop the ticker column for plotting
            if "ticker" in ticker_features.columns:
                ticker_features = ticker_features.drop(columns=["ticker"])

            safe_name = ticker.replace(".", "_").replace("^", "")

            try:
                plot_price_with_bollinger(
                    df, ticker_features, ticker=ticker,
                    save_path=charts_dir / f"{safe_name}_bollinger.png",
                )
            except Exception as exc:
                logger.warning("Could not plot Bollinger for %s: %s", ticker, exc)

            try:
                plot_rsi(
                    df, ticker_features, ticker=ticker,
                    save_path=charts_dir / f"{safe_name}_rsi.png",
                )
            except Exception as exc:
                logger.warning("Could not plot RSI for %s: %s", ticker, exc)

            try:
                plot_macd(
                    df, ticker_features, ticker=ticker,
                    save_path=charts_dir / f"{safe_name}_macd.png",
                )
            except Exception as exc:
                logger.warning("Could not plot MACD for %s: %s", ticker, exc)

        # One correlation heatmap for all features
        try:
            plot_feature_correlations(
                features.select_dtypes(include="number"),
                save_path=charts_dir / "feature_correlations.png",
            )
        except Exception as exc:
            logger.warning("Could not plot correlations: %s", exc)

        logger.info("Charts saved to %s", charts_dir)

    logger.info("Done!")


if __name__ == "__main__":
    main()
