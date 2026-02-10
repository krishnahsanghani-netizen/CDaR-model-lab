"""Data loading utilities for price series."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DataMetadata:
    """Basic source metadata for loaded prices."""

    source: str
    downloaded_at_utc: str
    tickers: list[str]
    start: str
    end: str
    frequency: str


@dataclass(frozen=True)
class DataLoadResult:
    """Container for loaded prices and associated metadata."""

    prices: pd.DataFrame
    metadata: DataMetadata


def load_from_csv(
    path: str | Path,
    date_col: str,
    price_cols: Sequence[str] | None = None,
    parse_dates: bool = True,
) -> pd.DataFrame:
    """Load price data from CSV with a date index."""
    frame = pd.read_csv(path, parse_dates=[date_col] if parse_dates else None)
    if date_col not in frame.columns:
        raise ValueError(f"date_col='{date_col}' not found in CSV columns.")
    frame = frame.set_index(date_col).sort_index()
    if price_cols is not None:
        missing = [column for column in price_cols if column not in frame.columns]
        if missing:
            raise ValueError(f"Requested price columns missing from CSV: {missing}")
        frame = frame.loc[:, list(price_cols)]
    return frame


def load_from_yfinance(
    tickers: Sequence[str],
    start: str,
    end: str,
    interval: str = "1d",
    frequency: str = "daily",
    cache_dir: str | Path = "cache",
    use_cache: bool = True,
    refresh: bool = False,
) -> DataLoadResult:
    """Load adjusted close price data from yfinance with local cache support."""
    tickers_clean = [ticker.strip().upper() for ticker in tickers if ticker.strip()]
    if not tickers_clean:
        raise ValueError("At least one ticker is required.")

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    cache_key = _make_cache_key(tickers_clean, start, end, interval, frequency)
    prices_file = cache_path / f"{cache_key}.csv"
    meta_file = cache_path / f"{cache_key}.meta.json"

    if use_cache and not refresh and prices_file.exists() and meta_file.exists():
        LOGGER.info("Loading prices from cache: %s", prices_file)
        prices = pd.read_csv(prices_file, index_col=0, parse_dates=True)
        meta_dict = json.loads(meta_file.read_text(encoding="utf-8"))
        return DataLoadResult(prices=prices, metadata=DataMetadata(**meta_dict))

    LOGGER.info(
        "Downloading prices from yfinance (tickers=%s, start=%s, end=%s, interval=%s)",
        ",".join(tickers_clean),
        start,
        end,
        interval,
    )
    raw = yf.download(
        tickers=tickers_clean,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )

    prices = _extract_adjusted_close(raw, tickers_clean)
    prices = _resample_prices(prices, frequency)

    metadata = DataMetadata(
        source="yfinance",
        downloaded_at_utc=datetime.now(timezone.utc).isoformat(),
        tickers=tickers_clean,
        start=start,
        end=end,
        frequency=frequency,
    )
    if use_cache:
        LOGGER.info("Writing prices to cache: %s", prices_file)
        prices.to_csv(prices_file)
        meta_file.write_text(json.dumps(asdict(metadata), indent=2), encoding="utf-8")
    return DataLoadResult(prices=prices, metadata=metadata)


def _extract_adjusted_close(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Extract price panel from yfinance response."""
    if raw.empty:
        raise ValueError("No data returned by yfinance for the requested query.")

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            raise ValueError("Could not find Close data in yfinance response.")
        prices = raw["Close"].copy()
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(tickers[0])
    else:
        if "Close" in raw.columns:
            prices = raw.loc[:, ["Close"]].rename(columns={"Close": tickers[0]})
        else:
            prices = raw.to_frame(name=tickers[0]) if isinstance(raw, pd.Series) else raw.copy()

    prices.columns = [str(col).upper() for col in prices.columns]
    prices = prices.sort_index()
    return prices


def _resample_prices(prices: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """Resample prices by frequency using last available observation."""
    freq = frequency.lower()
    if freq == "daily":
        return prices
    if freq == "weekly":
        return prices.resample("W-FRI").last().dropna(how="all")
    if freq == "monthly":
        return prices.resample("M").last().dropna(how="all")
    raise ValueError(f"Unsupported frequency='{frequency}'. Use daily|weekly|monthly.")


def _make_cache_key(
    tickers: list[str], start: str, end: str, interval: str, frequency: str
) -> str:
    payload = "|".join([",".join(sorted(tickers)), start, end, interval, frequency])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
