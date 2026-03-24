"""
le_fund/core/data.py
─────────────────────
DataEngine: chunked parallel download, parquet cache, coverage filtering.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import pandas as pd
import yfinance as yf
from tqdm import tqdm

from le_fund.config import DATA_DIR, OUTPUT_DIR, CHUNK_SIZE


class DataEngine:
    """
    Downloads, caches, and serves adjusted close prices for the full universe.

    Features
    ────────
    · Chunked parallel downloads via a thread pool (avoids API rate limits)
    · Parquet cache — skips re-download when cache is ≤ 3 days old
    · Coverage filter — drops tickers with insufficient history
    · Automatic forward-fill + dropna for clean returns
    """

    def __init__(self, tickers: List[str], start: str = "2020-01-01"):
        self.tickers  = tickers
        self.start    = start
        self.cache_path = DATA_DIR / "price_cache.parquet"
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def download(self, force: bool = False) -> pd.DataFrame:
        """
        Return a DataFrame of adjusted close prices (date index, ticker columns).

        Uses the parquet cache when fresh (≤ 3 days old); otherwise re-downloads
        all tickers in chunks, applies coverage filtering, and saves a new cache.
        """
        if not force and self.cache_path.exists():
            print(f"[DataEngine] Loading cached data from {self.cache_path}")
            df  = pd.read_parquet(self.cache_path)
            age = (datetime.now() - pd.Timestamp(df.index[-1])).days
            if age <= 3:
                print(f"[DataEngine] Cache is {age} day(s) old — using cached data")
                return df
            print(f"[DataEngine] Cache is {age} day(s) old — refreshing...")

        chunks = [
            self.tickers[i : i + CHUNK_SIZE]
            for i in range(0, len(self.tickers), CHUNK_SIZE)
        ]
        print(
            f"[DataEngine] Downloading {len(self.tickers)} tickers "
            f"in {len(chunks)} parallel chunks..."
        )

        def _fetch_chunk(chunk_idx: int, chunk: List[str]) -> Dict[str, pd.Series]:
            for attempt in range(3):
                try:
                    data = yf.download(
                        chunk,
                        start=self.start,
                        auto_adjust=True,
                        progress=False,
                        threads=True,
                    )
                    if isinstance(data.columns, pd.MultiIndex):
                        closes = data["Close"]
                    else:
                        closes = data[["Close"]].rename(columns={"Close": chunk[0]})
                    return {col: closes[col] for col in closes.columns}
                except Exception as exc:
                    print(
                        f"    ⚠ Chunk {chunk_idx + 1} attempt {attempt + 1} "
                        f"failed: {exc}"
                    )
                    time.sleep(2 ** attempt)
            return {}

        all_data: Dict[str, pd.Series] = {}
        with ThreadPoolExecutor(max_workers=min(8, len(chunks))) as pool:
            futures = {pool.submit(_fetch_chunk, i, c): i for i, c in enumerate(chunks)}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="  Chunks"):
                all_data.update(fut.result())

        prices = pd.DataFrame(all_data)
        prices.index = pd.to_datetime(prices.index)
        prices = prices.sort_index()

        # 40 % coverage threshold — allows newer listings (PLTR, SHOP, COIN etc.)
        threshold   = len(prices) * 0.40
        valid_cols  = prices.columns[prices.count() >= threshold]
        dropped     = set(prices.columns) - set(valid_cols)
        if dropped:
            print(
                f"[DataEngine] Dropped {len(dropped)} tickers with insufficient "
                f"data: {dropped}"
            )
        prices = prices[valid_cols].ffill().dropna()

        print(
            f"[DataEngine] Final dataset: "
            f"{prices.shape[0]} days × {prices.shape[1]} tickers"
        )
        prices.to_parquet(self.cache_path)
        return prices
