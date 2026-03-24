"""
le_fund/sentiment/engine.py
────────────────────────────
SentimentEngine: scores recent news headlines for each ticker using
FinBERT (ProsusAI/finbert) — a BERT model fine-tuned on financial text.

Pipeline
────────
1. Fetch up to SENTIMENT_HEADLINES recent headlines per ticker via yfinance.
2. Score each headline via the HuggingFace Inference API (primary path).
   Falls back to local FinBERT via the transformers library if the API is
   unavailable or rate-limited.
3. Compute per-ticker compound score:
       score = P(positive) − P(negative)  ∈ [−1, +1]
   averaged across all scored headlines.
4. Returns a pd.Series indexed by ticker; NaN for tickers with no data
   (treated as neutral = 0 downstream).

The scores are blended into expected returns in
RiskEngine.blended_expected_returns() as the third signal alongside
James-Stein shrinkage and price momentum.
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from le_fund.config import SENTIMENT_HEADLINES


class SentimentEngine:
    """FinBERT sentiment scorer for the ticker universe."""

    HF_API_URL  = "https://router.huggingface.co/hf-inference/models/ProsusAI/finbert"
    LOCAL_MODEL = "ProsusAI/finbert"
    _pipeline   = None   # lazy-loaded; shared across all instances

    def __init__(
        self,
        hf_token: Optional[str] = None,
        n_headlines: int = SENTIMENT_HEADLINES,
        timeout: int = 20,
    ):
        self.token       = hf_token or os.getenv("HF_TOKEN", "")
        self.n_headlines = n_headlines
        self.timeout     = timeout
        self._cache: Dict[str, float] = {}   # ticker → score (session-level)

    # ── Public interface ─────────────────────────────────────────────────

    def score_universe(self, tickers: List[str]) -> pd.Series:
        """
        Score all tickers in parallel (thread pool, I/O bound).
        Returns a Series of compound sentiment scores in [−1, +1].
        Prints a summary table for transparency.
        """
        print("\n  📰 Fetching & scoring news sentiment (FinBERT)…")
        with ThreadPoolExecutor(max_workers=min(8, len(tickers))) as pool:
            futures = {pool.submit(self._score_ticker, t): t for t in tickers}
            results: Dict[str, float] = {}
            for fut in as_completed(futures):
                t = futures[fut]
                try:
                    results[t] = fut.result()
                except Exception:
                    results[t] = float("nan")

        scores = pd.Series(results).reindex(tickers)
        self._print_table(scores)
        return scores

    # ── Private helpers ──────────────────────────────────────────────────

    def _score_ticker(self, ticker: str) -> float:
        if ticker in self._cache:
            return self._cache[ticker]

        headlines = self._fetch_headlines(ticker)
        if not headlines:
            return float("nan")

        scored = self._score_headlines(headlines)
        if not scored:
            return float("nan")

        compound = float(np.mean(scored))
        self._cache[ticker] = compound
        return compound

    def _fetch_headlines(self, ticker: str) -> List[str]:
        """Pull recent news titles from yfinance."""
        try:
            news   = yf.Ticker(ticker).news or []
            titles = []
            for item in news[: self.n_headlines]:
                content = item.get("content", item)
                title   = content.get("title", "")
                if title:
                    titles.append(title)
            return titles
        except Exception:
            return []

    def _score_headlines(self, headlines: List[str]) -> List[float]:
        """
        Try the HF Inference API first; fall back to local FinBERT.
        Returns a list of compound scores (positive − negative) per headline.
        """
        if self.token:
            try:
                return self._score_via_api(headlines)
            except Exception as exc:
                print(f"    ⚠ HF API error ({exc}), falling back to local FinBERT…")
        return self._score_via_local(headlines)

    def _score_via_api(self, headlines: List[str]) -> List[float]:
        """
        Score via HuggingFace Inference API — headlines sent in parallel.
        Response format: [[{label, score}, …]]
        Compound = P(positive) − P(negative)
        """
        import requests as _req

        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type":  "application/json",
        }
        session = _req.Session()
        session.headers.update(headers)

        def _post(headline: str) -> float:
            resp = session.post(
                self.HF_API_URL,
                json={"inputs": headline},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            label_scores = resp.json()
            if isinstance(label_scores, list) and label_scores:
                inner  = label_scores[0] if isinstance(label_scores[0], list) else label_scores
                lookup = {d["label"]: d["score"] for d in inner}
                return lookup.get("positive", 0.0) - lookup.get("negative", 0.0)
            return 0.0

        # Fire all headline requests concurrently (I/O bound — threads are ideal)
        with ThreadPoolExecutor(max_workers=len(headlines)) as pool:
            futures = [pool.submit(_post, h) for h in headlines]
            compounds = []
            for fut in futures:
                try:
                    compounds.append(fut.result())
                except Exception:
                    pass
        return compounds

    def _score_via_local(self, headlines: List[str]) -> List[float]:
        """
        Local FinBERT inference via HuggingFace transformers pipeline.
        Model (~440 MB) is downloaded once and cached by transformers.
        Uses Apple MPS if available, else CUDA, else CPU.
        """
        if SentimentEngine._pipeline is None:
            try:
                from transformers import pipeline as hf_pipeline
                import torch

                device = (
                    "mps"  if torch.backends.mps.is_available() else
                    "cuda" if torch.cuda.is_available()          else
                    "cpu"
                )
                print(f"    ℹ Loading local FinBERT on {device}…")
                SentimentEngine._pipeline = hf_pipeline(
                    "text-classification",
                    model=self.LOCAL_MODEL,
                    device=device,
                    top_k=None,
                )
            except Exception as exc:
                print(f"    ✗ Local FinBERT unavailable: {exc}")
                return []

        # Batch all headlines in one forward pass — much faster than looping
        truncated = [h[:512] for h in headlines]
        try:
            batch_results = SentimentEngine._pipeline(truncated, batch_size=len(truncated))
            compounds = []
            for result in batch_results:
                labels = result if isinstance(result, list) else [result]
                lookup = {d["label"]: d["score"] for d in labels}
                compounds.append(lookup.get("positive", 0.0) - lookup.get("negative", 0.0))
            return compounds
        except Exception:
            return []

    # ── Display helpers ──────────────────────────────────────────────────

    @staticmethod
    def _print_table(scores: pd.Series):
        print(f"  {'Ticker':<14} {'Score':>7}  {'Signal'}")
        print(f"  {'──────':<14} {'──────':>7}  {'──────'}")
        for ticker, score in scores.sort_values(ascending=False).items():
            if pd.isna(score):
                print(f"  {ticker:<14} {'  n/a ':>7}  neutral (no data)")
            elif score > 0.1:
                bar = "█" * int(score * 10)
                print(f"  {ticker:<14} {score:>+7.3f}  ▲ positive  {bar}")
            elif score < -0.1:
                bar = "█" * int(abs(score) * 10)
                print(f"  {ticker:<14} {score:>+7.3f}  ▼ negative  {bar}")
            else:
                print(f"  {ticker:<14} {score:>+7.3f}    neutral")
