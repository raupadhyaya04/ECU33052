"""
le_fund/core/risk.py
─────────────────────
RiskEngine: covariance estimation, volatility forecasting, regime detection,
risk contributions, and blended expected return construction.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from le_fund.config import (
    CVAR_ALPHA,
    EWMA_SPAN,
    MOMENTUM_WINDOW,
    SENTIMENT_WEIGHT,
    SHRINKAGE_ALPHA,
)
from le_fund.core.numba_kernels import (
    cvar_nb,
    risk_contributions_nb,
    semi_variance_nb,
)

# ── Module-level caches ──────────────────────────────────────────────────
# Shared across all strategies so LedoitWolf and semi-variance are computed
# only once per unique (window_start, window_end, shape).
_COV_CACHE: Dict[tuple, pd.DataFrame] = {}
_SEMI_VAR_CACHE: Dict[tuple, np.ndarray] = {}


def _cov_cache_key(returns: pd.DataFrame) -> tuple:
    return (returns.index[0], returns.index[-1], returns.shape)


class RiskEngine:
    """
    Provides risk analytics for portfolio construction:

    · Ledoit-Wolf shrinkage covariance  (robust for large N)
    · EWMA volatility forecasts
    · Volatility regime detection
    · Risk contribution / attribution
    · Blended expected returns  (James-Stein + momentum + FinBERT sentiment)
    """

    # ── Returns ─────────────────────────────────────────────────────────

    @staticmethod
    def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
        return prices.pct_change().dropna()

    # ── Covariance ───────────────────────────────────────────────────────

    @staticmethod
    def shrinkage_covariance(returns: pd.DataFrame) -> pd.DataFrame:
        """
        Ledoit-Wolf shrinkage covariance — ideal when N is large relative to T.
        Results are module-level cached by (first_date, last_date, shape).
        """
        key = _cov_cache_key(returns)
        if key not in _COV_CACHE:
            lw = LedoitWolf().fit(returns.values)
            _COV_CACHE[key] = pd.DataFrame(
                lw.covariance_,
                index=returns.columns,
                columns=returns.columns,
            )
            if len(_COV_CACHE) > 512:
                _COV_CACHE.pop(next(iter(_COV_CACHE)))
        return _COV_CACHE[key]

    @staticmethod
    def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
        cov = RiskEngine.shrinkage_covariance(returns)
        std = np.sqrt(np.diag(cov.values))
        corr = cov.values / np.outer(std, std)
        return pd.DataFrame(corr, index=returns.columns, columns=returns.columns)

    # ── Volatility ───────────────────────────────────────────────────────

    @staticmethod
    def ewma_volatility(returns: pd.DataFrame, span: int = EWMA_SPAN) -> pd.Series:
        """EWMA volatility forecast (annualised)."""
        ewma_var = returns.ewm(span=span).var().iloc[-1]
        return np.sqrt(ewma_var * 252)

    @staticmethod
    def rolling_volatility(returns: pd.DataFrame, window: int = 63) -> pd.DataFrame:
        """63-day (quarterly) rolling realised volatility."""
        return returns.rolling(window).std() * np.sqrt(252)

    @staticmethod
    def detect_vol_regime(
        returns: pd.DataFrame,
        short_window: int = 21,
        long_window: int = 126,
    ) -> str:
        """
        Compare recent vol to long-term vol to classify the current regime.
        Returns one of: 'HIGH_VOL' | 'NORMAL' | 'LOW_VOL'.
        """
        port_ret  = returns.mean(axis=1)
        short_vol = port_ret.iloc[-short_window:].std() * np.sqrt(252)
        long_vol  = port_ret.iloc[-long_window:].std() * np.sqrt(252)
        ratio = short_vol / long_vol if long_vol > 0 else 1.0

        if ratio > 1.5:
            return "HIGH_VOL"
        elif ratio < 0.7:
            return "LOW_VOL"
        return "NORMAL"

    # ── Risk contributions ───────────────────────────────────────────────

    @staticmethod
    def risk_contributions(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """Marginal risk contributions — delegates to numba JIT kernel."""
        return risk_contributions_nb(
            weights.astype(np.float64),
            cov.astype(np.float64),
        )

    # ── Expected returns ─────────────────────────────────────────────────

    @staticmethod
    def shrinkage_expected_returns(
        returns: pd.DataFrame,
        alpha: float = SHRINKAGE_ALPHA,
    ) -> np.ndarray:
        """
        James-Stein shrinkage: pull individual means toward the grand mean,
        dramatically reducing estimation error vs raw sample means.
        """
        mu = returns.mean().values * 252          # annualised raw means
        grand_mean = mu.mean()
        return alpha * grand_mean + (1 - alpha) * mu

    @staticmethod
    def momentum_score(
        prices: pd.DataFrame,
        window: int = MOMENTUM_WINDOW,
    ) -> pd.Series:
        """6-month cross-sectional price momentum."""
        if len(prices) < window:
            return pd.Series(0.0, index=prices.columns)
        return prices.iloc[-1] / prices.iloc[-window] - 1

    @staticmethod
    def blended_expected_returns(
        returns: pd.DataFrame,
        prices: pd.DataFrame,
        sentiment: Optional[pd.Series] = None,
    ) -> np.ndarray:
        """
        Blend three forward-looking signals:

        With sentiment   → 50 % James-Stein + 30 % momentum + 20 % FinBERT
        Without sentiment → 60 % James-Stein + 40 % momentum  (backtest fallback)

        All signals are normalised to the same scale before blending.
        """
        shrunk = RiskEngine.shrinkage_expected_returns(returns)
        mom    = (
            RiskEngine.momentum_score(prices)
            .reindex(returns.columns)
            .fillna(0)
            .values
        )
        mom_scaled = mom * (np.std(shrunk) / (np.std(mom) + 1e-10))

        if sentiment is None or sentiment.isna().all():
            # backtest fallback: original 60/40 split
            return 0.6 * shrunk + 0.4 * mom_scaled

        sent = sentiment.reindex(returns.columns).fillna(0).values
        sent_scaled = sent * (np.std(shrunk) / (np.std(sent) + 1e-10))

        return (
            (1.0 - SENTIMENT_WEIGHT) * (0.625 * shrunk + 0.375 * mom_scaled)
            + SENTIMENT_WEIGHT * sent_scaled
        )

    # ── CVaR / semi-variance ─────────────────────────────────────────────

    @staticmethod
    def cvar(returns_series: np.ndarray, alpha: float = CVAR_ALPHA) -> float:
        """CVaR (Expected Shortfall) at alpha level — numba JIT kernel."""
        return cvar_nb(np.sort(returns_series), alpha)

    @staticmethod
    def semi_variance(returns: pd.DataFrame) -> np.ndarray:
        """
        Downside semi-covariance matrix — numba JIT kernel.
        Cached by window key so all strategies on the same window pay
        the cost only once.
        """
        key = _cov_cache_key(returns)
        if key not in _SEMI_VAR_CACHE:
            _SEMI_VAR_CACHE[key] = semi_variance_nb(
                returns.values.astype(np.float64)
            )
            if len(_SEMI_VAR_CACHE) > 512:
                _SEMI_VAR_CACHE.pop(next(iter(_SEMI_VAR_CACHE)))
        return _SEMI_VAR_CACHE[key]
