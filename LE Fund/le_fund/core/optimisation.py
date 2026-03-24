"""
le_fund/core/optimisation.py
─────────────────────────────
OptimisationEngine: Max Sharpe, Min Volatility, Risk Parity portfolio
construction with cardinality constraints and beta filtering.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import cvxpy as cp

from le_fund.config import (
    CVAR_ALPHA,
    DD_PENALTY,
    MAX_CORR_FILTER,
    MAX_POSITIONS,
    MIN_POSITIONS,
    MAX_WEIGHT,
    MAX_CASH,
    MIN_ASSET_BETA,
    MIN_PORTFOLIO_BETA,
    MIN_WEIGHT,
    MOMENTUM_WINDOW,
    RISK_FREE_RATE,
)
from le_fund.core.risk import RiskEngine
from le_fund.core.numba_kernels import sharpe_objective_nb, risk_parity_objective_nb


class OptimisationEngine:
    """
    Three portfolio construction strategies — all with:

    · Cardinality constraint (max 15 positions)
    · Min / max weight bounds
    · Fully-invested constraint
    · Beta pre-screener (excludes assets below MIN_ASSET_BETA)
    · Blended expected returns (James-Stein + momentum + FinBERT sentiment)
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        cov: pd.DataFrame,
        max_positions: int = MAX_POSITIONS,
        min_weight: float = MIN_WEIGHT,
        max_weight: float = MAX_WEIGHT,
        prices: Optional[pd.DataFrame] = None,
        sentiment: Optional[pd.Series] = None,
    ):
        self.returns       = returns
        self.cov           = cov
        self.n             = len(returns.columns)
        self.tickers       = returns.columns.tolist()
        self.max_positions = max_positions
        self.min_weight    = min_weight
        self.max_weight    = max_weight
        self.prices        = prices

        # Blended expected returns (James-Stein + momentum + optional sentiment)
        if prices is not None and len(prices) > MOMENTUM_WINDOW:
            self.mu = RiskEngine.blended_expected_returns(returns, prices, sentiment)
        else:
            self.mu = RiskEngine.shrinkage_expected_returns(returns)

        self.sigma      = cov.values
        self.semi_sigma = RiskEngine.semi_variance(returns)

        # ── Per-asset beta vs Fama-French 3-factor market proxy ────────────
        from le_fund.core.fama_french import calculate_capm_ff3_betas
        raw_betas = calculate_capm_ff3_betas(returns, use_ff3=True)

        # Fall back to 1.0 for tickers with fewer than 60 observations
        obs_counts = returns.notna().sum().values
        raw_betas[obs_counts < 60] = 1.0
        self.betas = raw_betas

    # ── Candidate pre-screening ──────────────────────────────────────────

    def _select_top_assets(self, n: int) -> List[int]:
        """
        Diversification-aware pre-screening.

        Greedy: rank by Sharpe, apply beta filter, then accept only
        if max |correlation| with already-selected assets ≤ MAX_CORR_FILTER.
        Pads back to n if the correlation filter is too strict.
        """
        vols    = np.sqrt(np.diag(self.sigma))
        sharpes = (self.mu - RISK_FREE_RATE) / (vols + 1e-10)

        beta_mask = self.betas >= MIN_ASSET_BETA
        ranked    = np.argsort(sharpes)[::-1]
        ranked    = ranked[beta_mask[ranked]]

        std_outer = np.outer(vols, vols) + 1e-10
        corr      = self.sigma / std_outer

        selected   = np.empty(n, dtype=np.intp)
        n_selected = 0

        for idx in ranked:
            if n_selected >= n:
                break
            if n_selected > 0:
                if np.max(np.abs(corr[idx, selected[:n_selected]])) > MAX_CORR_FILTER:
                    continue
            selected[n_selected] = idx
            n_selected += 1

        if n_selected < n:
            selected_set = set(selected[:n_selected].tolist())
            for idx in ranked:
                if n_selected >= n:
                    break
                if int(idx) not in selected_set:
                    selected[n_selected] = idx
                    selected_set.add(int(idx))
                    n_selected += 1

        return sorted(selected[:n_selected].tolist())

    # ── Strategy 1: Max Sharpe ───────────────────────────────────────────

    def max_sharpe(self) -> pd.Series:
        """
        Maximise risk-adjusted return with a drawdown penalty.

        Objective:  max (Rp - Rf) / downside_vol  −  λ × CVaR

        · Semi-variance (downside vol) instead of total vol
        · CVaR penalty on worst 5 % of daily returns
        · Blended expected returns
        · Correlation-filtered candidate set
        · 20 parallel multi-start runs for better convergence
        """
        candidates     = self._select_top_assets(min(self.n, self.max_positions * 2))
        mu_sub         = self.mu[candidates].astype(np.float64)
        sigma_sub      = self.sigma[np.ix_(candidates, candidates)].astype(np.float64)
        semi_sigma_sub = self.semi_sigma[np.ix_(candidates, candidates)].astype(np.float64)
        n_sub          = len(candidates)
        betas_sub      = self.betas[candidates].astype(np.float64)

        daily_rets_sub    = self.returns.iloc[:, candidates].values.astype(np.float64)
        daily_rets_sorted = np.sort(daily_rets_sub, axis=0)

        _mu  = mu_sub
        _ss  = semi_sigma_sub
        _drs = daily_rets_sorted
        _rf  = float(RISK_FREE_RATE)
        _a   = float(CVAR_ALPHA)
        _ddp = float(DD_PENALTY)
        _mpb = float(MIN_PORTFOLIO_BETA)
        _b   = betas_sub

        def _obj(w: np.ndarray) -> float:
            return sharpe_objective_nb(
                w.astype(np.float64), _mu, _ss, _drs, _rf, _a, _ddp, _mpb, _b
            )

        constraints = [
            {"type": "eq",   "fun": lambda w: np.sum(w) - 1.0},
            {"type": "ineq", "fun": lambda w, b=betas_sub: w @ b - MIN_PORTFOLIO_BETA},
        ]
        bounds = [(0, self.max_weight)] * n_sub

        rng    = np.random.default_rng()
        starts = [rng.dirichlet(np.ones(n_sub)) for _ in range(20)]

        def _one_run(w0: np.ndarray):
            w0 = np.clip(w0, 0, self.max_weight)
            w0 /= w0.sum()
            res = minimize(
                _obj, w0, method="SLSQP",
                bounds=bounds, constraints=constraints,
                options={"maxiter": 2000, "ftol": 1e-14},
            )
            return res if res.success else None

        best_result, best_obj = None, np.inf
        with ThreadPoolExecutor(max_workers=min(8, 20)) as pool:
            for res in pool.map(_one_run, starts):
                if res is not None and res.fun < best_obj:
                    best_obj, best_result = res.fun, res.x

        if best_result is None:
            best_result = np.ones(n_sub) / n_sub


        # Cardinality enforcement (max positions)
        sorted_idx = np.argsort(best_result)
        best_result[sorted_idx[: n_sub - self.max_positions]] = 0
        best_result /= best_result.sum()


        # Enforce minimum number of positions (non-cash)
        nonzero = np.count_nonzero(best_result > self.min_weight / 2)
        if nonzero < MIN_POSITIONS:
            # Find indices with smallest nonzero weights and set to min_weight
            # If not enough, add from zero-weighted assets
            current_nonzero = np.flatnonzero(best_result > self.min_weight / 2)
            needed = MIN_POSITIONS - nonzero
            zero_idx = [i for i in range(n_sub) if best_result[i] <= self.min_weight / 2]
            # Set min_weight to enough zero-weighted assets
            for i in zero_idx[:needed]:
                best_result[i] = self.min_weight
            best_result /= best_result.sum()

        # Cap any single position at MAX_WEIGHT (should already be handled by bounds, but double-check)
        best_result = np.minimum(best_result, self.max_weight)
        best_result /= best_result.sum()

        mask = best_result > 0
        best_result[mask] = np.maximum(best_result[mask], self.min_weight)
        best_result /= best_result.sum()

        full_weights = np.zeros(self.n)
        for i, idx in enumerate(candidates):
            full_weights[idx] = best_result[i]
        weights = pd.Series(full_weights, index=self.tickers, name="max_sharpe")
        # Cash constraint will be enforced in VolatilityTargeter
        return weights

    # ── Strategy 2: Min Volatility ───────────────────────────────────────

    def min_volatility(self) -> pd.Series:
        """Global minimum variance portfolio with cardinality constraint."""
        candidates = self._select_top_assets(min(self.n, self.max_positions * 2))
        sigma_sub  = self.sigma[np.ix_(candidates, candidates)]
        n_sub      = len(candidates)

        w = cp.Variable(n_sub)
        prob = cp.Problem(
            cp.Minimize(cp.quad_form(w, cp.psd_wrap(sigma_sub))),
            [cp.sum(w) == 1, w >= 0, w <= self.max_weight],
        )
        try:
            prob.solve(solver=cp.SCS, verbose=False)
            weights = w.value
        except Exception:
            weights = None

        if weights is None:
            weights = np.ones(n_sub) / n_sub

        if np.count_nonzero(weights > self.min_weight / 2) > self.max_positions:
            sorted_idx = np.argsort(weights)
            weights[sorted_idx[: n_sub - self.max_positions]] = 0

        weights = np.maximum(weights, 0)
        mask    = weights > 0
        weights[mask] = np.maximum(weights[mask], self.min_weight)
        weights /= weights.sum()

        full_weights = np.zeros(self.n)
        for i, idx in enumerate(candidates):
            full_weights[idx] = weights[i]
        return pd.Series(full_weights, index=self.tickers, name="min_vol")

    # ── Strategy 3: Risk Parity ──────────────────────────────────────────

    def risk_parity(self) -> pd.Series:
        """Equal risk contribution portfolio."""
        candidates = self._select_top_assets(min(self.n, self.max_positions * 2))
        sigma_sub  = self.sigma[np.ix_(candidates, candidates)].astype(np.float64)
        n_sub      = len(candidates)
        target_rc  = float(1.0 / self.max_positions)
        _sigma     = sigma_sub

        def _obj(w: np.ndarray) -> float:
            return risk_parity_objective_nb(w.astype(np.float64), _sigma, target_rc)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds      = [(0, self.max_weight)] * n_sub

        rng    = np.random.default_rng()
        starts = [rng.dirichlet(np.ones(n_sub)) for _ in range(10)]

        def _one_run(w0: np.ndarray):
            w0 = np.clip(w0, 0, self.max_weight)
            w0 /= w0.sum()
            res = minimize(
                _obj, w0, method="SLSQP",
                bounds=bounds, constraints=constraints,
                options={"maxiter": 1000},
            )
            return res if res.success else None

        best_result, best_obj = None, np.inf
        with ThreadPoolExecutor(max_workers=min(8, 10)) as pool:
            for res in pool.map(_one_run, starts):
                if res is not None and res.fun < best_obj:
                    best_obj, best_result = res.fun, res.x

        if best_result is None:
            best_result = np.ones(n_sub) / n_sub

        if np.count_nonzero(best_result > self.min_weight / 2) > self.max_positions:
            sorted_idx = np.argsort(best_result)
            best_result[sorted_idx[: n_sub - self.max_positions]] = 0
            best_result /= best_result.sum()

        mask = best_result > 0
        best_result[mask] = np.maximum(best_result[mask], self.min_weight)
        best_result /= best_result.sum()

        full_weights = np.zeros(self.n)
        for i, idx in enumerate(candidates):
            full_weights[idx] = best_result[i]
        return pd.Series(full_weights, index=self.tickers, name="risk_parity")
