"""
le_fund/core/backtest.py
─────────────────────────
Backtester: walk-forward simulation with monthly rebalancing, transaction
costs, and a drawdown circuit-breaker.

The per-day NAV loop is dispatched to the numba JIT kernel `daily_nav_loop`
so compiled C code handles the hot path — no Python overhead per day.
"""

from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

from le_fund.config import (
    DD_CIRCUIT_BREAK,
    DD_RESUME_LEVEL,
    DD_SCALE_FACTOR,
    FUND_CAPITAL,
    LOOKBACK_DAYS,
    REBALANCE_FREQ,
    TRANSACTION_COST,
)
from le_fund.core.numba_kernels import daily_nav_loop
from le_fund.core.risk import RiskEngine
from le_fund.core.optimisation import OptimisationEngine
from le_fund.core.volatility import VolatilityTargeter


class Backtester:
    """
    Walk-forward backtest.

    Phase 1 (Python): iterate rebalance dates, compute optimal weights.
    Phase 2 (numba):  simulate daily NAV from the full weight schedule
                      in one compiled call.
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        initial_capital: float = FUND_CAPITAL,
    ):
        self.prices          = prices
        self.initial_capital = initial_capital
        self.returns         = prices.pct_change().dropna()

    def run(
        self,
        strategy_name: str,
        lookback: int = LOOKBACK_DAYS,
        method: str = "max_sharpe",
    ) -> pd.DataFrame:
        """
        Execute a walk-forward backtest for *method* and return a DataFrame
        with columns: nav, turnover, daily_return, drawdown.
        """
        dates       = self.returns.index
        rebal_dates = self.returns.resample(REBALANCE_FREQ).last().index
        rebal_dates = rebal_dates[rebal_dates >= dates[lookback]]

        targeter = VolatilityTargeter()

        print(f"\n{'═' * 60}")
        print(f"  BACKTESTING: {strategy_name} ({method})")
        print(f"  Rebalance dates: {len(rebal_dates)} | Lookback: {lookback}d")
        print(f"{'═' * 60}")

        # ── Phase 1: compute weights at each rebalance date ──────────
        rebal_weights_list: List[np.ndarray] = []
        base_weights_list:  List[np.ndarray] = []
        rebal_date_flags: List[bool] = [False] * len(dates)

        for date in tqdm(rebal_dates, desc=f"  {strategy_name} rebal"):
            loc = dates.searchsorted(date, side="right") - 1
            if loc < lookback:
                continue
            window = self.returns.iloc[max(0, loc - lookback) : loc]
            if len(window) < 60:
                continue

            cov          = RiskEngine.shrinkage_covariance(window)
            regime       = RiskEngine.detect_vol_regime(window)
            price_window = self.prices.iloc[: loc + 1]
            engine       = OptimisationEngine(window, cov, prices=price_window)

            if method == "max_sharpe":
                weights = engine.max_sharpe()
            elif method == "min_vol":
                weights = engine.min_volatility()
            elif method == "risk_parity":
                weights = engine.risk_parity()
            else:
                raise ValueError(f"Unknown method: {method!r}")

            weights = targeter.scale_weights(weights, cov, regime)

            rebal_weights_list.append(weights.values.astype(np.float64))
            base_weights_list.append(weights.values.astype(np.float64))
            rebal_date_flags[loc] = True

        if not rebal_weights_list:
            raise RuntimeError(
                "No valid rebalance windows found — increase data history."
            )

        # ── Phase 2: daily NAV via numba kernel ──────────────────────
        nav_arr, turnover_arr = daily_nav_loop(
            self.returns.values.astype(np.float64),
            np.array(rebal_date_flags, dtype=np.bool_),
            np.array(rebal_weights_list, dtype=np.float64),
            np.array(base_weights_list, dtype=np.float64),
            float(self.initial_capital),
            float(TRANSACTION_COST),
            float(DD_CIRCUIT_BREAK),
            float(DD_RESUME_LEVEL),
            float(DD_SCALE_FACTOR),
        )

        # ── Phase 3: assemble result DataFrame ───────────────────────
        result = pd.DataFrame(
            {"nav": nav_arr, "turnover": turnover_arr},
            index=dates,
        )
        result["daily_return"] = result["nav"].pct_change()
        result["drawdown"]     = result["nav"] / result["nav"].cummax() - 1
        return result
