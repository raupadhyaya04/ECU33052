"""
le_fund/core/volatility.py
───────────────────────────
VolatilityTargeter: regime-aware position scaling to hit a target
annualised volatility.
"""

import numpy as np
import pandas as pd

from le_fund.config import TARGET_VOL, MAX_CASH


class VolatilityTargeter:
    """
    Scales portfolio weights so that the ex-ante portfolio volatility
    matches TARGET_VOL.

    Regime adjustments (asymmetric):
    · HIGH_VOL  → cut scale by 40 % (aggressive de-risk)
    · NORMAL    → no adjustment
    · LOW_VOL   → gentle 5 % uplift
    Scale is always capped at 1.0 (no leverage / borrowing).
    """

    def __init__(self, target_vol: float = TARGET_VOL):
        self.target_vol = target_vol

    def scale_weights(
        self,
        weights: pd.Series,
        cov: pd.DataFrame,
        regime: str,
    ) -> pd.Series:
        w       = weights.values
        sigma   = cov.loc[weights.index, weights.index].values
        port_vol = np.sqrt(w @ sigma @ w) * np.sqrt(252)   # annualised

        if port_vol < 1e-8:
            return weights

        scale = self.target_vol / port_vol

        if regime == "HIGH_VOL":
            scale *= 0.60
        elif regime == "LOW_VOL":
            scale *= 1.05

        scale = min(scale, 1.0)

        scaled      = weights * scale
        cash_weight = 1.0 - scaled.sum()

        # Enforce max cash constraint (and bias to minimize cash)
        if cash_weight > MAX_CASH:
            # Re-scale weights so cash = MAX_CASH
            scale_adj = (1.0 - MAX_CASH) / scaled.sum()
            scaled = scaled * scale_adj
            cash_weight = 1.0 - scaled.sum()
        elif 0 < cash_weight < MAX_CASH:
            # If cash is positive but less than MAX_CASH, try to minimize it (aggressive bias)
            scale_adj = 1.0 / scaled.sum()
            scaled = scaled * scale_adj
            cash_weight = 1.0 - scaled.sum()

        print(
            f"  [VolTarget] Portfolio vol: {port_vol:.1%}, "
            f"Scale: {scale:.2f}, Cash: {cash_weight:.1%}, Regime: {regime}"
        )
        return scaled
