"""
le_fund/core/numba_kernels.py
──────────────────────────────
Numba JIT-compiled inner loops.

These are hot paths called thousands of times per optimisation run.
Compiling them to machine code via numba gives ~50× speed-up over
pure Python/NumPy for the tight loops inside the optimisers.

A lightweight stub is provided so the whole package degrades gracefully
(slower but functional) when numba is not installed.
"""

import numpy as np

try:
    import numba
except Exception:
    class _NumbaStub:
        def njit(self, *args, **kwargs):
            def _decorator(func):
                return func
            return _decorator
    numba = _NumbaStub()  # type: ignore[assignment]


# ── CVaR ────────────────────────────────────────────────────────────────

@numba.njit(cache=True, fastmath=True)
def cvar_nb(returns_sorted: np.ndarray, alpha: float) -> float:
    """CVaR on a PRE-SORTED (ascending) returns array."""
    cutoff = max(1, int(len(returns_sorted) * alpha))
    total = 0.0
    for i in range(cutoff):
        total += returns_sorted[i]
    return -(total / cutoff)


# ── Risk contributions ───────────────────────────────────────────────────

@numba.njit(cache=True, fastmath=True)
def risk_contributions_nb(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Marginal risk contributions, normalised to sum to 1."""
    port_var = 0.0
    n = len(weights)
    for i in range(n):
        for j in range(n):
            port_var += weights[i] * cov[i, j] * weights[j]
    port_vol = port_var ** 0.5
    rc = np.empty(n)
    for i in range(n):
        mc = 0.0
        for j in range(n):
            mc += cov[i, j] * weights[j]
        rc[i] = weights[i] * mc / port_vol
    total_rc = 0.0
    for i in range(n):
        total_rc += rc[i]
    for i in range(n):
        rc[i] /= total_rc
    return rc


# ── Semi-variance ────────────────────────────────────────────────────────

@numba.njit(cache=True, fastmath=True)
def semi_variance_nb(returns: np.ndarray) -> np.ndarray:
    """Downside semi-covariance matrix."""
    T, N = returns.shape
    mu = np.empty(N)
    for j in range(N):
        s = 0.0
        for i in range(T):
            s += returns[i, j]
        mu[j] = s / T
    below = np.empty_like(returns)
    for i in range(T):
        for j in range(N):
            d = returns[i, j] - mu[j]
            below[i, j] = d if d < 0.0 else 0.0
    semi_cov = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            s = 0.0
            for t in range(T):
                s += below[t, i] * below[t, j]
            semi_cov[i, j] = s / (T - 1)
    return semi_cov


# ── Daily NAV loop ───────────────────────────────────────────────────────

@numba.njit(cache=True, fastmath=True)
def daily_nav_loop(
    daily_returns: np.ndarray,      # (T, N)
    rebal_mask: np.ndarray,         # (T,) bool
    rebal_weights: np.ndarray,      # (R, N)
    base_weights_arr: np.ndarray,   # (R, N)
    initial_nav: float,
    transaction_cost: float,
    dd_circuit_break: float,
    dd_resume_level: float,
    dd_scale_factor: float,
):
    """Full daily NAV simulation compiled to machine code."""
    T, N = daily_returns.shape
    nav_arr = np.empty(T)
    turnover_arr = np.zeros(T)

    nav = initial_nav
    peak_nav = nav
    circuit_active = False
    prev_w = np.zeros(N)
    base_w = np.zeros(N)
    rebal_idx = 0

    for t in range(T):
        if rebal_mask[t] and rebal_idx < rebal_weights.shape[0]:
            new_w = rebal_weights[rebal_idx]
            base_w = base_weights_arr[rebal_idx]
            rebal_idx += 1

            turnover = 0.0
            for i in range(N):
                d = new_w[i] - prev_w[i]
                if d < 0.0:
                    turnover -= d
                else:
                    turnover += d
            tc = turnover * transaction_cost * nav
            nav -= tc
            turnover_arr[t] = turnover

            for i in range(N):
                prev_w[i] = new_w[i]

        if nav > peak_nav:
            peak_nav = nav
        current_dd = nav / peak_nav - 1.0

        if current_dd < dd_circuit_break and not circuit_active:
            circuit_active = True
            for i in range(N):
                prev_w[i] = base_w[i] * dd_scale_factor
        elif current_dd > dd_resume_level and circuit_active:
            circuit_active = False
            for i in range(N):
                prev_w[i] = base_w[i]

        port_ret = 0.0
        for i in range(N):
            port_ret += prev_w[i] * daily_returns[t, i]
        nav *= (1.0 + port_ret)
        nav_arr[t] = nav

    return nav_arr, turnover_arr


# ── Sharpe objective ─────────────────────────────────────────────────────

@numba.njit(cache=True, fastmath=True)
def sharpe_objective_nb(
    w: np.ndarray,
    mu_sub: np.ndarray,
    semi_sigma_sub: np.ndarray,
    daily_rets_sorted: np.ndarray,   # (T, n_sub) each column pre-sorted ascending
    risk_free: float,
    cvar_alpha: float,
    dd_penalty: float,
    min_portfolio_beta: float,
    betas_sub: np.ndarray,
) -> float:
    """Negative penalised Sharpe — called inside scipy's line-search."""
    n = len(w)
    T = daily_rets_sorted.shape[0]

    ret = 0.0
    for i in range(n):
        ret += w[i] * mu_sub[i]

    down_var = 0.0
    for i in range(n):
        for j in range(n):
            down_var += w[i] * semi_sigma_sub[i, j] * w[j]
    down_vol = (max(down_var, 1e-12) * 252.0) ** 0.5

    cutoff = max(1, int(T * cvar_alpha))
    port_rets = np.empty(T)
    for t in range(T):
        s = 0.0
        for i in range(n):
            s += w[i] * daily_rets_sorted[t, i]
        port_rets[t] = s

    worst = port_rets[:cutoff].copy()
    for t in range(cutoff, T):
        if port_rets[t] < worst[-1]:
            worst[-1] = port_rets[t]
            k = cutoff - 1
            while k > 0 and worst[k] < worst[k - 1]:
                worst[k], worst[k - 1] = worst[k - 1], worst[k]
                k -= 1
    cvar_val = 0.0
    for i in range(cutoff):
        cvar_val -= worst[i]
    cvar_val = (cvar_val / cutoff) * (252.0 ** 0.5)

    sharpe = (ret - risk_free) / down_vol if down_vol > 1e-8 else 0.0
    return -(sharpe - dd_penalty * cvar_val)


# ── Risk parity objective ────────────────────────────────────────────────

@numba.njit(cache=True, fastmath=True)
def risk_parity_objective_nb(
    w: np.ndarray,
    sigma_sub: np.ndarray,
    target_rc: float,
) -> float:
    """Sum of squared deviations from equal risk budget."""
    n = len(w)
    port_var = 0.0
    for i in range(n):
        for j in range(n):
            port_var += w[i] * sigma_sub[i, j] * w[j]
    if port_var < 1e-12:
        return 1e6
    port_vol = port_var ** 0.5

    rc_sum = 0.0
    rc = np.empty(n)
    for i in range(n):
        mc = 0.0
        for j in range(n):
            mc += sigma_sub[i, j] * w[j]
        rc[i] = w[i] * mc / port_vol
        rc_sum += rc[i]

    total = 0.0
    for i in range(n):
        diff = rc[i] / rc_sum - target_rc
        total += diff * diff
    return total
