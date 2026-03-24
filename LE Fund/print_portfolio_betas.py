import pandas as pd
import numpy as np
from le_fund.core.universe import UniverseLoader
from le_fund.core.data import DataEngine
from le_fund.core.risk import RiskEngine
from le_fund.core.fama_french import calculate_capm_ff3_betas
from le_fund.core.optimisation import OptimisationEngine

universe = UniverseLoader()
tickers = universe.get_all_tickers()
prices = DataEngine(tickers).download()
returns = RiskEngine.compute_returns(prices)

# Optimise again using FF3 since we just patched it
opt = OptimisationEngine(returns, prices.cov(), prices=prices)
w_sharpe = opt.max_sharpe()
w_minvol = opt.min_volatility()
w_rp = opt.risk_parity()

ff3_betas = opt.betas  # The newly computed FF3 betas array

def calc_port_beta(w, b, tickers_list):
    w_aligned = w.reindex(tickers_list).fillna(0).values
    return float(np.dot(w_aligned, b))

print(f"Max Sharpe FF3 Portfolio Beta: {calc_port_beta(w_sharpe, ff3_betas, opt.tickers):.3f}")
print(f"Min Volatility FF3 Portfolio Beta: {calc_port_beta(w_minvol, ff3_betas, opt.tickers):.3f}")
print(f"Risk Parity FF3 Portfolio Beta: {calc_port_beta(w_rp, ff3_betas, opt.tickers):.3f}")
