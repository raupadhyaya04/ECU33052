import pandas as pd
from le_fund.core.universe import UniverseLoader
from le_fund.core.data import DataEngine
from le_fund.core.risk import RiskEngine
from le_fund.core.fama_french import calculate_capm_ff3_betas

# Load universe
universe = UniverseLoader()
tickers = universe.get_all_tickers()
names = universe.get_ticker_names()

# Load data
data_engine = DataEngine(tickers)
prices = data_engine.download()

# Compute returns
returns = RiskEngine.compute_returns(prices)

# Compute betas
betas = calculate_capm_ff3_betas(returns, use_ff3=True)

# Print results
print(f"\n{'Ticker':<10} | {'Name':<25} | {'FF3 Beta':<10}")
print("-" * 50)
for ticker, beta in zip(returns.columns, betas):
    print(f"{ticker:<10} | {names.get(ticker, 'Unknown'):<25} | {beta:.3f}")
