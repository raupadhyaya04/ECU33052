import pandas as pd
import numpy as np
from dotenv import load_dotenv
load_dotenv()
from le_fund.core.data import DataEngine
from le_fund.core.risk import RiskEngine
from le_fund.core.universe import UniverseLoader
from le_fund.sentiment.engine import SentimentEngine

def run_forecast():
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║            PORTFOLIO EXPECTED FUTURE PERFORMANCE          ║")
    print("╚═══════════════════════════════════════════════════════════╝\n")
    
    loader = UniverseLoader()
    tickers = loader.get_all_tickers()
    names = loader.get_ticker_names()
    
    engine = DataEngine(tickers, start="2020-01-01")
    prices = engine.download()
    returns = RiskEngine.compute_returns(prices)
    
    sentiment_engine = SentimentEngine()
    sentiment = sentiment_engine.score_universe(tickers)
    
    # Blended Expected Returns (annualized)
    exp_returns = RiskEngine.blended_expected_returns(returns, prices, sentiment)
    # Expected Volatilities (annualized)
    ewma_vols = RiskEngine.ewma_volatility(returns).reindex(tickers).values
    # Covariance
    cov = RiskEngine.shrinkage_covariance(returns)
    
    # Hardcoded, original locked weights
    locked_weights_dict = {
        'NU': 0.2155,
        'ASML': 0.2137,
        'AVGO': 0.2127,
        'TSM': 0.2094,
        'ARM': 0.0263,
        'XPEV': 0.0280,
        'COIN': 0.0258,
        'PLTR': 0.0268
    }
    
    # Calculate total equity weight to scale cash correctly
    total_equity_weight = sum(locked_weights_dict.values())
    
    # Create the weights array for calculation 
    w_locked = pd.Series(locked_weights_dict).reindex(tickers).fillna(0).values
    
    def analyze(w, name, initial_capital=50000):
        if np.sum(w) == 0:
            return
        
        # Portfolio expected return (Arithmetic)
        port_ret = np.dot(w, exp_returns)
        # Portfolio risk 
        port_var = w.T @ cov.values @ w
        port_vol = np.sqrt(port_var * 252) # Assuming cov is daily
        sharpe = port_ret / port_vol if port_vol > 0 else 0
        
        # Geometric return (CAGR) accounting for volatility drag
        cagr = port_ret - (port_var * 252) / 2
        
        print(f"--- {name} Multi-Year Forecast ---")
        print(f"Expected Annual ROI (Arithmetic): {port_ret:.2%}")
        print(f"Expected Annual Volatility:       {port_vol:.2%}")
        print(f"Expected Annual CAGR (Geometric): {cagr:.2%}")
        print(f"Expected Sharpe Ratio:            {sharpe:.2f}")
        
        print(f"\nProjected Growth (Initial Investment: ${initial_capital:,.2f}):")
        for years in [1, 3, 10]:
            # Using Geometric return for long-term compound scaling
            expected_wealth = initial_capital * ((1 + cagr) ** years)
            cumulative_ret = (expected_wealth / initial_capital) - 1
            print(f"  {years:>2}-Year Horizon -> Expected Value: ${expected_wealth:>10,.2f} | Cumulative Return: {cumulative_ret:>7.2%}")
        
        print("\nAsset Breakdown:")
        print(f"{'Ticker':<8} | {'Weight':<8} | {'1Y Exp Return':<14} | {'Contribution to Risk':<12}")
        risk_contribs = RiskEngine.risk_contributions(w, cov.values)
        for i, t in enumerate(tickers):
            if w[i] > 0.0001:
                print(f"{t:<8} | {w[i]:.2%}   | {exp_returns[i]:.2%}           | {risk_contribs[i]:.2%}")
        
        cash_weight = 1.0 - total_equity_weight
        print(f"CASH     | {cash_weight:.2%}   | 0.00%            | 0.00%")
        print("="*75 + "\n")

    analyze(w_locked, "Locked Historical Portfolio")
    
if __name__ == "__main__":
    run_forecast()
