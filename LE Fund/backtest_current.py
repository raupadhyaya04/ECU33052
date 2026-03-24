import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from le_fund.core.data import DataEngine
from le_fund.core.universe import UniverseLoader

def run_backtest():
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║        HISTORICAL BACKTEST: CURRENT LOCKED PORTFOLIO      ║")
    print("╚═══════════════════════════════════════════════════════════╝\n")
    
    loader = UniverseLoader()
    tickers = loader.get_all_tickers()
    
    # Start from 2021 to capture most of these companies (COIN, NU, etc. IPO'd later)
    # Returns will automatically fill with 0 before a stock's IPO
    engine = DataEngine(tickers, start="2021-01-01")
    prices = engine.download()
    
    returns = prices.pct_change().fillna(0)
    
    locked_weights_dict = {
        'NU': 0.2155, 'ASML': 0.2137, 'AVGO': 0.2127, 'TSM': 0.2094,
        'ARM': 0.0263, 'XPEV': 0.0280, 'COIN': 0.0258, 'PLTR': 0.0268
    }
    
    w_locked = pd.Series(locked_weights_dict).reindex(returns.columns).fillna(0)
    
    # Assuming daily rebalancing back to your target weights
    port_returns = returns.dot(w_locked.values)
    
    # Metrics
    trading_days = len(port_returns)
    years = trading_days / 252.0
    
    cumulative_returns = (1 + port_returns).cumprod()
    total_return = cumulative_returns.iloc[-1] - 1
    
    cagr = (1 + total_return) ** (1 / years) - 1
    ann_vol = port_returns.std() * np.sqrt(252)
    sharpe = cagr / ann_vol if ann_vol > 0 else 0
    
    roll_max = cumulative_returns.cummax()
    drawdowns = cumulative_returns / roll_max - 1
    max_dd = drawdowns.min()
    
    initial_investment = 50000
    final_value = initial_investment * (1 + total_return)
    
    # Save the equity curve so you can chart it if needed
    dates = port_returns.index
    equity_df = pd.DataFrame({
        'Daily_Return': port_returns,
        'Cumulative_Return': cumulative_returns,
        'Portfolio_Value': cumulative_returns * initial_investment,
        'Drawdown': drawdowns
    }, index=dates)
    
    equity_df.to_csv('output/locked_portfolio_historical_backtest.csv')
    
    print(f"Backtest Period:           {dates[0].date()} to {dates[-1].date()} ({years:.2f} Years)")
    print(f"Initial Investment:        ${initial_investment:,.2f}")
    print(f"Final Value:               ${final_value:,.2f}")
    print("-" * 65)
    print(f"Total Cumulative Return:   {total_return:.2%}")
    print(f"Annualized Return (CAGR):  {cagr:.2%}")
    print(f"Annualized Volatility:     {ann_vol:.2%}")
    print(f"Sharpe Ratio:              {sharpe:.2f}")
    print(f"Max Drawdown:              {max_dd:.2%}")
    print("-" * 65)
    print("Pre-IPO Handling: If an asset wasn't publicly traded yet (e.g. ARM in 2021),")
    print("its weight was held securely as cash (0% return) until its IPO.\n")
    print("✅ Full daily equity curve saved to: 'output/locked_portfolio_historical_backtest.csv'")

if __name__ == "__main__":
    run_backtest()
