import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from le_fund.core.data import DataEngine
from le_fund.core.risk import RiskEngine
from le_fund.core.universe import UniverseLoader
from le_fund.core.optimisation import OptimisationEngine
import time

def run_walk_forward():
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║     OUT-OF-SAMPLE WALK-FORWARD VALIDATION (BACKTEST)      ║")
    print("╚═══════════════════════════════════════════════════════════╝\n")
    print("Loading data...")
    
    loader = UniverseLoader()
    tickers = loader.get_all_tickers()
    
    # We want a long lookback for a good backtest
    engine = DataEngine(tickers, start="2020-01-01")
    prices = engine.download()
    
    returns = prices.pct_change().dropna()
    
    # Backtest parameters
    TRAIN_WINDOW = 252 # 1 year of data to train the optimizer
    STEP = 63          # Rebalance every 1 quarter (approx 63 trading days)
    
    print(f"Total trading days available: {len(returns)}")
    print(f"Training Window: {TRAIN_WINDOW} days | Rebalance Frequency: {STEP} days")
    print(f"Universe Size: {len(tickers)} stocks\n")
    
    if len(returns) < TRAIN_WINDOW + STEP:
        print("Not enough data to run walk-forward.")
        return
        
    oos_portfolio_returns = []
    oos_dates = []
    
    # Also track an equal weight benchmark for the exact same out-of-sample periods
    ew_portfolio_returns = []
    
    print("Running Rolling Windows (This may take a minute)...\n")
    
    for start_idx in range(0, len(returns) - TRAIN_WINDOW, STEP):
        train_end = start_idx + TRAIN_WINDOW
        test_end = min(train_end + STEP, len(returns))
        
        train_returns = returns.iloc[start_idx:train_end]
        train_prices = prices.iloc[start_idx:train_end]
        
        test_returns = returns.iloc[train_end:test_end]
        
        # 1. OPTIMIZE weights using ONLY TRAIN DATA (No hindsight!)
        cov = RiskEngine.shrinkage_covariance(train_returns)
        opt = OptimisationEngine(train_returns, cov, max_weight=0.3, max_positions=15)
        
        # Override the optimizer's fallback so it doesn't crash if it can't find a solution
        try:
            w_opt = opt.max_sharpe()
        except:
            print(f"Optimizer failed to converge at {train_returns.index[-1].date()}. Falling back to Equal Weight.")
            w_opt = pd.Series(1/len(tickers), index=tickers)
            
        # 2. TEST on unseen future data
        # Ensure alignment
        w_opt = w_opt.reindex(test_returns.columns).fillna(0)
        
        period_oos_returns = test_returns.dot(w_opt.values)
        oos_portfolio_returns.extend(period_oos_returns)
        oos_dates.extend(test_returns.index)
        
        # Benchmark: Equal Weight of the exact same universe out-of-sample
        ew_weights = np.ones(len(tickers)) / len(tickers)
        ew_period_returns = test_returns.dot(ew_weights)
        ew_portfolio_returns.extend(ew_period_returns)
        
        date_str = train_returns.index[-1].strftime("%Y-%m-%d")
        print(f"Optimized on {date_str} -> Top allocation: {w_opt.idxmax()} ({w_opt.max():.1%})")

    # Reconstruct timeline
    oos_series = pd.Series(oos_portfolio_returns, index=oos_dates)
    ew_series = pd.Series(ew_portfolio_returns, index=oos_dates)
    
    # Calculate performance metrics
    def calc_metrics(series):
        ann_ret = series.mean() * 252
        ann_vol = series.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        cum_ret = (1 + series).prod() - 1
        
        # Max Drawdown
        roll_max = (1 + series).cumprod().cummax()
        drawdown = (1 + series).cumprod() / roll_max - 1
        max_dd = drawdown.min()
        
        return ann_ret, ann_vol, sharpe, cum_ret, max_dd

    opt_ret, opt_vol, opt_sharpe, opt_cum, opt_dd = calc_metrics(oos_series)
    ew_ret, ew_vol, ew_sharpe, ew_cum, ew_dd = calc_metrics(ew_series)
    
    # Calculate the "Overfit" Baseline we did before (Full in-sample data)
    # Using the 8 locked weights over the full out-of-sample timeframe
    locked_weights_dict = {
        'NU': 0.2155, 'ASML': 0.2137, 'AVGO': 0.2127, 'TSM': 0.2094,
        'ARM': 0.0263, 'XPEV': 0.0280, 'COIN': 0.0258, 'PLTR': 0.0268
    }
    w_locked = pd.Series(locked_weights_dict).reindex(tickers).fillna(0).values
    # Test locked weights over the exact same date range as OOS testing
    locked_test_returns = returns.reindex(oos_dates)
    locked_series = locked_test_returns.dot(w_locked)
    lock_ret, lock_vol, lock_sharpe, lock_cum, lock_dd = calc_metrics(locked_series)
    
    print("\n" + "="*75)
    print(" 🚨 THE TRUTH: OUT-OF-SAMPLE VS BACKWARD-LOOKING OVERFIT 🚨 ")
    print("="*75)
    
    print(f"\n1) THE FANTASY (In-Sample Overfit / Hindsight Bias):")
    print(f"   What happens if you perfectly predicted the 2024 winners back in 2021.")
    print(f"   - Annualized Return: {lock_ret:.2%}")
    print(f"   - Annualized Volatility: {lock_vol:.2%}")
    print(f"   - Realized Sharpe: {lock_sharpe:.2f}")
    print(f"   - Max Drawdown: {lock_dd:.2%}")
    print(f"   - Total Growth: +{lock_cum:.2%}")

    print(f"\n2) THE REALITY (Out-Of-Sample Walk Forward):")
    print(f"   What the Optimizer actually chose BEFORE seeing the future.")
    print(f"   - Annualized Return: {opt_ret:.2%}")
    print(f"   - Annualized Volatility: {opt_vol:.2%}")
    print(f"   - Realized Sharpe: {opt_sharpe:.2f}")
    print(f"   - Max Drawdown: {opt_dd:.2%}")
    print(f"   - Total Growth: +{opt_cum:.2%}")

    print(f"\n3) THE BASELINE (Dumb Equal-Weighting):")
    print(f"   If you're just randomly buying these 8 stocks without an optimizer.")
    print(f"   - Annualized Return: {ew_ret:.2%}")
    print(f"   - Realized Sharpe: {ew_sharpe:.2f}")
    print(f"   - Total Growth: +{ew_cum:.2%}")
    print("="*75)

if __name__ == "__main__":
    run_walk_forward()
