import pandas as pd
import numpy as np
from dotenv import load_dotenv
load_dotenv()
from le_fund.core.data import DataEngine
from le_fund.core.risk import RiskEngine
from le_fund.core.universe import UniverseLoader
from le_fund.sentiment.engine import SentimentEngine

def run_monte_carlo_with_sentiment():
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║     MONTE CARLO FORECAST: STOCHASTIC SENTIMENT RIDES      ║")
    print("╚═══════════════════════════════════════════════════════════╝\n")
    
    loader = UniverseLoader()
    tickers = loader.get_all_tickers()
    
    engine = DataEngine(tickers, start="2020-01-01")
    prices = engine.download()
    returns = RiskEngine.compute_returns(prices)
    
    sentiment_engine = SentimentEngine()
    sentiment = sentiment_engine.score_universe(tickers)
    
    # Baseline expected variables
    base_exp_returns = RiskEngine.blended_expected_returns(returns, prices, sentiment)
    base_cov = RiskEngine.shrinkage_covariance(returns).values
    
    locked_weights_dict = {
        'NU': 0.2155, 'ASML': 0.2137, 'AVGO': 0.2127, 'TSM': 0.2094,
        'ARM': 0.0263, 'XPEV': 0.0280, 'COIN': 0.0258, 'PLTR': 0.0268
    }
    w_locked = pd.Series(locked_weights_dict).reindex(tickers).fillna(0).values
    
    port_ret_initial = np.dot(w_locked, base_exp_returns)
    port_var = w_locked.T @ base_cov @ w_locked
    port_vol = np.sqrt(port_var * 252)
    
    # ==========================================
    # STOCHASTIC SENTIMENT ENGINE (Ornstein-Uhlenbeck Process)
    # ==========================================
    N_PATHS = 10000
    YEARS = 10
    N_DAYS = YEARS * 252
    DT = 1 / 252
    INITIAL_CAPITAL = 50000
    
    LONG_TERM_MEAN = 0.09 
    
    # We treat the difference between today's high return and the long-term
    # average as 'Alpha' generated purely by hype/momentum/current sentiment.
    INITIAL_ALPHA = port_ret_initial - LONG_TERM_MEAN
    
    # Behavior parameters of Market Sentiment:
    HALF_LIFE_DAYS = 252 * 1.5         # Sentiment mean-reverts (decays) naturally over 1.5 years
    KAPPA = np.log(2) / HALF_LIFE_DAYS # Speed of reversion to baseline
    
    SIGMA_ALPHA_ANN = 0.15             # Sentiment has 15% volatility out of nowhere
    SIGMA_ALPHA = SIGMA_ALPHA_ANN / np.sqrt(252) # Daily sentiment shock volatility
    
    # When sentiment crashes, price crashes. We model a 60% correlation between these.
    RHO = 0.60 
    
    np.random.seed(42)
    
    # 1. Random noise for Sentiment Shocks
    Z_alpha = np.random.standard_normal((N_DAYS, N_PATHS))
    
    # 2. Random noise for Price Shocks (partially correlated to sentiment)
    Z_uncorr = np.random.standard_normal((N_DAYS, N_PATHS))
    Z_price = RHO * Z_alpha + np.sqrt(1 - RHO**2) * Z_uncorr
    
    # 3. Simulate Sentiment Alpha Paths (using Euler-Maruyama for OU)
    alpha = np.zeros((N_DAYS, N_PATHS))
    alpha[0] = INITIAL_ALPHA
    for t in range(1, N_DAYS):
        # The new day's sentiment alpha is yesterday's, minus the natural decay, plus a random new news shock
        alpha[t] = alpha[t-1] - (KAPPA * alpha[t-1]) + (SIGMA_ALPHA * Z_alpha[t])
        
    # The expected drift of the stock on any given day is the structural mean + that day's sentiment alpha score
    mu_t_paths = LONG_TERM_MEAN + alpha
    
    # ==========================================
    # APPLY TO PRICES & COMPOUND
    # ==========================================
    drift = (mu_t_paths - 0.5 * (port_vol ** 2)) * DT
    daily_log_returns = drift + (port_vol * np.sqrt(DT)) * Z_price
    
    cumulative_log_returns = np.cumsum(daily_log_returns, axis=0)
    wealth_paths = INITIAL_CAPITAL * np.exp(cumulative_log_returns)
    
    # ==========================================
    # RESULTS EXTRACTION
    # ==========================================
    def print_horizon_stats(year):
        day_idx = (year * 252) - 1
        path_values = wealth_paths[day_idx, :]
        
        p5 = np.percentile(path_values, 5)
        p25 = np.percentile(path_values, 25) 
        p50 = np.percentile(path_values, 50)
        p75 = np.percentile(path_values, 75)
        p95 = np.percentile(path_values, 95)
        
        prob_loss = np.mean(path_values < INITIAL_CAPITAL) * 100
        
        # Analyze the distribution of sentiment at this milestone in the future
        avg_alpha_at_t = np.mean(alpha[day_idx, :])
        std_alpha_at_t = np.std(alpha[day_idx, :])
        
        print(f"--- Horizon: {year} Year(s) ---")
        print(f"Underlying Sentiment State at end of period:")
        print(f"  • Median Market Expected Return Driven by Sentiment: {LONG_TERM_MEAN + avg_alpha_at_t:.2%}")
        print(f"  • Widest Range of Expected Returns from Sentiment:   {(LONG_TERM_MEAN + avg_alpha_at_t) - (2*std_alpha_at_t):.2%}  <  to  >  {(LONG_TERM_MEAN + avg_alpha_at_t) + (2*std_alpha_at_t):.2%}")
        print(f"  • Probability of Absolute Financial Loss:            {prob_loss:.1f}%\n")
        
        print(f"Portfolio Value Percentiles:")
        print(f"  🔴  5th Percentile (AI Winter / Terrible Sentiment) : ${p5:>10,.2f} ({(p5/INITIAL_CAPITAL)-1:+.2%})")
        print(f"  🟠 25th Percentile (Mild Bear Market)               : ${p25:>10,.2f}")
        print(f"  🟡 50th Percentile (Median Path)                    : ${p50:>10,.2f} ({(p50/INITIAL_CAPITAL)-1:+.2%})")
        print(f"  🟢 75th Percentile (Mild Bull Market)               : ${p75:>10,.2f}")
        print(f"  🌟 95th Percentile (Sustained Hype Cycle)           : ${p95:>10,.2f} ({(p95/INITIAL_CAPITAL)-1:+.2%})")
        print("=" * 75 + "\n")

    print(f"Starting Baseline Expected Return (Today's Sentiment): {port_ret_initial:.2%}")
    print(f"Structural Baseline Return (Zero Sentiment Target):    {LONG_TERM_MEAN:.2%}")
    print(f"Initial Investment:                                  ${INITIAL_CAPITAL:,.2f}\n")
    
    print_horizon_stats(1)
    print_horizon_stats(3)
    print_horizon_stats(10)
    
if __name__ == "__main__":
    run_monte_carlo_with_sentiment()
