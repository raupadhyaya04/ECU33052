import pandas as pd
import numpy as np
from dotenv import load_dotenv
load_dotenv()
from le_fund.core.data import DataEngine
from le_fund.core.risk import RiskEngine
from le_fund.core.universe import UniverseLoader
from le_fund.sentiment.engine import SentimentEngine

def run_stress_tests():
    print("╔═══════════════════════════════════════════════════════════╗")
    print("║            PORTFOLIO STRESS TESTING & SCENARIOS           ║")
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
    
    # Locked portfolio
    locked_weights_dict = {
        'NU': 0.2155, 'ASML': 0.2137, 'AVGO': 0.2127, 'TSM': 0.2094,
        'ARM': 0.0263, 'XPEV': 0.0280, 'COIN': 0.0258, 'PLTR': 0.0268
    }
    w_locked = pd.Series(locked_weights_dict).reindex(tickers).fillna(0).values
    
    def project_scenario(name, shock_returns, shock_vol_multiplier, initial_capital=50000):
        # Apply shocks
        scenario_returns = np.copy(base_exp_returns)
        scenario_cov = np.copy(base_cov)
        
        for i, t in enumerate(tickers):
            if t in shock_returns:
                scenario_returns[i] += shock_returns[t]
            
            if t in shock_vol_multiplier:
                # Scale the variance-covariance matrix for this asset
                mult = shock_vol_multiplier[t]
                scenario_cov[i, :] *= mult
                scenario_cov[:, i] *= mult
                
        # Recalculate metrics
        port_ret = np.dot(w_locked, scenario_returns)
        port_var = w_locked.T @ scenario_cov @ w_locked
        port_vol = np.sqrt(port_var * 252)
        cagr = port_ret - (port_var * 252) / 2
        
        # Guard against mathematically absurd negative compounding bounds
        cagr = max(cagr, -0.99)
        
        print(f"--- Scenario: {name} ---")
        print(f"Expected Annual CAGR: {cagr:.2%}  |  Expected Volatility: {port_vol:.2%}")
        
        vals = []
        for years in [1, 3, 10]:
            expected_wealth = initial_capital * ((1 + cagr) ** years)
            vals.append(expected_wealth)
            print(f"  {years:>2}-Year Projected Value: ${expected_wealth:>10,.2f}")
        print("-" * 60)
        return vals

    print("Baseline (No Shock)")
    project_scenario("Baseline (Current Quant Forecast)", {}, {})

    # Scenario 1: AI / Semiconductor Bubble Burst
    # Tech giants get a heavy expected return penalty (-25% absolute reduction) and 1.5x volatility
    ai_crash_ret = {t: -0.25 for t in ['ASML', 'AVGO', 'TSM', 'ARM', 'PLTR']}
    ai_crash_vol = {t: 1.5 for t in ['ASML', 'AVGO', 'TSM', 'ARM', 'PLTR']}
    project_scenario("AI & Semiconductor Crash (-25% Structural Hit, High Volatility)", ai_crash_ret, ai_crash_vol)

    # Scenario 2: Severe Broad Market Recession (Risk-Off)
    # High beta stocks suffer massive drawdowns.
    recession_ret = {t: -0.20 for t in tickers}
    recession_ret['COIN'] = -0.40 # Harder hit on purely speculative
    recession_ret['XPEV'] = -0.30
    recession_vol = {t: 1.8 for t in tickers} # Correlation and vol spikes universally
    project_scenario("Severe Global Recession (Broad Risk-Off, VIX Spike)", recession_ret, recession_vol)

    # Scenario 3: Higher-For-Longer / Stagflation
    # Growth multiples compress. Hard hits to emerging markets and unprofitable tech.
    stagflation_ret = {'NU': -0.15, 'XPEV': -0.20, 'PLTR': -0.15, 'ARM': -0.15, 'COIN': -0.20}
    stagflation_vol = {t: 1.2 for t in stagflation_ret.keys()}
    project_scenario("Stagflation / Prolonged High Interest Rates (Growth Multiple Compression)", stagflation_ret, stagflation_vol)

if __name__ == "__main__":
    run_stress_tests()
