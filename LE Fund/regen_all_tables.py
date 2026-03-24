import pandas as pd
import numpy as np
import yfinance as yf
import sys
import os

sys.path.append(os.path.abspath('le_fund'))
from core.fama_french import calculate_capm_ff3_betas

# Constants (1 Month)
RF_ANNUAL = 0.045
MRP_ANNUAL = 0.055
RF_MO = RF_ANNUAL / 12
MRP_MO = MRP_ANNUAL / 12
PORTFOLIO_TOTAL = 10000

def process_portfolio(pf_name, yf_tickers, display_tickers, names, weights, prices_initial, fixed_current_prices=None):
    current_prices = []
    sectors = []
    yf_betas = []
    
    for i, t in enumerate(yf_tickers):
        ticker = yf.Ticker(t)
        info = ticker.fast_info if hasattr(ticker, "fast_info") else ticker.info
        
        if fixed_current_prices is not None:
            current_prices.append(fixed_current_prices[i])
        else:
            hist = ticker.history(period="1mo")
            if not hist.empty:
                current_prices.append(hist['Close'].iloc[-1])
            else:
                current_prices.append(prices_initial[i]) # fallback
                
        sectors.append(ticker.info.get('sector', 'Unknown') if hasattr(ticker, "info") else "Unknown")
        yf_betas.append(ticker.info.get('beta', 1.0) if hasattr(ticker, "info") and ticker.info.get('beta') else 1.0)

    actual_returns = [(c - i) / i for c, i in zip(current_prices, prices_initial)]
    
    data = yf.download(yf_tickers, start="2022-01-01", end="2024-03-20")
    if isinstance(data.columns, pd.MultiIndex):
        data = data.loc[:, ('Close', slice(None))]
        data.columns = data.columns.droplevel(0)
    else:
        # single ticker fallback (not applicable here)
        pass
    
    returns = data.pct_change().dropna()
    returns = returns[yf_tickers]
    
    # 1-Month Volatility (sqrt(21) trading days approx)
    volatilities_mo = returns.std() * np.sqrt(21)
    
    ff3_betas_arr = calculate_capm_ff3_betas(returns, use_ff3=True)
    
    capm_er_mo = [RF_MO + b * MRP_MO for b in yf_betas]
    ff3_er_mo = [RF_MO + b * MRP_MO for b in ff3_betas_arr]
    
    capm_alphas = [a - e for a, e in zip(actual_returns, capm_er_mo)]
    ff3_alphas = [a - e for a, e in zip(actual_returns, ff3_er_mo)]
    
    sharpe_ratios = [(a - RF_MO) / v for a, v in zip(actual_returns, volatilities_mo.values)]
    treynor_ratios = [(a - RF_MO) / b if b != 0 else 0 for a, b in zip(actual_returns, ff3_betas_arr)]
    
    cash_weight = 1.0 - sum(weights)
    
    print(f"### Table 1: {pf_name} - Holdings & Returns")
    print("| **Name** | **Ticker** | **Sector** | **Weight (%)** | **Quantity** | **Initial Price** | **Current Price** | **1-Mo Actual Return (%)** |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    
    t1_rows = []
    t2_rows = []
    
    for i in range(len(yf_tickers)):
        qty = (weights[i] * PORTFOLIO_TOTAL) / prices_initial[i]
        print(f"| {names[i]} | **{display_tickers[i]}** | {sectors[i]} | {weights[i]*100:.2f}% | {qty:.2f} | ${prices_initial[i]:.2f} | ${current_prices[i]:.2f} | {actual_returns[i]*100:.2f}% |")
        
        t2_rows.append({
            "YF Beta": yf_betas[i], "FF3 Beta": ff3_betas_arr[i], "Vol": volatilities_mo.values[i],
            "CAPM E(R)": capm_er_mo[i], "FF3 E(R)": ff3_er_mo[i], 
            "CAPM Alpha": capm_alphas[i], "FF3 Alpha": ff3_alphas[i], 
            "Sharpe": sharpe_ratios[i], "Treynor": treynor_ratios[i]
        })
        
    if cash_weight > 0.0001:
        print(f"| Cash Reserves | **CASH** | Cash | {cash_weight*100:.2f}% | {cash_weight*PORTFOLIO_TOTAL:.2f} | $1.00 | $1.00 | 0.00% |")
    
    print(f"\n### Table 2: {pf_name} - 1-Month Risk-Adjusted Performance")
    print("| **Name** | **Ticker** | **CAPM Beta** | **FF3 Beta** | **1-Mo Volatility** | **1-Mo CAPM E(R)** | **1-Mo FF3 E(R)** | **1-Mo CAPM Alpha** | **1-Mo FF3 Alpha** | **1-Mo Sharpe Ratio** | **1-Mo Treynor Ratio** |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    
    for i in range(len(yf_tickers)):
        r = t2_rows[i]
        print(f"| {names[i]} | **{display_tickers[i]}** | {r['YF Beta']:.3f} | {r['FF3 Beta']:.3f} | {r['Vol']*100:.2f}% | {r['CAPM E(R)']*100:.2f}% | {r['FF3 E(R)']*100:.2f}% | {r['CAPM Alpha']*100:.2f}% | {r['FF3 Alpha']*100:.2f}% | {r['Sharpe']:.2f} | {r['Treynor']:.2f} |")
        
    if cash_weight > 0.0001:
        print(f"| Cash Reserves | **CASH** | 0.000 | 0.000 | 0.00% | {RF_MO*100:.2f}% | {RF_MO*100:.2f}% | {-RF_MO*100:.2f}% | {-RF_MO*100:.2f}% | 0.00 | 0.00 |")
        
    df_t2 = pd.DataFrame(t2_rows)
    all_weights = weights + ([cash_weight] if cash_weight > 0.0001 else [])
    
    w_yf_beta = np.average(df_t2["YF Beta"].tolist() + ([0] if cash_weight>0 else []), weights=all_weights)
    w_ff3_beta = np.average(df_t2["FF3 Beta"].tolist() + ([0] if cash_weight>0 else []), weights=all_weights)
    w_vol = np.average(df_t2["Vol"].tolist() + ([0] if cash_weight>0 else []), weights=all_weights)
    w_capm_er = np.average(df_t2["CAPM E(R)"].tolist() + ([RF_MO] if cash_weight>0 else []), weights=all_weights)
    w_ff3_er = np.average(df_t2["FF3 E(R)"].tolist() + ([RF_MO] if cash_weight>0 else []), weights=all_weights)
    w_capm_alpha = np.average(df_t2["CAPM Alpha"].tolist() + ([-RF_MO] if cash_weight>0 else []), weights=all_weights)
    w_ff3_alpha = np.average(df_t2["FF3 Alpha"].tolist() + ([-RF_MO] if cash_weight>0 else []), weights=all_weights)
    w_sharpe = np.average(df_t2["Sharpe"].tolist() + ([0] if cash_weight>0 else []), weights=all_weights)
    w_treynor = np.average(df_t2["Treynor"].tolist() + ([0] if cash_weight>0 else []), weights=all_weights)
    
    print(f"| **Weighted Average** | **PORTFOLIO** | **{w_yf_beta:.3f}** | **{w_ff3_beta:.3f}** | **{w_vol*100:.2f}%** | **{w_capm_er*100:.2f}%** | **{w_ff3_er*100:.2f}%** | **{w_capm_alpha*100:.2f}%** | **{w_ff3_alpha*100:.2f}%** | **{w_sharpe:.2f}** | **{w_treynor:.2f}** |")
    print("\n---\n")

# P1
pf1_tickers = ["NU", "ASML", "AVGO", "TSM", "ARM", "XPEV", "COIN", "PLTR"]
pf1_weights = [0.2155, 0.2137, 0.2127, 0.2094, 0.0263, 0.0280, 0.0258, 0.0268]
pf1_init = [16.65, 1526.51, 332.31, 387.73, 131.74, 18.18, 183.94, 134.19]
pf1_curr = [14.70, 1389.06, 325.13, 340.90, 137.64, 18.98, 202.47, 159.00]
pf1_names = ["Nu Holdings", "ASML Holding", "Broadcom", "TSMC", "ARM Holdings", "XPeng", "Coinbase", "Palantir"]
process_portfolio("Portfolio 1 (Yours)", pf1_tickers, pf1_tickers, pf1_names, pf1_weights, pf1_init, pf1_curr)

# P2
pf2_tickers = ["PG", "KO", "GSK", "UNH", "NEE", "MSFT", "GOOG", "META", "AMZN", "V", "JPM", "XOM", "TXT", "FFIV"]
pf2_weights = [0.08, 0.07, 0.07, 0.09, 0.05, 0.08, 0.06, 0.08, 0.10, 0.08, 0.06, 0.06, 0.04, 0.05]
pf2_init = [163.39, 80.47, 59.54, 284.20, 95.11, 400.60, 313.03, 653.69, 210.64, 312.99, 303.30, 149.06, 96.64, 278.55]
pf2_names = ["Procter & Gamble", "Coca-Cola", "GSK", "United Health", "NextEra Energy", "Microsoft", "Alphabet", "Meta Platforms", "Amazon", "Visa", "JPMorgan", "Exxon Mobil", "Textron", "F5"]
process_portfolio("Portfolio 2 (Friend 1)", pf2_tickers, pf2_tickers, pf2_names, pf2_weights, pf2_init, None)

# P3
pf3_yf = ["PG", "FNV", "LMT", "NOC", "BRK-B", "RSG", "IBE.MC", "ALV.DE", "MUV2.DE", "CVX", "NVS", "KO", "GSK", "NEE"]
pf3_disp = ["PG", "FNV", "LMT", "NOC", "BRK-B", "RSG", "IBE", "ALV.DE", "MUV2.DE", "CVX", "NVS", "KO", "GSK", "NEE"]
pf3_weights = [0.07, 0.07, 0.09, 0.07, 0.07, 0.06, 0.07, 0.07, 0.05, 0.07, 0.06, 0.06, 0.06, 0.07]
pf3_init = [163.67, 276.00, 648.00, 703.65, 493.99, 221.59, 24.03, 451.13, 657.98, 184.22, 166.85, 80.47, 59.54, 95.11]
pf3_names = ["Procter & Gamble", "Franco-Nevada", "Lockheed Martin", "Northrop Grumman", "Berkshire Hathaway", "Republic Services", "Iberdrola", "Allianz", "Munich Reinsurance", "Chevron", "Novartis", "Coca-Cola", "GSK", "NextEra Energy"]
process_portfolio("Portfolio 3 (Friend 2)", pf3_yf, pf3_disp, pf3_names, pf3_weights, pf3_init, None)

