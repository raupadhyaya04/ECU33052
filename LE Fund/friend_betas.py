import pandas as pd
import yfinance as yf
import numpy as np
from le_fund.core.fama_french import calculate_capm_ff3_betas

tickers = ["PG", "KO", "GSK", "UNH", "NEE", "MSFT", "GOOG", "META", "AMZN", "V", "JPM", "XOM", "TXT", "FFIV"]
weights = {"PG": 0.08, "KO": 0.07, "GSK": 0.07, "UNH": 0.09, "NEE": 0.05, 
           "MSFT": 0.08, "GOOG": 0.06, "META": 0.08, "AMZN": 0.10, "V": 0.08, 
           "JPM": 0.06, "XOM": 0.06, "TXT": 0.04, "FFIV": 0.05, "CASH": 0.03}

names = {
    "PG": "The Procter & Gamble Company", "KO": "The Coca-Cola Company", "GSK": "GSK plc", 
    "UNH": "United Health Group Incorporated", "NEE": "NextEra Energy, Inc.", 
    "MSFT": "Microsoft Corporation", "GOOG": "Alphabet Inc.", "META": "Meta Platforms, Inc.", 
    "AMZN": "Amazon.com, Inc.", "V": "Visa Inc.", "JPM": "JPMorgan Chase & Co.", 
    "XOM": "Exxon Mobil Corporation", "TXT": "Textron Inc.", "FFIV": "F5, Inc.", "CASH": "Cash reserves"
}

print("Downloading historical data...")
data = yf.download(tickers, start="2020-01-01", progress=False)['Close']
returns = data.pct_change().dropna()

print("Calculating FF3 betas...")
# calculate_capm_ff3_betas relies on the returns DataFrame's columns index
ff3_betas = calculate_capm_ff3_betas(returns, use_ff3=True)
beta_dict = {ticker: beta for ticker, beta in zip(returns.columns, ff3_betas)}

print("Fetching YF betas...")
yf_betas = {}
for t in tickers:
    try:
        info = yf.Ticker(t).info
        yf_betas[t] = info.get("beta", 0.0)
    except:
        yf_betas[t] = 0.0

# Calculate portfolio betas
port_ff3 = sum(weights[t] * beta_dict[t] for t in tickers)
port_yf = sum(weights[t] * yf_betas[t] for t in tickers)

print("\n| Name | Ticker | Weight | Calculated FF3 Beta | Yahoo Finance Beta |")
print("| :--- | :--- | :--- | :--- | :--- |")
for t in tickers: # Keep initial order
    print(f"| {names[t]} | **{t}** | {weights[t]*100:.0f}% | {beta_dict[t]:.3f} | {yf_betas[t]:.2f} |")
print(f"| Cash reserves | **CASH** | {weights['CASH']*100:.0f}% | 0.000 | 0.00 |")
print(f"| **Totals** | **TOTAL** | **100%** | **{port_ff3:.3f}** | **{port_yf:.3f}** |")

