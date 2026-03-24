import pandas as pd
import yfinance as yf
import numpy as np
from le_fund.core.fama_french import calculate_capm_ff3_betas

tickers = ["PG", "FNV", "LMT", "NOC", "BRK-B", "RSG", "IBE.MC", "ALV.DE", "MUV2.DE", "CVX", "NVS", "KO", "GSK", "NEE"]
weights = {
    "PG": 0.07, "FNV": 0.07, "LMT": 0.09, "NOC": 0.07, "BRK-B": 0.07,
    "RSG": 0.06, "IBE.MC": 0.07, "ALV.DE": 0.07, "MUV2.DE": 0.05, 
    "CVX": 0.07, "NVS": 0.06, "KO": 0.06, "GSK": 0.06, "NEE": 0.07, "CASH": 0.05
}

names = {
    "PG": "THE PROCTER & GAMBLE COMPANY",
    "FNV": "Franco-Nevada Corporation",
    "LMT": "LOCKHEED MARTIN CORPORATION",
    "NOC": "NORTHROP GRUMMAN CORPORATION",
    "BRK-B": "BERKSHIRE HATHAWAY INC.",
    "RSG": "REPUBLIC SERVICES, INC.",
    "IBE.MC": "Iberdrola SA",   # YF uses .MC for Spain or sometimes .BME
    "ALV.DE": "Allianz SE",
    "MUV2.DE": "Munich Reinsurance Company",
    "CVX": "CHEVRON CORPORATION",
    "NVS": "Novartis Inc.",
    "KO": "The Coca-Cola Company",
    "GSK": "GSK plc",
    "NEE": "NextEra Energy, Inc.",
    "CASH": "Cash Reserve"
}

ticker_display = {
    "PG": "PG", "FNV": "FNV", "LMT": "LMT", "NOC": "NOC", "BRK-B": "BRK-B",
    "RSG": "RSG", "IBE.MC": "IBE", "ALV.DE": "ALV.DE", "MUV2.DE": "MUV2.DE", 
    "CVX": "CVX", "NVS": "NVS", "KO": "KO", "GSK": "GSK", "NEE": "NEE"
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
    print(f"| {names[t]} | **{ticker_display[t]}** | {weights[t]*100:.0f}% | {beta_dict[t]:.3f} | {yf_betas[t]:.2f} |")
print(f"| Cash reserves | **CASH** | {weights['CASH']*100:.0f}% | 0.000 | 0.00 |")
print(f"| **Totals** | **TOTAL** | **100%** | **{port_ff3:.3f}** | **{port_yf:.3f}** |")

