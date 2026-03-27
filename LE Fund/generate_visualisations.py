import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('output/visualisations', exist_ok=True)

portfolios = {
    "LE Fund": {
        'NU': 0.2155, 'ASML': 0.2137, 'AVGO': 0.2127, 'TSM': 0.2094,
        'ARM': 0.0263, 'XPEV': 0.0280, 'COIN': 0.0258, 'PLTR': 0.0268, 'BIL': 0.0418 
    },
    "OW Fund": {
        'PG': 0.08, 'KO': 0.07, 'GSK': 0.07, 'UNH': 0.09, 'NEE': 0.05, 
        'MSFT': 0.08, 'GOOG': 0.06, 'META': 0.08, 'AMZN': 0.10, 'V': 0.08, 
        'JPM': 0.06, 'XOM': 0.06, 'TXT': 0.04, 'FFIV': 0.05, 'BIL': 0.03
    },
    "SH Fund": {
        'PG': 0.07, 'FNV': 0.07, 'LMT': 0.09, 'NOC': 0.07, 'BRK-B': 0.07, 
        'RSG': 0.06, 'IBE.MC': 0.07, 'ALV.DE': 0.07, 'MUV2.DE': 0.05, 'CVX': 0.07, 
        'NVS': 0.06, 'KO': 0.06, 'GSK': 0.06, 'NEE': 0.07, 'BIL': 0.06
    }
}

all_tickers = set()
for p in portfolios.values():
    all_tickers.update(p.keys())
all_tickers = list(all_tickers)

print("Fetching data from Yahoo Finance...")
data = yf.download(all_tickers, period="2y", interval="1d")['Adj Close']
data = data.ffill().dropna()
returns = data.pct_change().dropna()

# 1. PIE CHARTS
print("1. Generating Allocation Pie Charts...")
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('Fund vs Friends: Portfolio Asset Allocations', fontsize=22, weight='bold')

for ax, (name, weights) in zip(axes, portfolios.items()):
    labels = list(weights.keys())
    sizes = list(weights.values())
    
    def autopct_format(pct):
        return ('%1.1f%%' % pct) if pct > 3 else ''
        
    ax.pie(sizes, labels=labels, autopct=autopct_format, startangle=140, 
           textprops={'fontsize': 10}, colors=sns.color_palette('tab20'))
    ax.set_title(name, fontsize=16, weight='bold')

plt.tight_layout()
plt.savefig('output/visualisations/1_allocations.png', dpi=300)
plt.close()

# 2. HISTORICAL RETURNS
print("2. Generating Historical Cumulative Returns...")
cum_returns = pd.DataFrame(index=returns.index)

for name, weights in portfolios.items():
    port_returns = pd.Series(0.0, index=returns.index)
    for ticker, weight in weights.items():
        if ticker in returns.columns:
            port_returns += returns[ticker] * weight
    cum_returns[name] = (1 + port_returns).cumprod() - 1

plt.figure(figsize=(14, 7))
colors = ['#d62728', '#1f77b4', '#2ca02c']
for idx, col in enumerate(cum_returns.columns):
    plt.plot(cum_returns.index, cum_returns[col] * 100, label=col, linewidth=2, color=colors[idx])
    
plt.title('2-Year Historical Cumulative Backtest', fontsize=18, weight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Cumulative Return (%)', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.axhline(0, color='black', linewidth=1)
plt.tight_layout()
plt.savefig('output/visualisations/2_historical_returns.png', dpi=300)
plt.close()

# 3. RISK VS RETURN PLOT
print("3. Generating Risk vs Return Scatter Plot...")
stats = []
annualization_factor = 252

for name, weights in portfolios.items():
    port_returns = pd.Series(0.0, index=returns.index)
    for ticker, weight in weights.items():
        if ticker in returns.columns:
            port_returns += returns[ticker] * weight
            
    ann_return = port_returns.mean() * annualization_factor
    ann_vol = port_returns.std() * np.sqrt(annualization_factor)
    stats.append({
        'Portfolio': name,
        'Annualized Return': ann_return,
        'Annualized Volatility': ann_vol
    })

stats_df = pd.DataFrame(stats)

plt.figure(figsize=(12, 7))
for i, row in stats_df.iterrows():
    plt.scatter(row['Annualized Volatility']*100, row['Annualized Return']*100, 
                s=500, label=row['Portfolio'], color=colors[i], edgecolor='black', zorder=5)
    
    plt.annotate(row['Portfolio'], 
                 (row['Annualized Volatility']*100, row['Annualized Return']*100),
                 xytext=(15, 0), textcoords='offset points', fontsize=14, va='center', weight='bold')

plt.title('Risk vs Return Profile (2-Year Historical Annualized)', fontsize=18, weight='bold')
plt.xlabel('Annualized Volatility (%)', fontsize=14)
plt.ylabel('Annualized Return (%)', fontsize=14)
plt.grid(True, alpha=0.3, zorder=0)

min_vol = stats_df['Annualized Volatility'].min()*100
max_vol = stats_df['Annualized Volatility'].max()*100
plt.xlim(max(0, min_vol - 5), max_vol + 15)

# Add Sharpe Ratio conceptual lines
plt.plot([0, max_vol + 20], [4.5, 4.5 + (0.5 * (max_vol + 20))], 'k--', alpha=0.3, label='Sharpe = 0.5 (Base)')
plt.plot([0, max_vol + 20], [4.5, 4.5 + (1.0 * (max_vol + 20))], 'k-.', alpha=0.3, label='Sharpe = 1.0 (Good)')

plt.legend(fontsize=12, loc='upper left')
plt.tight_layout()
plt.savefig('output/visualisations/3_risk_vs_return.png', dpi=300)
plt.close()

# 4. CORRELATION HEATMAPS
print("4. Generating Correlation Heatmaps...")
def plot_corr(port_name, p_dict, filename):
    tickers = [t for t in p_dict.keys() if t in returns.columns]
    if not tickers: return
    corr_matrix = returns[tickers].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='RdYlGn', vmin=-0.2, vmax=1, fmt=".2f",
                linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title(f'Asset Correlation Matrix: {port_name}', fontsize=18, weight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

plot_corr("LE Fund", portfolios["LE Fund"], 'output/visualisations/4_corr_le_fund.png')
plot_corr("OW Fund", portfolios["OW Fund"], 'output/visualisations/5_corr_ow_fund.png')
plot_corr("SH Fund", portfolios["SH Fund"], 'output/visualisations/6_corr_sh_fund.png')

# 5. MONTE CARLO CONES (10 YEARS)
print("5. Generating Monte Carlo Value Projections (10 Years)...")
mc_params = {
    "LE Fund": {"mu": 0.1416, "vol": 0.1087 * np.sqrt(12)},
    "OW Fund": {"mu": 0.0990, "vol": 0.0819 * np.sqrt(12)},
    "SH Fund":  {"mu": 0.0744, "vol": 0.0610 * np.sqrt(12)}
}

N_PATHS = 1000
YEARS = 10
N_DAYS = YEARS * 252
DT = 1 / 252

fig, axes = plt.subplots(1, 3, figsize=(22, 7), sharey=True)
fig.suptitle('10-Year Monte Carlo Wealth Projections (Initial Capital: $50,000)', fontsize=22, weight='bold')

np.random.seed(42)
for ax, ((name, params), color) in zip(zip(axes, mc_params.items()), colors):
    mu = params["mu"]
    vol = params["vol"]
    
    Z = np.random.standard_normal((N_DAYS, N_PATHS))
    drift = (mu - 0.5 * vol**2) * DT
    sim_returns = drift + vol * np.sqrt(DT) * Z
    
    wealth = 50000 * np.exp(np.cumsum(sim_returns, axis=0))
    wealth = np.vstack([np.ones((1, N_PATHS)) * 50000, wealth])
    
    p5 = np.percentile(wealth, 5, axis=1)
    p50 = np.percentile(wealth, 50, axis=1)
    p95 = np.percentile(wealth, 95, axis=1)
    
    time_arr = np.linspace(0, YEARS, N_DAYS + 1)
    
    ax.fill_between(time_arr, p5, p95, color=color, alpha=0.15, label='90% Confidence Interval')
    ax.plot(time_arr, p50, color=color, linewidth=3, label='Median Expected Path')
    
    for i in range(15):
        ax.plot(time_arr, wealth[:, i], color=color, alpha=0.05, linewidth=1)
        
    ax.set_title(f"{name}\nAssumed E(R): {mu*100:.1f}% | Vol: {vol*100:.1f}%", fontsize=15, weight='bold')
    ax.set_xlabel('Years', fontsize=12)
    ax.set_yscale('log')
    
    import matplotlib.ticker as ticker
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, pos: f'${y:,.0f}'))
    
    if ax == axes[0]:
        ax.set_ylabel('Portfolio Value (USD) - Log Scale', fontsize=14)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='upper left', fontsize=11)

plt.tight_layout()
plt.savefig('output/visualisations/7_monte_carlo_projections.png', dpi=300)
plt.close()

print("✅ Success! All visualisations generated dynamically.")
