import numpy as np

# Portfolio parameters based on their FF3 returns and 1-month volatilities (annualized)
# 1-Mo vol * sqrt(12) = Annual Vol
portfolios = {
    "Portfolio 1 (Yours)":    {"ret_initial": 0.1416, "vol_ann": 0.1087 * np.sqrt(12)},
    "Portfolio 2 (Friend 1)": {"ret_initial": 0.0990, "vol_ann": 0.0819 * np.sqrt(12)},
    "Portfolio 3 (Friend 2)": {"ret_initial": 0.0744, "vol_ann": 0.0610 * np.sqrt(12)},
}

N_PATHS = 10000
YEARS = 10
N_DAYS = YEARS * 252
DT = 1 / 252

LONG_TERM_MEAN = 0.09 
HALF_LIFE_DAYS = 252 * 1.5
KAPPA = np.log(2) / HALF_LIFE_DAYS
SIGMA_ALPHA_ANN = 0.15
SIGMA_ALPHA = SIGMA_ALPHA_ANN / np.sqrt(252)
RHO = 0.60

np.random.seed(42) # Consistent random paths

# Pre-generate noise for speed (shared market macro sentiment shocks!)
Z_alpha = np.random.standard_normal((N_DAYS, N_PATHS))
Z_uncorr = np.random.standard_normal((N_DAYS, N_PATHS))
Z_price_shared = RHO * Z_alpha + np.sqrt(1 - RHO**2) * Z_uncorr

print("### Expected Future Performance Projections (Based on Custom Monte Carlo Engine)")
print(f"| **Portfolio** | **Initial E(R)** | **Structural Target** | **3-Year Expected Cumulative Return (Median)** | **10-Year Expected Cumulative Return (Median)** |")
print(f"| :--- | :--- | :--- | :--- | :--- |")

for name, params in portfolios.items():
    port_ret_initial = params["ret_initial"]
    port_vol = params["vol_ann"]
    
    INITIAL_ALPHA = port_ret_initial - LONG_TERM_MEAN
    
    alpha = np.zeros((N_DAYS, N_PATHS))
    alpha[0] = INITIAL_ALPHA
    for t in range(1, N_DAYS):
        alpha[t] = alpha[t-1] - (KAPPA * alpha[t-1]) + (SIGMA_ALPHA * Z_alpha[t])
        
    mu_t_paths = LONG_TERM_MEAN + alpha
    
    drift = (mu_t_paths - 0.5 * (port_vol ** 2)) * DT
    daily_log_returns = drift + (port_vol * np.sqrt(DT)) * Z_price_shared
    
    cumulative_log_returns = np.cumsum(daily_log_returns, axis=0)
    wealth_paths = np.exp(cumulative_log_returns) # Initial capital = 1.0 multiplier
    
    ret_3yr = np.percentile(wealth_paths[(3 * 252) - 1, :], 50) - 1
    ret_10yr = np.percentile(wealth_paths[(10 * 252) - 1, :], 50) - 1
    
    print(f"| {name} | {port_ret_initial*100:.2f}% | 9.00% | {ret_3yr*100:.2f}% | {ret_10yr*100:.2f}% |")

