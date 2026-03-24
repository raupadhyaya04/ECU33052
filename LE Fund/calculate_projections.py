import numpy as np

# Annual Assumptions
rf_annual = 0.045
mrp_annual = 0.055

# Portfolio Betas (from previous tables)
portfolios = {
    "Portfolio 1 (Yours)": {"beta": 1.365},
    "Portfolio 2 (Friend 1)": {"beta": 0.779},
    "Portfolio 3 (Friend 2)": {"beta": 0.462},
}

print("### Expected Future Performance Projections (Based on CAPM)")
print(f"| **Portfolio** | **Annual CAPM E(R)** | **3-Year Expected Cumulative Return** | **10-Year Expected Cumulative Return** |")
print(f"| :--- | :--- | :--- | :--- |")

for name, data in portfolios.items():
    beta = data["beta"]
    # Annual expected return
    annual_er = rf_annual + beta * mrp_annual
    
    # Cumulative returns
    cum_3yr = ((1 + annual_er) ** 3) - 1
    cum_10yr = ((1 + annual_er) ** 10) - 1
    
    print(f"| {name} | {annual_er*100:.2f}% | {cum_3yr*100:.2f}% | {cum_10yr*100:.2f}% |")

