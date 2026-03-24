import numpy as np

# Portfolio 1-Mo FF3 E(R) from your tables
portfolios = {
    "Portfolio 1 (Yours)": {"ff3_1mo": 0.0111},
    "Portfolio 2 (Friend 1)": {"ff3_1mo": 0.0079},
    "Portfolio 3 (Friend 2)": {"ff3_1mo": 0.0060},
}

print("### Expected Future Performance Projections (Based on Fama-French 3-Factor)")
print(f"| **Portfolio** | **Annual FF3 E(R)** | **3-Year Expected Cumulative Return** | **10-Year Expected Cumulative Return** |")
print(f"| :--- | :--- | :--- | :--- |")

for name, data in portfolios.items():
    # Convert 1-Month expected return down to nominal annual, then use the same compounding
    # In finance you normally compound the monthly expected return:
    annual_er = ((1 + data["ff3_1mo"]) ** 12) - 1
    
    cum_3yr = ((1 + annual_er) ** 3) - 1
    cum_10yr = ((1 + annual_er) ** 10) - 1
    
    print(f"| {name} | {annual_er*100:.2f}% | {cum_3yr*100:.2f}% | {cum_10yr*100:.2f}% |")

