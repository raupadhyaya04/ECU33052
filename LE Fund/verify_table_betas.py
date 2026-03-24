weights = {
    'ARM': 0.0263,
    'ASML': 0.2137,
    'AVGO': 0.2127,
    'COIN': 0.0258,
    'NU': 0.2155,
    'PLTR': 0.0268,
    'TSM': 0.2094,
    'XPEV': 0.0280
}

calc_beta = {
    'ARM': 2.157,
    'ASML': 1.491,   # Your table says 1.43, catching that
    'AVGO': 1.784,
    'COIN': 2.191,
    'NU': 1.175,
    'PLTR': 2.07,
    'TSM': 1.41,
    'XPEV': 0.737
}

yf_beta = {
    'ARM': 4.13,
    'ASML': 1.43,
    'AVGO': 1.26,
    'COIN': 3.71,
    'NU': 1.11,
    'PLTR': 1.74,
    'TSM': 1.28,
    'XPEV': 1.2
}

calc_port_beta = sum(weights[t] * calc_beta[t] for t in weights)
yf_port_beta = sum(weights[t] * yf_beta[t] for t in weights)

print(f"Weight sum: {sum(weights.values()):.4f}")
print(f"Calc Port Beta: {calc_port_beta:.4f}")
print(f"YF Port Beta: {yf_port_beta:.4f}")
