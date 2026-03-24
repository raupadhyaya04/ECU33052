import io
import urllib.request
import zipfile
import pandas as pd
import numpy as np

_FF_CACHE = None

def get_fama_french_daily() -> pd.DataFrame:
    global _FF_CACHE
    if _FF_CACHE is not None:
        return _FF_CACHE
    
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    
    try:
        with urllib.request.urlopen(req) as response:
            with zipfile.ZipFile(io.BytesIO(response.read())) as z:
                csv_name = z.namelist()[0]
                with z.open(csv_name) as f:
                    # Skip the first 4 rows which are Kenneth French headers
                    df = pd.read_csv(f, skiprows=4)
                    
                    # Ensure we only grab the first 5 columns to avoid garbage at the end
                    df = df.iloc[:, :5]
                    df.columns = ["Date", "Mkt-RF", "SMB", "HML", "RF"]
                    df = df.dropna()
                    
                    # Parse dates
                    df["Date"] = pd.to_numeric(df["Date"], errors='coerce')
                    df = df.dropna(subset=["Date"])
                    df["Date"] = pd.to_datetime(df["Date"].astype(int).astype(str), format="%Y%m%d")
                    df.set_index("Date", inplace=True)
                    
                    # Convert percentages to decimals
                    df = df / 100.0
                    _FF_CACHE = df
                    return df
    except Exception as e:
        print(f"Warning: Could not fetch Fama-French data: {e}")
        return pd.DataFrame()

def calculate_capm_ff3_betas(returns: pd.DataFrame, use_ff3: bool = False) -> np.ndarray:
    """
    Computes CAPM or FF3 market betas for each asset.
    If use_ff3=True, regresses against Mkt-RF, SMB, and HML.
    If use_ff3=False (CAPM), regresses only against Mkt-RF.
    """
    ff_data = get_fama_french_daily()
    
    if ff_data.empty:
        # Fallback to equal weighted proxy if FF download fails
        mkt_returns = returns.mean(axis=1)
        var_m = mkt_returns.var()
        if var_m < 1e-12:
            return np.ones(returns.shape[1])
        cov_sm = returns.apply(lambda x: x.cov(mkt_returns))
        return (cov_sm / var_m).values
        
    # Align dates
    common_idx = returns.index.intersection(ff_data.index)
    if len(common_idx) < 30:
        # Fallback if too few overlapping days
        mkt_returns = returns.mean(axis=1)
        var_m = mkt_returns.var()
        if var_m < 1e-12:
            return np.ones(returns.shape[1])
        cov_sm = returns.apply(lambda x: x.cov(mkt_returns))
        return (cov_sm / var_m).values
        
    ret_aligned = returns.loc[common_idx]
    ff_aligned = ff_data.loc[common_idx]
    
    rf = ff_aligned["RF"].values[:, np.newaxis]
    mkt_rf = ff_aligned["Mkt-RF"].values
    
    # Asset excess returns
    excess_ret = ret_aligned.values - rf
    
    betas = []
    
    if use_ff3:
        # X: [Intercept, Mkt-RF, SMB, HML]
        X = np.column_stack([
            np.ones(len(common_idx)),
            mkt_rf,
            ff_aligned["SMB"].values,
            ff_aligned["HML"].values
        ])
    else:
        # X: [Intercept, Mkt-RF]
        X = np.column_stack([np.ones(len(common_idx)), mkt_rf])
        
    # Solve regression using pseudo-inverse
    # coefficients = (X^T X)^-1 X^T Y
    try:
        coefs, _, _, _ = np.linalg.lstsq(X, excess_ret, rcond=None)
        # Market Beta is always the second coefficient index (index 1)
        market_betas = coefs[1, :]
        return market_betas
    except Exception:
        return np.ones(returns.shape[1])
