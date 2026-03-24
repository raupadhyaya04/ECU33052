import tempfile
import urllib.request
import zipfile
import pandas as pd
import io

def get_ff_daily():
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        with zipfile.ZipFile(io.BytesIO(response.read())) as z:
            csv_name = z.namelist()[0]
            with z.open(csv_name) as f:
                df = pd.read_csv(f, skiprows=4)
                # Keep first 5 cols to fix mismatch if last rows have empty Strings
                df = df.iloc[:, :5]
                df.columns = ["Date", "Mkt-RF", "SMB", "HML", "RF"]
                df = df.dropna()
                df["Date"] = pd.to_numeric(df["Date"], errors='coerce')
                df = df.dropna(subset=["Date"])
                df["Date"] = pd.to_datetime(df["Date"].astype(int).astype(str), format="%Y%m%d")
                df.set_index("Date", inplace=True)
                return df

print(get_ff_daily().tail())
