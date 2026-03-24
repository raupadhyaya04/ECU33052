import pandas_datareader.data as web
import datetime
import pandas as pd
df = web.DataReader('F-F_Research_Data_Factors_daily', 'famafrench', start='2020-01-01')[0]
print(df.head())
