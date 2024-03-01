
import numpy as np
import pandas as pd
import pytz
import yfinance as yf

tickers = {
    '^GSPC': 'SPY',
    '^VIX': 'VIX',
    '^DJI': 'DJI',
    '^IXIC': 'Nasdaq',
    '^RUT': 'RUSSELL 2000',
    '^GDAXI': 'DAX',
    '^FTSE': 'FTSE',
    '^FCHI': 'CAC',
    '^HSI': 'HSI',
    '000001.SS': 'SSE',
    '^N225': 'Nikkei',
    'XLE': 'ENERGY',
    'XLF': 'FINANCIALS',
    'XLK': 'TECHNOLOGY',
    'XLV': 'HEALTHCARE',
    'XLP': 'CONSUMER STAPLES',
    'XLI': 'INDUSTRIALS',
    'XLU': 'UTLITIES',
    'XLY': 'CONSUMER DISCRETIONARY',
    'XLB': 'MATERIALS',
    '^FVX': '5Y YIELD',
    '^TNX': '10Y YIELD',
    '^TYX': '30Y YIELD',
}

ticker_data_df = pd.DataFrame()

utc_tz = pytz.timezone('Etc/UTC')

for symbol, name in tickers.items():
    ticker_data = yf.download(symbol, interval='1d', start="2003-01-01", end="2023-01-07")['Close']
    ticker_data.index = ticker_data.index.tz_localize('UTC').tz_convert(utc_tz)
    # Calculate daily return from close
    daily_returns_df = ticker_data.pct_change().apply(lambda x: np.log(1 + x)) * 100
    
    # Forward fill NaN values
    filled_daily_returns_df = daily_returns_df.ffill()
    # Drop remaining NaN values
    filtered_daily_returns_df = filled_daily_returns_df.dropna()
    
    # Assign daily returns DataFrame to the DataFrame with ticker names
    ticker_data_df[name] = filtered_daily_returns_df

# Save DataFrame to CSV
ticker_data_df.to_csv('ticker_data.csv')



