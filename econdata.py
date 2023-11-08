from datetime import datetime
from fredapi import Fred
import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import pytz
from statsmodels.tsa.stattools import adfuller
import yfinance as yf

# Connect to yfinance and collect data
def yf_data():

    tickers = {
    '^GSPC': 'SPY',
    '^VIX': 'VIX',
    '^DJI': 'DJI',
    '^IXIC': 'Nasdaq',
    '^GDAXI': 'DAX',
    '^FTSE': 'FTSE',
    '^FCHI': 'CAC',
    '^HSI': 'HSI',
    '000001.SS': 'SSE',
    '^N225': 'Nikkei',
}
    daily_returns_df = pd.DataFrame()

    utc_tz = pytz.timezone('Etc/UTC')

    for symbol, name in tickers.items():
        ticker_data = yf.download(symbol, interval='1d', start="2003-01-01", end="2023-01-07")['Close']
        ticker_data.index = ticker_data.index.tz_localize('UTC').tz_convert(utc_tz)
        # Calculate daily return from close
        daily_returns = ticker_data.pct_change().apply(lambda x: np.log(1 + x)) * 100
        daily_returns_df[name] = daily_returns

        # Convert index to UTC timezone
        

        # Forward fill NaN values
        filled_daily_returns_df = daily_returns_df.ffill()
        # Drop remaining NaN values
        filtered_daily_returns_df = filled_daily_returns_df.dropna()
        

    return filtered_daily_returns_df
    
# Connect to FRED and collect economic data
def fed_data():

    indicators = {
        'CPIAUCNS': 'CPI',
        'POILWTIUSDM': 'WTI',
        'DGS1MO': 'TB1M',
        'TB3MS': 'TB3M',
        'GS2': 'TB2Y',
        'GS10': 'TB10Y',
        'PSAVERT': 'PERSONAL_SAVINGS',
        'WM1NS': 'M1',
        'WM2NS': 'M2',
        'WALCL': 'FED_BALANCE_SHEET',
    }
    utc_tz = pytz.timezone('Etc/UTC')

    fred = Fred(api_key='7cdaa2609ea03808ab1cd13fcb0ad384')
    indicator_data_dict = {}

    for tag, indicator_name in indicators.items():
        data = fred.get_series_latest_release(tag)
        data.index = pd.to_datetime(data.index).tz_localize('UTC').tz_convert(utc_tz)
        indicator_data_dict[indicator_name] = data

    # Concatenate data and forward fill NaN values
    indicator_data_df = pd.concat(indicator_data_dict.values(), axis=1, keys=indicator_data_dict.keys())
    indicator_data_df.ffill(inplace=True)

    # Filter data for the desired date range
    start_date = datetime(2003, 1, 1).replace(tzinfo=utc_tz)
    end_date = datetime(2023, 1, 1).replace(tzinfo=utc_tz)
    indicator_data_df = indicator_data_df.loc[start_date:end_date]
    return indicator_data_df

# connect to mt5 and collect forex data
def collect_forex_data():
    forex_symbols = ['EURUSD', 'USDJPY', 'AUDUSD', 'USDCAD']
    forex_dataframes = []
    # set dates to utc
    timezone = pytz.timezone("Etc/UTC")
    start_date = datetime(2003, 1, 1)
    end_date = datetime(2023, 1, 2)
    utc_from = datetime(start_date.year, start_date.month, start_date.day, tzinfo=timezone)
    utc_to = datetime(end_date.year, end_date.month, end_date.day, hour=23, minute=59, second=59, tzinfo=timezone)
    
    for symbol in forex_symbols:
        mt5.initialize()
        forex_data = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_D1, utc_from, utc_to)
        forex_df = pd.DataFrame(forex_data, columns=['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume'])
        forex_df['time'] = pd.to_datetime(forex_df['time'], unit='s')
        forex_df.set_index('time', inplace=True)
        symbol_close = forex_df['close']
        symbol_returns = symbol_close.pct_change() * 100
        symbol_returns = symbol_returns.rename(f'{symbol}')
        # ffill Nan values
        symbol_returns = symbol_returns.ffill()
        # drop remaining Nan
        symbol_returns = symbol_returns.dropna()
        forex_dataframes.append(symbol_returns)
        mt5.shutdown()

    return pd.concat(forex_dataframes, axis=1)

# adf test for market data
def adf_test(dataframe):
    results = {}
    for column in dataframe.columns:
        adf_result = adfuller(dataframe[column])
        results[column] = {
            'Test Statistic': adf_result[0],
            'P-Value': adf_result[1],
            'Lags Used': adf_result[2],
            'Number of Observations Used': adf_result[3],
            'Critical Values': adf_result[4]
        }
    
    adf_results_df = pd.DataFrame(results).T
    return adf_results_df

filtered_daily_return_df = yf_data()
indicator_data_df = fed_data()
forex_data = collect_forex_data()
adf_results = adf_test(filtered_daily_return_df)
combined_data = pd.concat([filtered_daily_return_df, indicator_data_df, forex_data], axis=1)

print(combined_data)
