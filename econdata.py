from datetime import datetime
from fredapi import Fred
import matplotlib.pyplot as plt
import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import pandas_ta as ta
import pytz
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import yfinance as yf

# Connect to yfinance and collect data
def yf_data():

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
    daily_returns_df = pd.DataFrame()

    utc_tz = pytz.timezone('Etc/UTC')

    for symbol, name in tickers.items():
        ticker_data = yf.download(symbol, interval='1d', start="2003-01-01", end="2023-01-07")['Close']
        ticker_data.index = ticker_data.index.tz_localize('UTC').tz_convert(utc_tz)
        # Calculate daily return from close
        daily_returns = ticker_data.pct_change().apply(lambda x: np.log(1 + x)) * 100
        daily_returns_df[name] = daily_returns
        
        # Forward fill NaN values
        filled_daily_returns_df = daily_returns_df.ffill()
        # Drop remaining NaN values
        filtered_daily_returns_df = filled_daily_returns_df.dropna()
        

    return filtered_daily_returns_df

# collect spy volume and other TA
def spy_ta():
    tickers = ['SPY']
    start_date = '2003-01-01'
    end_date = '2023-01-07'

    # Fetch historical data from Yahoo Finance
    data = yf.download(tickers, start=start_date, end=end_date)
    volume = yf.download(tickers, start=start_date, end=end_date)['Volume']

    # Convert index to UTC timezone
    utc_tz = pytz.timezone('Etc/UTC')
    data.index = data.index.tz_localize('UTC').tz_convert(utc_tz)
    volume.index = volume.index.tz_localize('UTC').tz_convert(utc_tz)
    data = pd.DataFrame(data)
    volume = pd.DataFrame(volume)

    # simple moving average for 5,10,20 days
    data.ta.sma(length=5, append=True)
    data.ta.sma(length=10, append=True)
    data.ta.sma(length=20, append=True)
    # exponential moving average for 5,10,20 days
    data.ta.ema(length=5, append=True)
    data.ta.ema(length=10, append=True)
    data.ta.ema(length=20, append=True)
    # rsi for 7,14,20 days
    data.ta.rsi(length=7, append=True)
    data.ta.rsi(length=14, append=True)
    data.ta.rsi(length=20, append=True)
    # atr for 14 days
    data.ta.atr(length=14, append=True)
    data.ta.obv(high='high', low='low', close='close', volume='volume', append=True)
    

    ta_data = pd.concat([data, volume], axis=1)

    return ta_data

    
# Connect to FRED and collect economic data
def fed_data():

    indicators = {
        'CPIAUCSL': 'CPI',
        'MEDCPIM158SFRBCLE': 'MEDIAN CPI',
        'STICKCPIM157SFRBATL': 'STICKY CPI',
        'PCE': 'PCE',
        'PSAVERT': 'PERSONAL SAVINGS',
        'PPIACO': 'PPI (ALL COMMODITIES)',
        'T10YIE': '10Y EXPECTED INFLATION',
        'T5YIE' : '5Y EXPECTED INFLATION',
        'EXPINF1YR': '1Y EXPECTED INFLATION',
        'EXPINF2YR': '2Y EXPECTED INFLATION',
        'DCOILBRENTEU': 'BRENT',
        'DCOILWTICO': 'WTI',
        'T10Y2Y': '10Y-2Y: YIELD CURVE',
        'DGS1MO': '1M TBILL',
        'DTB3': '3M TBILL',
        'RPTTLD': 'FED OPEN MARKET OPERATIONS: TOTAL SECURITIES',
        'M1SL': 'M1',
        'WM2NS': 'M2',
        'BAMLC0A1CAAAEY': 'AAA YIELD',
        'USALORSGPNOSTSAM': 'US GDP',
        'GEPUCURRENT': 'POLICY UNCERTAINTY INDEX',
        'TLRESCONS': 'RESIDENTIAL CONSTRUCTION SPENDING',
        'HOUST': 'NEW PRIVATE HOUSING',
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
    
    timezone = pytz.timezone("Etc/UTC")
    start_date = datetime(2003, 1, 1)
    end_date = datetime(2023, 1, 2)
    utc_from = timezone.localize(datetime(start_date.year, start_date.month, start_date.day))
    utc_to = timezone.localize(datetime(end_date.year, end_date.month, end_date.day, hour=23, minute=59, second=59))
    
    for symbol in forex_symbols:
        mt5.initialize()
        forex_data = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_D1, utc_from, utc_to)
        forex_df = pd.DataFrame(forex_data, columns=['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume'])
        forex_df['time'] = pd.to_datetime(forex_df['time'], unit='s')
        forex_df['time'] = forex_df['time'].dt.tz_localize(timezone)  # Localize the time column
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

# correlation analysis
def correlation_test(dataframe):
    correlation_matrix = dataframe.corr()
    sns.heatmap(correlation_matrix, cmap='coolwarm')
    plt.title("Correlation Matrix")
    plt.show()

# stationarize data
def station_data(dataframe):
    for column in dataframe.columns:
        # Perform seasonal adjustment
        decomposition = seasonal_decompose(indicator_data_df[column], model='additive', period=1)
        dataframe[column + '_seasonally_adjusted'] = decomposition.trend

        # Differencing to make the series stationary
        dataframe[column + '_stationary'] = dataframe[column + '_seasonally_adjusted'].diff()

    # Drop NaN values resulted from differencing
    dataframe = dataframe.dropna()

    return dataframe

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

# principal component analysis if needed
def apply_pca(dataframe):
    features = dataframe.drop(columns='SPY')
    target = dataframe['SPY']
    # apply standardization 
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    pca = PCA(n_components=0.95)
    features_pca = pca.fit_transform(scaled_features)

    # Check the explained variance ratio for each component
    explained_variance_ratio = pca.explained_variance_ratio_
    print("Explained Variance Ratio:", explained_variance_ratio)

    # Check the cumulative explained variance
    cumulative_explained_variance = explained_variance_ratio.cumsum()
    print("Cumulative Explained Variance:", cumulative_explained_variance)

    # plot for explained variance
    plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o')
    plt.title('Screen Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.show()

    # plot for cumulative variance
    plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o')
    plt.title('Cumulative Explained Variance Plot')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.show()

    return features_pca



#filtered_daily_return_df = yf_data()
indicator_data_df = fed_data()
indicator_data_df = indicator_data_df.dropna()
indicator_data_df = station_data(indicator_data_df)
#forex_data = collect_forex_data()
#ta_data = spy_ta()
adf_results = adf_test(indicator_data_df)
print(adf_results)
#combined_data = pd.concat([filtered_daily_return_df, indicator_data_df, forex_data, ta_data], axis=1)
#clean_combined_data = combined_data.dropna()
#data_pca = apply_pca(clean_combined_data)
#heatmap = correlation_test(clean_combined_data)


