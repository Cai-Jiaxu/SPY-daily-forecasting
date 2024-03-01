
import pandas as pd
import pandas_ta as ta
import pytz
import yfinance as yf
#from datetime import datetime


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
ta_data.to_csv('ta_data.csv')


 


