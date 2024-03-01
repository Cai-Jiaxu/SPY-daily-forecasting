import MetaTrader5 as mt5
from datetime import datetime
import pytz
import pandas as pd


# Define forex symbols
forex_symbols = ['EURUSD', 'USDJPY', 'AUDUSD', 'USDCAD']
forex_dataframes = []

# Set dates to UTC timezone
timezone = pytz.timezone("Etc/UTC")
start_date = datetime(2003, 1, 1)
end_date = datetime(2023, 1, 2)
utc_from = timezone.localize(datetime(start_date.year, start_date.month, start_date.day))
utc_to = timezone.localize(datetime(end_date.year, end_date.month, end_date.day, hour=23, minute=59, second=59))

# Fetch data for each symbol
for symbol in forex_symbols:
    mt5.initialize()
    forex_data = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_D1, utc_from, utc_to)
    mt5.shutdown()

    # Convert data to DataFrame
    forex_df = pd.DataFrame(forex_data, columns=['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume'])
    forex_df['time'] = pd.to_datetime(forex_df['time'], unit='s')
    forex_df['time'] = forex_df['time'].dt.tz_localize(timezone)  # Localize the time column
    forex_df.set_index('time', inplace=True)
    
    symbol_close = forex_df['close']
    symbol_returns = symbol_close.pct_change() * 100
    symbol_returns = symbol_returns.rename(f'{symbol}')
    symbol_returns = symbol_returns.ffill().dropna()
    
    forex_dataframes.append(symbol_returns)


fx_data = pd.concat(forex_dataframes, axis=1)
fx_data.to_csv("fx_data.csv")



