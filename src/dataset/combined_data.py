import pandas as pd
import numpy as np

fred_data_file_path = 'C:\\Users\\feget\\SPY-daily-forecasting\\data\\fred_data.csv'
fx_data_file_path = 'C:\\Users\\feget\\SPY-daily-forecasting\\data\\fx_data.csv'
spy_feat_file_path = 'C:\\Users\\feget\\SPY-daily-forecasting\\data\\spy_feat.csv'


fred_data_df = pd.read_csv(fred_data_file_path)
fred_data_df = fred_data_df.rename(columns={'time': 'Date'})
fred_data_df['Date'] = pd.to_datetime(fred_data_df['Date']).dt.tz_convert(None)
fx_data_df = pd.read_csv(fx_data_file_path)
fx_data_df = fx_data_df.rename(columns={'time': 'Date'})
fx_data_df['Date'] = pd.to_datetime(fx_data_df['Date']).dt.tz_convert(None)
spy_feat_df = pd.read_csv(spy_feat_file_path)
spy_feat_df['Date'] = pd.to_datetime(spy_feat_df['Date']).dt.tz_convert(None)



# Generate a DataFrame with all the unique dates
start_date = '2003-01-01'
end_date = '2023-01-06'
date_range = pd.date_range(start=start_date, end=end_date, freq='D')
date_df = pd.DataFrame(date_range, columns=['Date'])

# Merge each DataFrame with the date DataFrame using left join
fred_data_df = pd.merge(date_df, fred_data_df, on='Date', how='left')
fx_data_df = pd.merge(date_df, fx_data_df, on='Date', how='left')
spy_feat_df = pd.merge(date_df, spy_feat_df, on='Date', how='left')


combined_data_df = pd.concat([spy_feat_df, fred_data_df, fx_data_df], axis=1)
combined_data_df = combined_data_df.loc[:, ~combined_data_df.columns.duplicated()]
combined_data_df.replace([np.inf, -np.inf], np.nan, inplace=True)
combined_data_df.dropna(inplace=True)
combined_data_df = combined_data_df.drop(columns=[combined_data_df.columns[1]])
combined_data_df = combined_data_df.drop(columns=['Close', 'Adj Close'])
combined_data_df.to_csv('combined_data.csv')



