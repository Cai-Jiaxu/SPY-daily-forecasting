import pandas as pd
import pytz
from fredapi import Fred
from datetime import datetime

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

indicator_data_df = pd.concat(indicator_data_dict.values(), axis=1, keys=indicator_data_dict.keys())
indicator_data_df.ffill(inplace=True)

indicator_data_df.index.name = 'time'  


start_date = datetime(2003, 1, 1).replace(tzinfo=utc_tz)
end_date = datetime(2023, 1, 1).replace(tzinfo=utc_tz)
indicator_data_df = indicator_data_df.loc[start_date:end_date]

indicator_data_df.to_csv('fred_data.csv')



