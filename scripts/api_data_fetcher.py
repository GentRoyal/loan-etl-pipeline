import pandas as pd
import os
import requests
from io import StringIO
from config import settings

class MyAPI:
    def __init__(self):
        self.api_key = settings.api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.path = '../data/raw'
        if not os.path.exists(self.path):
                os.makedirs(self.path)
        
    def get_stock(self, symbol, size="compact"):
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "outputsize": size,
            "apikey": self.api_key,
            "datatype": "csv"
        }

        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()  # raises an error for bad responses
            df = pd.read_csv(StringIO(response.text))
            df.set_index('timestamp', inplace=True)
            df.sort_index(ascending = False)

            df.to_csv(f'{self.path}/{symbol}.csv')

            return df
            
        except Exception as e:
            print(f"Error fetching stock data: {e}")
            
            return pd.DataFrame()  # return empty DataFrame on failure

    def get_crypto(self, symbol, market, size="compact"):
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "market": market,
            "outputsize": size,
            "apikey": self.api_key,
            "datatype": "csv"
        }

        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()  # raises an error for bad responses
            df = pd.read_csv(StringIO(response.text))
            df.set_index('timestamp', inplace=True)
            df.sort_index(ascending = False)
            
            df.to_csv(f'{self.path}/{symbol}_{market}.csv')
            
            return df
            
        except Exception as e:
            print(f"Error fetching crypto data: {e}")
            
            return pd.DataFrame()  # return empty DataFrame on failure
