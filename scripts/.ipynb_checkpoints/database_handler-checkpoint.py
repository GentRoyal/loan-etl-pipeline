import sqlite3
import pandas as pd
from config import settings

class MySQL:
    def __init__(self):
        self.db_name = settings.db_name
        try:
            self.conn = sqlite3.connect(f"../data/{self.db_name}")
        except sqlite3.Error as e:
            print(f"Error connecting to database: {e}")
            self.conn = None

    def to_table(self, df, symbol, if_exists = 'replace'):
        if self.conn is None:
            print("No database connection.")
            return

        try:
            df.to_sql(name = symbol, con = self.conn, if_exists = if_exists, index = False)
            print(f"Data inserted into table '{symbol}' successfully.")
        except Exception as e:
            print(f"Error inserting data into table '{symbol}': {e}")

    def from_table(self, symbol):
        if self.conn is None:
            print("No database connection.")
            return pd.DataFrame()

        try:
            query = f"SELECT * FROM {symbol}"
            df = pd.read_sql(query, self.conn)
            print(f"Data retrieved from table '{symbol}' successfully.")
            return df
        except Exception as e:
            print(f"Error reading from table '{symbol}': {e}")
            return pd.DataFrame()
