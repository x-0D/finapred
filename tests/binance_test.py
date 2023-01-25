import unittest
import pandas as pd
from app import get_binance_data

class TestBinanceData(unittest.TestCase):

    def test_get_binance_data(self):
        symbol = 'BTCUSDT'
        interval = '1m'
        from_time = 'Today'

        df = get_binance_data(symbol, interval, from_time)

        self.assertIsInstance(df, pd.DataFrame) #check if returned data is a DataFrame 
        self.assertEqual(df.shape[1], 12) #check if the DataFrame has 11 columns 
        self.assertEqual(df['Open time'].dtype, 'datetime64[ns]') #check if Open time column is in datetime format 

        
if __name__ == '__main__': 
    unittest.main()
