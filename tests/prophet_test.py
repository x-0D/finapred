import unittest
import pandas as pd
from app import predict_forecast

class TestPredictForecast(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({'Open time': ['2020-01-01 00:00:00', '2020-01-02 00:00:00', '2020-01-03 00:00:00'], 
                                'Close': [100, 200, 300]})

    def test_predict_forecast(self):
        period_slider = 50 
        freq = 'H' 

        dataframe, forecast = predict_forecast(self.df, period_slider, freq)

        self.assertEqual(dataframe['ds'].tolist(), ['2020-01-01 00:00:00', '2020-01-02 00:00:00', '2020-01-03 00:00:00'])  # check if ds column is correct 
        self.assertEqual(dataframe['y'].tolist(), [100.0, 200.0, 300.0]) # check if y column is correct 

        self.assertEqual(len(forecast), 53) # check if forecast dataframe has 53 rows (3 original + 50 predicted) 
        self.assertEqual(forecast['ds'].tolist()[3], pd.Timestamp('2020-01-03 01:00:00')) # check if first predicted row is correct  
        
if __name__ == "__main__": 
    unittest.main()
