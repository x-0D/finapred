import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from binance.client import Client
from prophet import Prophet

# Create a client to connect to the Binance API 
_client = Client()

_columns = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume','Ignore']
_intervals = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h','8h','12h','1d','3d','1w','1M']
_frequencies = ['1min', '3min', '5min', '15min', '30min', '1h', '2h', '4h', '6h','8h','12h','1d','3d','1w','1M']
_symbols = [sym.get('symbol', None) for sym in _client.get_exchange_info().get('symbols', {})]

def streamlit_app():
    st.set_page_config(
        page_title="FINAPRED",
        page_icon="•",
        layout="centered",
        initial_sidebar_state="expanded",
    )
    with st.sidebar.form("take parameters"):
        # symbol = st.text_input('Symbol', 'BTCUSDT')  # symbol to get data for 
        symbol = st.multiselect('Symbol', _symbols, ['BTCUSDT'], max_selections=1)[0]
        interval = st.selectbox('Interval', _intervals) # interval of the data 
        from_time = st.date_input('From Time') # start time of the data 
        # Create a slider to control the number of periods in the dataframe 
        period_slider = st.slider('Number of prediction Periods', min_value=1, max_value=365, value=15) 
        frequency = st.select_slider('Prediction frequency', options=_frequencies)

        compute = st.form_submit_button('Get Data') # when button is clicked, get the data from binance 
    st.sidebar.image('finapred.png')
    st.sidebar.title("FinaPred Finance Prediction")

    if compute:
        df = get_binance_data(symbol, interval, str(from_time))
        
        st.subheader('{} from {}, {} step with {} in future'.format(symbol,from_time,interval,period_slider))
        # plot_candlesticks(df)
        dataframe, forecast = predict_forecast(df, period_slider, frequency)
        plot_forecast(dataframe, forecast)
        show_forecast(forecast, period_slider, dataframe, symbol)


def plot_forecast(dataframe, forecast):
    fig = go.Figure(layout=go.Layout(hovermode = 'x',margin = dict(t=20,b=50,l=60,r=10)))
    fig.add_trace(go.Scatter(x=dataframe['ds'], y=dataframe['y'], name='Actual', line=dict(color='royalblue', width=2)))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'],marker=dict(color='rgba(0,0,0,0)'),hoverinfo='none',showlegend=False))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], name='Confidence', fill='tonexty', line=dict(color='gray', width=2),hoverinfo='none',mode='none'))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast', line=dict(color='firebrick', width=2)))

    st.plotly_chart(fig)


def predict_forecast(df, period_slider, freq):
    df.set_index('Open time', inplace=True) #set index as open time 
    df['Close'] = df['Close'].astype(float) #convert close column to float type 

	#prepare dataframe for fbprophet  
    dataframe = df[['Close']].reset_index() #reset index and select only close column  
    dataframe.columns = ['ds','y'] #rename columns to ds and y  

	#pass dataframe to fbprophet for prediction  
    m = Prophet(
        changepoint_prior_scale=0.5,
        interval_width=0.95,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
    )

    m.fit(dataframe)
    future = m.make_future_dataframe(periods=period_slider, freq=freq) #predict for next 50 hours  
    forecast = m.predict(future)
    return dataframe, forecast

def get_binance_data(symbol, interval, from_time):
    hist = _client.get_historical_klines(symbol, interval, from_time)
    # Convert data into a DataFrame 
    df = pd.DataFrame(hist, columns=_columns)

    df['Open time'] = pd.to_datetime(df['Open time'], unit='ms') #convert to datetime format 
    return df

def show_forecast(forecast, p, df_train, symbol):
    currency = _client.get_avg_price(symbol=symbol).get('price', None)
    # Show and plot forecast
    st.subheader("Forecast data")

    original = df_train["y"]
    prediction = forecast["yhat"][:-p]

    # st.write(original)
    # st.write(prediction)

    only_forecast = forecast  # [len(data)-1:len(forecast)]
    only_forecast["Accuracy (%)"] = (prediction / original) * 100
    only_forecast["Accuracy (%)"].where(
        only_forecast["Accuracy (%)"] < 100,
        200 - only_forecast["Accuracy (%)"],
        inplace=True,
    )
    only_forecast["Actual Price"] = df_train["y"]
    only_forecast["Date"] = only_forecast["ds"].astype(str)
    only_forecast["Predicted Price"] = only_forecast["yhat"]
    only_forecast["Predicted Price (Lower)"] = only_forecast["yhat_lower"]
    only_forecast["Predicted Price (Upper)"] = only_forecast["yhat_upper"]

    rmpse = (
        np.sqrt(np.nanmean(np.square(((original - prediction) / original)))) * 100
    ) ** 2
    mean_acc = round(only_forecast["Accuracy (%)"].mean(), 3)
    accuracy = round(mean_acc - rmpse, 2)

    st.write("Mean Accuracy =", str(mean_acc), "%")
    st.write("RMSPE =", str(round(rmpse, 3)), "%")
    st.write("Accuracy =", accuracy, "%")

    # Tomorrow's Price metric
    label = "Future Price (confidence: " + str(accuracy) + "%)"
    prd_price = round(only_forecast["Predicted Price"].iloc[-1], 4)
    if prd_price > 99:
        prd_price = round(prd_price, 2)
    act_price = only_forecast["Actual Price"].iloc[-(p + 1)]
    value = str(prd_price) + " " + currency
    if p > 1:
        tm = "last hour"
    else:
        tm = "today"
    delta = str(round(((prd_price - act_price) / act_price) * 100, 2)) + f"% since {tm}"
    with st.spinner("Predicting price..."):
        st.metric(label=label, value=value, delta=delta)


streamlit_app()