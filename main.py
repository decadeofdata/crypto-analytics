# pip install streamlit prophet yfinance plotly
import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet #not fbprophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Crypto Analytics')

symbol = ('BTC-USD', 'ETH-USD', 'BNB-USD', 'MATIC-USD', 'SOL-USD', 'ADA-USD', 'DOT-USD', 'TRX-USD', 'DOGE-USD', 'LTC-USD')
selected_symbol = st.selectbox('Select crypto', symbol)

@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

	
data_load_state = st.text('Loading data...')
data = load_data(selected_symbol)
data_load_state.text('Updated.')

# Plot raw data

# Gets latest price
closePrices = data['Close']
lastPrice = closePrices.iloc[-1]
lastPrice = round(lastPrice, 2)
plotTitle = "Crypto price $" + str(lastPrice)

def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x = data['Date'], y = data['Open'], name = "stock_open"))
	fig.add_trace(go.Scatter(x = data['Date'], y = data['Close'], name = "stock_close"))
	fig.layout.update(title_text = plotTitle, xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()

st.subheader('Volume traded')

st.line_chart(data.Volume)

st.subheader('Raw data')
st.write(data.tail())

# Predict forecast with Prophet.

st.subheader('Forecast demonstration')
st.text("This is not financial advice. This is a demonstration for the forecast feature \n we are working on. You will lose money if you follow these predictions.")

# Slider

n_years = st.slider('Years of prediction:', 1, 4)
period = n_years * 365

df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show and plot forecast
st.subheader('Forecast data')

st.write(forecast.tail())
    
st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)