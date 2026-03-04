import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import numpy as np
from sklearn.linear_model import LinearRegression
from streamlit_autorefresh import st_autorefresh

# ======================
# SUPPORT / RESISTANCE
# ======================

def detect_support_resistance(df):

    supports = []
    resistances = []

    for i in range(2, len(df)-2):

        if df["Low"].iloc[i] < df["Low"].iloc[i-1] and df["Low"].iloc[i] < df["Low"].iloc[i+1]:
            supports.append(df["Low"].iloc[i])

        if df["High"].iloc[i] > df["High"].iloc[i-1] and df["High"].iloc[i] > df["High"].iloc[i+1]:
            resistances.append(df["High"].iloc[i])

    return supports, resistances


# ======================
# TRENDLINE
# ======================

def detect_trendline(df):

    x = np.arange(len(df))
    y = df["Close"].values

    coeff = np.polyfit(x, y, 1)

    slope = coeff[0]
    intercept = coeff[1]

    trendline = slope * x + intercept

    return trendline


# ======================
# BREAKOUT SIGNAL
# ======================

def breakout_signal(price, supports, resistances):

    signal = "NEUTRAL"

    if len(resistances) > 0:
        if price > max(resistances[-3:]):
            signal = "BREAKOUT BUY 🚀"

    if len(supports) > 0:
        if price < min(supports[-3:]):
            signal = "BREAKDOWN SELL ⚠️"

    return signal


# ======================
# PAGE CONFIG
# ======================

st.set_page_config(page_title="AI Trading Dashboard", layout="wide")

st_autorefresh(interval=5000, key="refresh")

st.title("📊 AI Multi-Market Trading Analytics Dashboard")


# ======================
# MARKET SELECTOR
# ======================

market = st.selectbox(
    "Select Market",
    ["Stocks / ETF", "Forex", "Crypto"]
)

if market == "Stocks / ETF":
    ticker = st.text_input("Example: RELIANCE.NS")

elif market == "Forex":
    ticker = st.text_input("Example: EURUSD=X")

else:
    ticker = st.text_input("Example: BTC-USD")


# ======================
# TIMEFRAME
# ======================

timeframe = st.selectbox(
    "Timeframe",
    ["1m","5m","15m","1h","4h","1d","1wk"]
)


# ======================
# DATA FETCH
# ======================

if ticker:

    if timeframe == "1m":
        stock = yf.download(ticker, period="1d", interval="1m")

    elif timeframe == "5m":
        stock = yf.download(ticker, period="5d", interval="5m")

    elif timeframe == "15m":
        stock = yf.download(ticker, period="5d", interval="15m")

    elif timeframe == "1h":
        stock = yf.download(ticker, period="1mo", interval="60m")

    elif timeframe == "4h":
        stock = yf.download(ticker, period="3mo", interval="60m")

    elif timeframe == "1d":
        stock = yf.download(ticker, period="1y", interval="1d")

    else:
        stock = yf.download(ticker, period="5y", interval="1wk")

    if not stock.empty:

        stock.columns = stock.columns.get_level_values(0)

        stock["MA20"] = stock["Close"].rolling(20).mean()
        stock["MA50"] = stock["Close"].rolling(50).mean()

        delta = stock["Close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()

        rs = avg_gain / avg_loss
        stock["RSI"] = 100 - (100/(1+rs))

        stock["Signal"] = 0
        stock.loc[stock["MA20"] > stock["MA50"], "Signal"] = 1
        stock.loc[stock["MA20"] < stock["MA50"], "Signal"] = -1

        buy = stock[(stock["Signal"].shift(1) == -1) & (stock["Signal"] == 1)]
        sell = stock[(stock["Signal"].shift(1) == 1) & (stock["Signal"] == -1)]

        current_price = stock["Close"].iloc[-1]

        supports, resistances = detect_support_resistance(stock)
        trendline = detect_trendline(stock)

        breakout = breakout_signal(current_price, supports, resistances)

        # ======================
        # CHART
        # ======================

        fig = go.Figure()

        fig.add_trace(go.Candlestick(
            x=stock.index,
            open=stock["Open"],
            high=stock["High"],
            low=stock["Low"],
            close=stock["Close"],
            name="Price"
        ))

        fig.add_trace(go.Scatter(
            x=stock.index,
            y=stock["MA20"],
            name="MA20"
        ))

        fig.add_trace(go.Scatter(
            x=stock.index,
            y=stock["MA50"],
            name="MA50"
        ))

        fig.add_trace(go.Scatter(
            x=stock.index,
            y=trendline,
            name="Trendline"
        ))

        for s in supports[-3:]:
            fig.add_hline(y=s, line_color="green", line_dash="dot")

        for r in resistances[-3:]:
            fig.add_hline(y=r, line_color="red", line_dash="dot")

        fig.add_trace(go.Scatter(
            x=buy.index,
            y=buy["Close"],
            mode="markers",
            name="BUY"
        ))

        fig.add_trace(go.Scatter(
            x=sell.index,
            y=sell["Close"],
            mode="markers",
            name="SELL"
        ))

        fig.update_layout(height=600, xaxis_rangeslider_visible=False)

        st.plotly_chart(fig, use_container_width=True)

        # ======================
        # BREAKOUT ALERT
        # ======================

        st.subheader("🚨 Breakout Alert")
        st.write(breakout)

        # ======================
        # AI PREDICTION
        # ======================

        st.subheader("🤖 AI Prediction")

        close_prices = stock["Close"].values

        X = np.array(range(len(close_prices))).reshape(-1,1)
        y = close_prices

        model = LinearRegression()
        model.fit(X,y)

        next_step = np.array([[len(close_prices)]])
        prediction = model.predict(next_step)[0]

        st.metric("Predicted Next Price", round(prediction,2))

        # ======================
        # AI TRADE SIGNAL
        # ======================

        ai_trade = "BUY 📈" if prediction > current_price else "SELL 📉"

        st.subheader("🤖 AI Trade Signal")
        st.metric("AI Recommendation", ai_trade)

    else:

        st.error("Invalid symbol or data unavailable")