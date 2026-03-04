import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# user input
ticker = input("Enter NSE/BSE stock or ETF: ")

# download data
stock = yf.download(ticker, period="1d", interval="5m")

# fix multi index
stock.columns = stock.columns.get_level_values(0)

# moving averages
stock['MA20'] = stock['Close'].rolling(window=20).mean()
stock['MA50'] = stock['Close'].rolling(window=50).mean()

# signals
stock['Signal'] = 0
stock.loc[stock['MA20'] > stock['MA50'], 'Signal'] = 1
stock.loc[stock['MA20'] < stock['MA50'], 'Signal'] = -1

buy = stock[(stock['Signal'].shift(1) == -1) & (stock['Signal'] == 1)]
sell = stock[(stock['Signal'].shift(1) == 1) & (stock['Signal'] == -1)]

# chart
plt.figure(figsize=(12,6))

plt.plot(stock['Close'], label='Close Price')
plt.plot(stock['MA20'], label='20 Day MA')
plt.plot(stock['MA50'], label='50 Day MA')

plt.scatter(buy.index, buy['Close'], marker='^', label='BUY', s=100)
plt.scatter(sell.index, sell['Close'], marker='v', label='SELL', s=100)

current_price = stock['Close'].iloc[-1]
last_date = stock.index[-1]

plt.axhline(current_price, linestyle='--', label=f'Current Price: {current_price:.2f}')
plt.text(last_date, current_price, f"{current_price:.2f}")

plt.title(f"{ticker} Stock Analysis")
plt.xlabel("Date")
plt.ylabel("Price")

plt.legend()
plt.show()