import datetime as dt
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from pytz import timezone  # Import the timezone module from pytz


def calculate_implied_volatility(data):
    # Calculate implied volatility for calls and puts
    data['impliedVolatility'] = pd.to_numeric(data['impliedVolatility'], errors='coerce')
    return data


def process_expiration(exp_td_str, tk):
    options = tk.option_chain(exp_td_str)
    calls = options.calls
    puts = options.puts

    # Calculate implied volatility
    calls = calculate_implied_volatility(calls)
    puts = calculate_implied_volatility(puts)

    return calls, puts


def process_symbol(symbol):
    tk = yf.Ticker(symbol)
    expirations = tk.options

    data_calls = pd.DataFrame()
    data_puts = pd.DataFrame()

    for exp_td_str in expirations[:1]:
        calls, puts = process_expiration(exp_td_str, tk)
        data_calls = pd.concat([data_calls, calls], ignore_index=True)
        data_puts = pd.concat([data_puts, puts], ignore_index=True)

    return data_calls, data_puts


SYMBOLS = ['AAPL', 'MSFT', 'XOM', 'BAC']
for symbol in SYMBOLS:
    calls, puts = process_symbol(symbol)

    # Plot implied volatilities
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Strike Price')
    ax1.set_ylabel('Implied Volatility', color=color)
    ax1.plot(calls['strike'], calls['impliedVolatility'], label='Calls IV', color=color)
    ax1.plot(puts['strike'], puts['impliedVolatility'], label='Puts IV', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:orange'
    ax2.set_ylabel('Days Left to Expire', color=color)  # we already handled the x-label with ax1
    expiration_dates = pd.to_datetime(calls['lastTradeDate'])
    expiration_dates_puts = pd.to_datetime(puts['lastTradeDate'])

    # Ensure both timestamps have the same timezone (e.g., UTC)
    current_time = dt.datetime.now(timezone('UTC'))
    days_to_expire = (expiration_dates - current_time).dt.days
    days_to_expire_puts = (expiration_dates_puts - current_time).dt.days

    ax2.plot(puts['strike'], days_to_expire_puts, label='Days Left', color='tab:cyan')
    ax2.plot(calls['strike'], days_to_expire, label='Days Left', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.invert_yaxis()  # Invert y-axis for days left
    ax2.legend(loc='upper right')

    # Add vertical dotted black line for current price
    current_price = yf.Ticker(symbol).history(period='1d')['Close'][0]
    ax1.axvline(x=current_price, color='black', linestyle='--', label='Current Price')
    ax1.legend(loc='upper left')

    # Add horizontal lines for bid and ask prices
    for idx, row in calls.iterrows():
        ax1.hlines(y=row['impliedVolatility'], xmin=row['strike']-(row['bid'])/10, xmax=row['strike']+(row['ask'])/10, colors='green', linestyles='solid',
                   alpha=0.5)
        # ax1.hlines(y=row['ask'], xmin=row['strike'], xmax=row['strike'] + 0.5, colors='red', linestyles='solid',
        #            alpha=0.5)

    for idx, row in puts.iterrows():
        ax1.hlines(y=row['impliedVolatility'], xmin=row['strike']-(row['bid'])/10, xmax=row['strike']+(row['ask'])/10, colors='blue', linestyles='solid',
                   alpha=0.5)
        # ax1.hlines(y=row['ask'], xmin=row['strike'], xmax=row['strike'] + 0.5, colors='purple', linestyles='solid',
        #            alpha=0.5)

    plt.title(f'Implied Volatility and Days Left for {symbol}')
    plt.grid(True)
plt.show()

