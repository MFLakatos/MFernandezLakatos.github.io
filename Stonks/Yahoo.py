import datetime as dt
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
# https://www.fintut.com/yahoo-finance-options-python/

def calculate_implied_volatility(data):
    # Calculate implied volatility for calls and puts
    data['impliedVolatility'] = pd.to_numeric(data['impliedVolatility'], errors='coerce')
    return data

def process_expiration(exp_td_str,tk):
    """
    Download Yahoo Finance call and put option quotes
    for a single expiration
    Input:
    exp_td_str = expiration date string "%Y-$m-$d"
        (a single item from yfinance.Ticker.options tuple)
    Return pandas.DataFrame with merged calls and puts data
    """

    options = tk.option_chain(exp_td_str) # object of the yfinance.Options class. Two attributes: calls and puts
    # print(options)
    calls = options.calls   #pandas DataFrames
    puts = options.puts     #pandas DataFrames

    # Option chain DataFrame columns
    # contractSymbol, lastTradeDate, strike, lastPrice, bid, ask, change, percentChange, volume, openInterest,
    # impliedVolatility, inTheMoney, contractSize, currency

    # contractSymbol: it is a string such as "AAPL230616C00195000". It is a concatenation of:
    # Underlying symbol ("AAPL")
    # Expiration date ("230616" – only two digits for year)
    # Option type ("C" / "P")
    # Strike price (the rest, here "00195000" means $195 strike)

    # Calculate ATM price
    th = 0.5
    underlying_price2 = tk.history(period='1d')['Close'][0]
    underlying_price = tk.info['currentPrice']
    atm_price = round(underlying_price, 2)
    print(underlying_price,underlying_price2,atm_price)

    # Add optionType and ITM/ATM/OTM columns
    calls['optionType'] = 'C'
    calls['ITM'] = calls['strike'] < underlying_price - th
    calls['ATM'] = (underlying_price - th < calls['strike']) & (calls['strike'] < underlying_price + th)
    calls['OTM'] = calls['strike'] > underlying_price + th

    puts['optionType'] = 'P'
    puts['ITM'] = puts['strike'] > underlying_price + th
    puts['ATM'] = (underlying_price - th < puts['strike']) & (puts['strike'] < underlying_price + th)
    puts['OTM'] = puts['strike'] < underlying_price - th

    # Merge calls and puts into a single dataframe
    exp_data = pd.concat(objs=[calls, puts], ignore_index=True)

    return exp_data

def process_symbol(symbol):
    print()
    print('Symbol: ', symbol)
    tk = yf.Ticker(symbol)          # object
    expirations = tk.options        # Example: ('2023-02-03',...'2025-12-19') strings ("%Y-%m-%d") include all expiration cycles (monthly, weekly)

    # Create empty DataFrame, then add individual expiration data to it
    data = pd.DataFrame()

    for exp_td_str in expirations[:1]:
        exp_data = process_expiration(exp_td_str,tk)
        data = pd.concat(objs=[data, exp_data], ignore_index=True)

    # Add underlyingSymbol column
    data['underlyingSymbol'] = symbol

    return data


# SYMBOLS = ['AAPL', 'MSFT', 'XOM', 'BAC']
SYMBOLS = ['AAPL', 'MSFT', 'XOM', 'BAC']
for symbol in SYMBOLS:
    data = process_symbol(symbol)

    # Separate data for ITM, ATM, and OTM options
    itm_calls = data[(data['optionType'] == 'C') & data['ITM']]
    atm_calls = data[(data['optionType'] == 'C') & data['ATM']]
    otm_calls = data[(data['optionType'] == 'C') & data['OTM']]

    itm_puts = data[(data['optionType'] == 'P') & data['ITM']]
    atm_puts = data[(data['optionType'] == 'P') & data['ATM']]
    otm_puts = data[(data['optionType'] == 'P') & data['OTM']]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.bar(['ATM Calls',  'OTM Calls', 'ITM Calls', 'ITM Puts', 'OTM Puts', 'ATM Puts'],
            [len(itm_calls), len(atm_calls), len(otm_calls), len(itm_puts), len(atm_puts), len(otm_puts)])
    plt.title(f'Option Distribution for {symbol}')
    plt.xlabel('Option Type')
    plt.ylabel('Count')

plt.show()