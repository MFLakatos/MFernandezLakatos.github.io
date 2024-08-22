import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from smabacktest import SMABacktester

symbol =  'AAPL'
start = "2024-02-01"
end = "2024-03-13"      # y-m-d
# SMABacktester(symbol,SMA_S, SMA_L,start,end)

Apple = yf.download('AAPL', start=start, end=end)

print(Apple)