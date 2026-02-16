import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime

r = 0.05  # trial risk-free rate

expiry = datetime(2026, 3, 13)
today = datetime.today()
T = (expiry-today).days/365


ticker = yf.Ticker("AAPL")

spot = ticker.history(period="1d")["Close"].iloc[-1]
options = ticker.option_chain(expiry.strftime("%Y-%m-%d"))
calls = options.calls
puts = options.puts


def black_scholes_call(S0, K, vol, r, T):
    d1 = (np.log(S0/K) + (r + 0.5*(vol**2))*T)/(vol*np.sqrt(T))
    d2 = d1 - vol*np.sqrt(T)

    return S0*norm.cdf(d1) - K*np.exp(-1*r*T)*norm.cdf(d2)


def black_scholes_put(S0, K, vol, r, T):
    d1 = (np.log(S0/K) + (r + 0.5*(vol**2))*T)/(vol*np.sqrt(T))
    d2 = d1 - vol*np.sqrt(T)

    return K*np.exp(-1*r*T)*norm.cdf(-1*d2) - S0*norm.cdf(-1*d1)


def impliedVolatility(C_market, S, K, r, T, option_type, tol=1e-3, init=0.2):
    vol = init
    price = tol + C_market + 1
    while (abs(price-C_market) > tol):
        if (option_type=="call"):
            price = black_scholes_call(S, K, vol, r, T)
        else :
            price = black_scholes_put(S, K, vol, r, T)

        vega = S*norm.pdf((np.log(S/K)+(r+0.5*vol**2)*T)/(vol*np.sqrt(T)))*np.sqrt(T)
        vol -= (price - C_market)/vega
    return vol


calls["iv"] = calls.apply(lambda row: impliedVolatility(row["lastPrice"],spot,row["strike"],r, T,"call"),axis=1)
puts["iv"] = puts.apply(lambda row: impliedVolatility(row["lastPrice"],spot,row["strike"],r,T,"put"),axis=1)


print("Spot Price:", spot)
print("Calls with IV:")
print(calls[["strike", "lastPrice", "iv"]].head())

print("\nPuts with IV:")
print(puts[["strike", "lastPrice", "iv"]].head())
