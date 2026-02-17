import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
from datetime import datetime

r = 0.05  # trial risk-free rate

expiry = datetime(2026,2,27)
today = datetime.today()
T = (expiry-today).days/365


ticker = yf.Ticker("NVDA")
print(ticker.options)
spot = ticker.history(period="1d")["Close"].iloc[-1]
options = ticker.option_chain('2026-02-27')
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


def impliedVolatility(C_market, S, K, r, T, option_type):

    def objective(vol):
        if option_type == "call":
            return black_scholes_call(S, K, vol, r, T) - C_market
        else:
            return black_scholes_put(S, K, vol, r, T) - C_market

    try:
        sol = root_scalar(objective, bracket=[1e-6, 5], method="brentq")
        return sol.root
    except:
        return np.nan


calls["iv"] = calls.apply(lambda row: impliedVolatility(row["lastPrice"],spot,row["strike"],r, T,"call"),axis=1)
puts["iv"] = puts.apply(lambda row: impliedVolatility(row["lastPrice"],spot,row["strike"],r,T,"put"),axis=1)


print("Spot Price:", spot)
print("Calls with IV:")
print(calls[["strike", "lastPrice", "iv"]].head())

print("\nPuts with IV:")
print(puts[["strike", "lastPrice", "iv"]].head())


calls_clean = calls[(calls["iv"].notna()) & (calls["iv"] > 0) &(calls["iv"] < 3)].copy()

puts_clean = puts[(puts["iv"].notna()) & (puts["iv"] > 0) & (puts["iv"] < 3)].copy()


plt.figure(figsize=(10,6))

plt.scatter(calls_clean["strike"], calls_clean["iv"], label="Calls")
plt.scatter(puts_clean["strike"], puts_clean["iv"], label="Puts")

plt.axvline(spot, linestyle="--", label="Spot")

plt.xlabel("Strike")
plt.ylabel("Implied Volatility")
plt.title("NVDA Volatility Smile (Expiry 2026-02-27)")
plt.legend()

plt.savefig("nvda_smile.png", dpi=300)
plt.close()