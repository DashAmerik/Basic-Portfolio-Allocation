import yfinance as yf
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# define list of assets in our portfolio
assets = ['AAPL','MSFT','AMZN','GOOGL','META','NVDA','TSLA','BRK-B','JPM','V',
    'JNJ','WMT','PG','MA','HD','XOM','BAC','CVX','ABBV','AVGO',
    'COST','ADBE','KO','PEP','PFE','TMO','CSCO','MRK','ABT','ACN',
    'CRM','DIS','NFLX','INTC','AMD','MCD','NKE','TXN','LIN','UNH',
    'VZ','QCOM','UPS','RTX','IBM','AMAT','CAT','LOW','HON','ORCL']

# record number of assets
num_assets = len(assets)

#randomly initialize portfolio weights, avoid bias towards equal weighted portfolio
weights = np.random.rand(num_assets)
weights = weights / np.sum(weights)

# download data from yfinance for 3 years and calculate daily returns from our data as percentages, removing missing values (i.e. first value in collumn)
data = yf.download(assets, start="2022-01-01", end="2025-01-01")['Close']
daily_returns = data.pct_change().dropna()

# fetches current risk free rate based on most recent treasury yield, stores it as a float so it can be used in sharpe ratio calculations
risk_free_rate = yf.Ticker("^TNX").history(period="1d")["Close"].iloc[-1] / 100

#function for calculating portfolio performance
def portfolio_performance(weights, returns):
    portfolio_return = np.sum(returns.mean() * weights) *252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov()*252,weights)))
    return portfolio_return, portfolio_volatility

#function for calculating negative sharp ratio
def negative_sharp_ratio(weights, returns, risk_free_rate):
    portfolio_return, portfolio_volatility = portfolio_performance(weights, returns)
    negative_sharpe_ratio = -((portfolio_return - risk_free_rate) / portfolio_volatility)
    return negative_sharpe_ratio

#define our weight constraint
def weight_constraint(weights):
    return np.sum(weights)-1

constraints = ({'type': 'eq', 'fun': weight_constraint}) #set constraints to 'equal' type with function weight_constrain
bounds = tuple((0,1) for _ in range(num_assets))

#minimize the negative_sharp_ratio (maximise sharp ratio) by changing weights using our downloaded daily_returns and risk free rate
result = minimize(negative_sharp_ratio, weights, args=(daily_returns, risk_free_rate), method = 'SLSQP',bounds=bounds,constraints=constraints)

optimized_weights = result.x #get our optimized weights from our result

optimized_weights_percent = optimized_weights * 100

optimized_return, optimized_volatility = portfolio_performance(optimized_weights, daily_returns) #calculate our optimized_return and optimized volatility from out optimized weights

optimized_sharpe_ratio = (optimized_return - risk_free_rate) / optimized_volatility

print("Optimized Weights (in %):")
for i, weight in enumerate(optimized_weights_percent):
    print(f"{assets[i]}: {weight:.2f}%")

print("\nOptimized Weights Sum Check:", np.sum(optimized_weights_percent))
print("Optimized Return:", optimized_return)
print("Optimized Volatility:", optimized_volatility)
print("Optimized Sharpe Ratio:", optimized_sharpe_ratio)

equal_weights = np.ones(num_assets) / num_assets
equal_return, equal_volatility = portfolio_performance(equal_weights, daily_returns)
equal_sharpe = (equal_return - risk_free_rate) / equal_volatility
print("\n--Comparison to Equally Weighter Portfolio--: ")
print("Equally Weighted Portf olio Return:", equal_return)
print("Equally Weighted Portfolio Volatility:", equal_volatility)
print("Equally Weighted Sharpe Ratio:", equal_sharpe)

plt.figure(figsize=(14,6))
plt.bar(assets, optimized_weights_percent)
plt.xticks(rotation=90)
plt.ylabel("% Portfolio Allocation")
plt.title("Optimized Portfolio Weights")
plt.tight_layout()
plt.show()
