import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt


tickers = ['AAPL','MSFT','AMZN']
number_of_shares = np.array([1, 2, 4])


start_date = datetime(2020, 1, 1)
#end_date = datetime(2025, 1, 1)
end_date = datetime.today()
print(end_date)
data = yf.download(tickers, start=start_date, end=end_date)
returns = np.log(data['Close'] / data['Close'].shift(1)).dropna()
last_close = data['Close'].iloc[-1]


'''
def weights_of_shares(tickers=tickers, number_of_shares=number_of_shares, last_close=last_close):
    for i in np.arange(0, len(tickers)):
        dollars_of_stock = last_close[tickers[i]] * number_of_shares[i]
        print(dollars_of_stock)

        sum_of_value = 0
        sum_of_value = sum_of_value + dollars_of_stock
        print(sum_of_value)
weights_of_shares()
'''


def weights_of_shares(tickers, number_of_shares, last_close):
    values = [last_close[t] * s for t, s in zip(tickers, number_of_shares)]
    total = sum(values)
    
    # Calculate portfolio weights
    weights = [v / total for v in values]
    
    for t, v, w in zip(tickers, values, weights):
        print(f"{t}: ${v:,.2f} ({w:.2%} of portfolio)")
    print(f"Total portfolio value: ${total:,.2f}")
    
    return weights, total

weights, total_value = weights_of_shares(tickers, number_of_shares, last_close)

investment = total_value

mean_returns = returns.mean()
cov_matrix = returns.cov()

def monte_carlo(num_iterations=100000, num_days=365):
    results = np.zeros(num_iterations)

    for i in range(num_iterations):
        daily_returns = np.random.multivariate_normal(mean_returns, cov_matrix, num_days)
        portfolio_returns = np.dot(daily_returns, weights)
        portfolio_growth = np.exp(np.sum(portfolio_returns))
        results[i] = investment * portfolio_growth
    return results

simulate_results = monte_carlo()

VaR_95 = np.percentile(simulate_results, 5)
CVaR_95 = simulate_results[simulate_results <= VaR_95].mean()

print(f"Mean Final Portfolio Value: ${simulate_results.mean():,.2f}")
print(f"95% VaR: ${investment - VaR_95:,.2f}")
print(f"95% CVaR: ${investment - CVaR_95:,.2f}")


# Histogram plot
plt.figure(figsize=(10,6))
plt.hist(simulate_results, bins=50)
plt.axvline(VaR_95, color='r', linestyle='dashed', linewidth=2, label='VaR 95%')
plt.title('Monte Carlo Portfolio Simulation')
plt.xlabel('Portfolio Value')
plt.ylabel('Frequency')
plt.legend()
plt.show()
