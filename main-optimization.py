# Advanced Portfolio Optimization
import numpy as np
import pandas as pd
import torch

# Step 1: Define your portfolio
assets = ['AAPL', 'GOOGL', 'AMZN', 'MSFT', 'NEB', 'SPY', 'QQQ'] 
initial_prices = torch.tensor([175.0, 135.0, 125.0, 310.0, 25.0, 430.0, 350.0], device='cuda')

# Expected annual returns (approximate)
mu = torch.tensor([0.12, 0.10, 0.15, 0.11, 0.20, 0.08, 0.09], device='cuda')

# Annual volatilities (approximate)
sigma = torch.tensor([0.30, 0.25, 0.35, 0.28, 0.50, 0.15, 0.18], device='cuda')

# Custom target allocation ranges (min/max per asset, sum to 1)
min_alloc = torch.tensor([0.05, 0.05, 0.05, 0.05, 0.05, 0.10, 0.10], device='cuda')
max_alloc = torch.tensor([0.25, 0.20, 0.20, 0.25, 0.20, 0.30, 0.25], device='cuda')

num_simulations = 500_000
num_days = 252

# Step 2: Simulate price paths using GBM 
dt = 1 / num_days
simulations = torch.zeros((num_simulations, len(assets)), device='cuda')

for i, (S0, mu_i, sigma_i) in enumerate(zip(initial_prices, mu, sigma)):
    rand = torch.randn(num_simulations, num_days, device='cuda')
    price_paths = S0 * torch.exp(torch.cumsum((mu_i - 0.5*sigma_i**2)*dt + sigma_i*torch.sqrt(dt)*rand, dim=1))
    simulations[:, i] = price_paths[:, -1]

# Step 3: Generate random portfolio weights within min/max bounds
weights = torch.rand((num_simulations, len(assets)), device='cuda')
# Scale weights to min/max allocation
weights = min_alloc + weights * (max_alloc - min_alloc)
weights /= weights.sum(dim=1, keepdim=True)

# Portfolio returns and volatility
portfolio_returns = (simulations * weights).sum(dim=1)
portfolio_volatility = torch.std(portfolio_returns)

# Sharpe ratio assuming risk-free rate = 0
sharpe_ratio = portfolio_returns / portfolio_volatility

# Find best portfolio
best_idx = torch.argmax(sharpe_ratio)
best_weights = weights[best_idx].cpu().numpy()

# Step 4: Output results
print("\nOptimal Portfolio Allocation (based on Sharpe Ratio):")
for asset, weight in zip(assets, best_weights):
    print(f"{asset}: {weight:.2%}")

print(f"\nExpected Portfolio Return: {portfolio_returns[best_idx].item():.2f}")
print(f"Portfolio Volatility: {portfolio_volatility.item():.2f}")
print(f"Sharpe Ratio: {sharpe_ratio[best_idx].item():.2f}")

# Optional: save all simulated portfolio data
pd.DataFrame({
    'Return': portfolio_returns.cpu().numpy(),
    'Sharpe': sharpe_ratio.cpu().numpy()
}).to_csv('gpu_portfolio_simulation.csv', index=False)

print("\nSimulation complete! CSV saved with all simulated portfolios.")
