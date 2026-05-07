# Quantitative Stock Prediction Based on Machine Learning

## Project Overview

This project builds a quantitative trading strategy based on machine learning using A-share historical stock data.

The project constructs technical indicators and quantitative factors, applies a Random Forest model to predict short-term stock price movements, and performs strategy backtesting to evaluate the effectiveness of the trading strategy.

---

## Tech Stack

- Python
- pandas
- numpy
- scikit-learn
- matplotlib
- akshare

---

## Workflow

1. Collect A-share historical stock data
2. Perform data preprocessing
3. Construct quantitative factors
4. Train machine learning model
5. Predict stock movement direction
6. Backtest trading strategy
7. Visualize cumulative returns

---

## Feature Engineering

The following quantitative factors are constructed:

- Daily Return
- MA5 (5-day Moving Average)
- MA10 (10-day Moving Average)
- Volatility
- Momentum Factor
- Moving Average Difference
- Volume Change Rate

---

## Model

The project uses:

```python
RandomForestClassifier
![Backtest](images/backtest.png)
