# randomforestquant
# Overview:
# This script implements a realistic backtesting framework for a forex trading strategy on GBP/JPY using OANDA's API. It fetches H4 candlestick data from 2020-2024, detects signals based on equal highs and lows, and backtests across 120 parameter combinations (tolerance, stop-loss, take-profit). A Random Forest model optimizes parameters for total pips, and an institutional-style summary provides performance metrics. The script is designed to run in Google Colab.
#
# Usage Instructions:
# 1. Open Google Colab (colab.research.google.com) and create a new notebook.
# 2. Install required libraries: `!pip install oandapyV20 scikit-learn`.
# 3. Set up your OANDA API key:
#    - Replace `OANDA_API_KEY = "YOUR_OANDA_API_KEY_HERE"` with your key.
#    - Alternatively, use Colab’s input prompt: `OANDA_API_KEY = input("Enter OANDA API key: ")`.
#    - Do not hardcode credentials to ensure security.
# 4. Copy and paste this code into a cell and run it using Shift + Enter.
# 5. Review outputs: backtest results for each combination, best parameters from Random Forest, top 5 by win rate, and institutional summary (pips, Sharpe ratio, monthly performance, trade durations).
#
# Dependencies:
# - Python 3.x
# - Libraries: oandapyV20, pandas, numpy, scikit-learn
# - Note: Install `oandapyV20` and `scikit-learn` in Colab; pandas and numpy are pre-installed.
#
# Adapting to Your Needs:
# - Change the instrument by modifying `instrument = "GBP_JPY"` (e.g., to "EUR_USD").
# - Adjust the time frame by changing `granularity = "H4"` (e.g., to "H1", "D").
# - Modify the date range in `start_time` and `end_time` for different periods.
# - Update `tolerance_range`, `sl_pips_range`, or `tp_pips_range` to change parameter search space.
# - Customize `detect_equal_highs_lows` to implement your own signal logic.
# - Adjust `pip_value`, `slippage`, `spread`, or `commission_per_lot` in `RealisticBacktester` to match your broker’s conditions.
# - Modify the Random Forest model (e.g., `n_estimators`) or use a different algorithm.
#
# Notes:
# - Ensure a stable internet connection, as the script fetches data from OANDA’s practice API.
# - The script uses a practice environment (`environment="practice"`). For live data, update to `environment="live"` with caution and a live API key.
# - Backtesting 120 combinations may take time depending on data size and Colab’s resources.
# - Cache is used to avoid redundant API calls; clear `tester.cached_data` if data updates are needed.
# - Monitor Colab’s memory usage for large datasets to avoid crashes.
# - Contact for support or enhancements (e.g., adding new metrics, strategies, or visualizations).
