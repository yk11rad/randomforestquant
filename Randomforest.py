# Import libraries
import oandapyV20
from oandapyV20.endpoints.instruments import InstrumentsCandles
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# OANDA API Credentials
# Note: Replace with your own API key securely (see README for instructions)
OANDA_API_KEY = "YOUR_OANDA_API_KEY_HERE"  # Placeholder
client = oandapyV20.API(access_token=OANDA_API_KEY, environment="practice")

class RealisticBacktester:
    def __init__(self):
        self.cached_data = {}
        self.slippage = 0.02  # 2 pips for JPY pairs
        self.partial_fill_prob = 0.1
        self.order_delay = 2
        self.spread = 0.02  # 2 pips spread for JPY pairs
        self.commission_per_lot = 0.5  # 0.5 pips commission

    def fetch_candles(self, instrument, granularity, from_time=None, to_time=None):
        cache_key = f"{instrument}_{granularity}_{from_time}_{to_time}"
        if cache_key in self.cached_data:
            print(f"Using cached data for {instrument} ({granularity})")
            return self.cached_data[cache_key]
            
        print(f"Fetching {granularity} candles for {instrument} from {from_time} to {to_time}...")
        all_data = []
        current_from = from_time
        max_candles = 5000
        
        while current_from < to_time:
            chunk_to = min(current_from + timedelta(hours=max_candles * 4), to_time)
            params = {
                "granularity": granularity,
                "from": current_from.strftime('%Y-%m-%dT%H:%M:%SZ'),
                "to": chunk_to.strftime('%Y-%m-%dT%H:%M:%SZ')
            }
            request = InstrumentsCandles(instrument=instrument, params=params)
            response = client.request(request)
            candles = [candle for candle in response['candles'] if candle.get('complete')]
            if not candles:
                break
            df_chunk = pd.DataFrame([{
                'time': pd.to_datetime(candle['time']),
                'open': float(candle['mid']['o']),
                'high': float(candle['mid']['h']),
                'low': float(candle['mid']['l']),
                'close': float(candle['mid']['c']),
                'volume': candle['volume']
            } for candle in candles])
            all_data.append(df_chunk)
            current_from = chunk_to
            time.sleep(1)
        if all_data:
            full_df = pd.concat(all_data).drop_duplicates(subset=['time']).sort_values('time').reset_index(drop=True)
            self.cached_data[cache_key] = full_df
            print(f"Fetched {len(full_df)} candles")
            return full_df
        raise ValueError("No data retrieved")

    def execute_trades(self, df, pip_value, sl_pips, tp_pips):
        trades = []
        active_positions = []
        
        df['signal'] = df['signal'].fillna('NONE')
        
        for i in range(0, len(df)-1):
            current = df.iloc[i]
            
            if current['signal'] != 'NONE' and pd.notna(current['signal_time']) and current['signal_time'] == current['time']:
                if current['signal'] == 'BUY':
                    executed_price, fill_ratio = self.simulate_execution('BUY_STOP', current['close'], current, pip_value)
                    sl = executed_price - sl_pips * pip_value
                    tp = executed_price + tp_pips * pip_value
                    pos_type = 'LONG'
                elif current['signal'] == 'SELL':
                    executed_price, fill_ratio = self.simulate_execution('SELL_STOP', current['close'], current, pip_value)
                    sl = executed_price + sl_pips * pip_value
                    tp = executed_price - tp_pips * pip_value
                    pos_type = 'SHORT'
                
                if fill_ratio > 0:
                    entry_time_with_delay = current['time'] + timedelta(seconds=self.order_delay)
                    active_positions.append({
                        'entry_price': executed_price,
                        'sl': sl,
                        'tp': tp,
                        'type': pos_type,
                        'entry_time': entry_time_with_delay,
                        'size': fill_ratio
                    })
            
            for pos in active_positions[:]:
                if current['time'] <= pos['entry_time']:
                    continue
                
                duration = (current['time'] - pos['entry_time']).total_seconds() / 60
                commission_cost = self.commission_per_lot * pos['size']
                
                if pos['type'] == 'LONG':
                    if current['low'] <= pos['sl']:
                        pips = -sl_pips * pos['size'] - commission_cost
                        trades.append({
                            'entry': pos['entry_price'], 'exit': pos['sl'], 'pips': pips,
                            'type': 'LONG', 'outcome': 'SL', 'duration': duration, 'size': pos['size'],
                            'exit_time': current['time']
                        })
                        active_positions.remove(pos)
                    elif current['high'] >= pos['tp']:
                        pips = tp_pips * pos['size'] - commission_cost
                        trades.append({
                            'entry': pos['entry_price'], 'exit': pos['tp'], 'pips': pips,
                            'type': 'LONG', 'outcome': 'TP', 'duration': duration, 'size': pos['size'],
                            'exit_time': current['time']
                        })
                        active_positions.remove(pos)
                else:
                    if current['high'] >= pos['sl']:
                        pips = -sl_pips * pos['size'] - commission_cost
                        trades.append({
                            'entry': pos['entry_price'], 'exit': pos['sl'], 'pips': pips,
                            'type': 'SHORT', 'outcome': 'SL', 'duration': duration, 'size': pos['size'],
                            'exit_time': current['time']
                        })
                        active_positions.remove(pos)
                    elif current['low'] <= pos['tp']:
                        pips = tp_pips * pos['size'] - commission_cost
                        trades.append({
                            'entry': pos['entry_price'], 'exit': pos['tp'], 'pips': pips,
                            'type': 'SHORT', 'outcome': 'TP', 'duration': duration, 'size': pos['size'],
                            'exit_time': current['time']
                        })
                        active_positions.remove(pos)
        
        return trades

    def simulate_execution(self, order_type, price, candle, pip_value):
        if order_type == 'BUY_STOP':
            executed_price = max(price, candle['open']) + self.slippage + self.spread
            return executed_price, 1.0 if np.random.random() >= self.partial_fill_prob else 0.5
        elif order_type == 'SELL_STOP':
            executed_price = min(price, candle['open']) - self.slippage - self.spread
            return executed_price, 1.0 if np.random.random() >= self.partial_fill_prob else 0.5
        return price, 1.0

    def evaluate_performance(self, trades):
        if not trades:
            return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        pips = np.array([t['pips'] for t in trades])
        durations = np.array([t['duration'] for t in trades])
        total_pips = pips.sum()
        total_trades = len(trades)
        wins = pips[pips > 0]
        losses = pips[pips < 0]
        win_rate = len(wins) / total_trades if total_trades > 0 else 0
        profit_factor = wins.sum() / abs(losses.sum()) if losses.any() else float('inf')
        expectancy = pips.mean() if total_trades > 0 else 0
        sharpe_ratio = pips.mean() / pips.std() * np.sqrt(252) if pips.std() != 0 else 0
        sortino_ratio = pips.mean() / np.std(losses) * np.sqrt(252) if losses.std() != 0 else 0
        running = np.cumsum(pips)
        peak = running[0]
        max_dd = 0
        for value in running:
            if value > peak:
                peak = value
            dd = peak - value
            if dd > max_dd:
                max_dd = dd
        avg_duration = durations.mean() if durations.any() else 0
        downside_risk = np.std(losses) if losses.any() else 0
        return total_pips, total_trades, win_rate, profit_factor, expectancy, sharpe_ratio, sortino_ratio, max_dd, avg_duration, downside_risk

# Strategy Detection Function
def detect_equal_highs_lows(data, tolerance, pip_value):
    data['signal'] = None
    data['signal_time'] = pd.Series(np.nan, index=data.index, dtype='datetime64[ns, UTC]')
    
    eq_high = (data['high'] - data['high'].shift(1)).abs() < tolerance * pip_value
    eq_low = (data['low'] - data['low'].shift(1)).abs() < tolerance * pip_value
    eq_buy = eq_low & (data['close'] > data['open']) & (data['close'].shift(-1) > data['high'])
    eq_sell = eq_high & (data['close'] < data['open']) & (data['close'].shift(-1) < data['low'])
    
    data.loc[eq_buy, 'signal'] = 'BUY'
    data.loc[eq_buy, 'signal_time'] = data.loc[eq_buy, 'time']
    data.loc[eq_sell, 'signal'] = 'SELL'
    data.loc[eq_sell, 'signal_time'] = data.loc[eq_sell, 'time']
    
    return data

# Backtest Function
def backtest(tester, data, tolerance, sl_pips, tp_pips, pip_value):
    data_with_signals = detect_equal_highs_lows(data.copy(), tolerance, pip_value)
    trades = tester.execute_trades(data_with_signals, pip_value, sl_pips, tp_pips)
    return trades, *tester.evaluate_performance(trades)

# Institutional Summary Function
def institutional_summary(best_result):
    trades = best_result['trades']
    pips = np.array([t['pips'] for t in trades])
    durations = np.array([t['duration'] for t in trades])
    
    monthly_pips = {}
    for trade in trades:
        year_month = trade['exit_time'].strftime('%Y-%m')
        monthly_pips[year_month] = monthly_pips.get(year_month, 0) + trade['pips']
    
    duration_buckets = {
        '<4h': len([d for d in durations if d < 240]),
        '4h-12h': len([d for d in durations if 240 <= d < 720]),
        '12h-24h': len([d for d in durations if 720 <= d < 1440]),
        '>24h': len([d for d in durations if d >= 1440])
    }
    
    print("\n=== Institutional Trader Summary (Best Parameters) ===")
    print(f"Parameters: tolerance={best_result['tolerance']:.2f}, SL={best_result['sl_pips']:.2f}, TP={best_result['tp_pips']:.2f}")
    print(f"Total Pips: {best_result['total_pips']:.1f}")
    print(f"Annualized Sharpe Ratio: {best_result['sharpe_ratio']:.2f}")
    print(f"Annualized Sortino Ratio: {best_result['sortino_ratio']:.2f}")
    print(f"Max Drawdown: {best_result['max_dd']:.1f} pips")
    print(f"Downside Risk (Std Dev of Losses): {best_result['downside_risk']:.1f} pips")
    print(f"Profit Factor: {best_result['profit_factor']:.2f}")
    print(f"Total Trades: {len(trades)}")
    
    print("\nMonthly Pip Performance (2020-2024):")
    print("| Year-Month | Pips      |")
    print("|------------|-----------|")
    for year_month, pips in sorted(monthly_pips.items()):
        print(f"| {year_month:<10} | {pips:>9.1f} |")
    
    print("\nTrade Duration Distribution:")
    print("| Range    | Trades |")
    print("|----------|--------|")
    for bucket, count in duration_buckets.items():
        print(f"| {bucket:<8} | {count:>6} |")
    
    print("\nSuggested Chart: Equity Curve")
    print("Plot cumulative pips over time using trade exit timestamps and pip profits.")
    print("X-axis: Date (2020-2024), Y-axis: Cumulative Pips")

# Main Execution
if __name__ == "__main__":
    instrument = "GBP_JPY"
    print(f"\nBacktesting {instrument} on H4 with Equal Highs and Lows...")
    
    # Fetch data once
    tester = RealisticBacktester()
    start_time = datetime(2020, 1, 1, tzinfo=timezone.utc)
    end_time = datetime(2024, 12, 31, tzinfo=timezone.utc)
    data = tester.fetch_candles(instrument, "H4", start_time, end_time)
    pip_value = 0.01
    
    # Define parameter ranges and generate combinations
    tolerance_range = np.linspace(5.0, 30.0, 6)  # 6 points
    sl_pips_range = np.linspace(10.0, 50.0, 5)   # 5 points
    tp_pips_range = np.linspace(30.0, 150.0, 4)  # 4 points
    param_combinations = [(t, sl, tp) for t in tolerance_range for sl in sl_pips_range for tp in tp_pips_range]
    np.random.shuffle(param_combinations)  # Randomize order
    param_combinations = param_combinations[:120]  # Limit to 120
    
    # Run backtests for all combinations
    results = []
    for i, (tolerance, sl_pips, tp_pips) in enumerate(param_combinations, 1):
        print(f"\nRunning backtest {i}/{len(param_combinations)}: tolerance={tolerance:.2f}, sl_pips={sl_pips:.2f}, tp_pips={tp_pips:.2f}")
        trades, total_pips, total_trades, win_rate, profit_factor, expectancy, sharpe_ratio, sortino_ratio, max_dd, avg_duration, downside_risk = backtest(tester, data, tolerance, sl_pips, tp_pips, pip_value)
        results.append({
            'tolerance': tolerance,
            'sl_pips': sl_pips,
            'tp_pips': tp_pips,
            'total_pips': total_pips,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_dd': max_dd,
            'avg_duration': avg_duration,
            'downside_risk': downside_risk,
            'trades': trades
        })
    
    # Prepare data for Random Forest
    X = np.array([[r['tolerance'], r['sl_pips'], r['tp_pips']] for r in results])
    y = np.array([r['total_pips'] for r in results])
    
    # Train Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=1)
    rf.fit(X, y)
    
    # Predict and find best parameters
    predictions = rf.predict(X)
    best_idx = np.argmax(predictions)
    best_result = results[best_idx]
    
    # Display Best Parameters
    print("\n=============================================================")
    print("\n=== Best Parameters (Random Forest Prediction) ===")
    print(f"tolerance: {best_result['tolerance']:.2f}")
    print(f"sl_pips: {best_result['sl_pips']:.2f}")
    print(f"tp_pips: {best_result['tp_pips']:.2f}")
    print(f"Predicted Total Pips: {predictions[best_idx]:.1f}")
    print(f"Actual Total Pips: {best_result['total_pips']:.1f}")
    
    # Find Top 5 by Win Rate
    sorted_by_winrate = sorted(results, key=lambda x: x['win_rate'], reverse=True)[:5]
    print("\n=== Top 5 Parameters by Win Rate ===")
    for i, result in enumerate(sorted_by_winrate, 1):
        print(f"\nTop {i}:")
        print(f"tolerance: {result['tolerance']:.2f}")
        print(f"sl_pips: {result['sl_pips']:.2f}")
        print(f"tp_pips: {result['tp_pips']:.2f}")
        print(f"Total Pips: {result['total_pips']:.1f}")
        print(f"Win Rate: {result['win_rate']:.2%}")
    
    # Institutional Summary for Best Result
    institutional_summary(best_result)

# README
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