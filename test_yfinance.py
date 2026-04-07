from src.data_loader import DataLoader
import pandas as pd
import yfinance as yf

# Test with Apple stock - you can change this ticker
ticker = "NVDA"
start_date = "2023-01-01"
end_date = "2024-01-01"

print(f"✅ Fetching data for {ticker} from {start_date} to {end_date}...")
data = DataLoader.fetch_data(ticker, start_date, end_date)

print("\n📊 Raw Pandas DataFrame Preview:")
print("=" * 80)
print(f"✅ Total rows fetched: {len(data)}")
print(f"✅ Date range: {data.index[0].date()} to {data.index[-1].date()}")
print(f"✅ Columns present: {list(data.columns)}")
print("\nFirst 5 rows:")
print(data.head())
print("\nLast 5 rows:")
print(data.tail())
print("\nDataFrame info:")
data.info()
print("\nDescriptive statistics:")
print(data.describe())

# Also test raw yfinance directly to compare
print("\n" + "="*80)
print("🔍 Testing raw yfinance download directly:")
raw_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
print(f"Raw yfinance returned {len(raw_data)} rows")
print(f"Raw columns: {list(raw_data.columns)}")

if len(data) == 0:
    print("\n❌ ERROR: NO DATA RETURNED!")
    print("This is the root cause of your backtest not working")
    print("\nPossible fixes:")
    print("1. Check your internet connection")
    print("2. Yahoo Finance might be blocking your IP temporarily")
    print("3. Try a different ticker symbol")
    print("4. Update yfinance: pip install --upgrade yfinance")