"""
Test script to verify Hyperliquid data fetching functionality
"""
import pandas as pd
from src.data.data_loader import DataLoader


def test_hyperliquid_data_fetch():
    print("=" * 70)
    print("Testing Hyperliquid Data Fetching")
    print("=" * 70)
    print()
    
    try:
        # Fetch BTC 1m data
        print("🔍 Fetching BTC 1m data from Hyperliquid...")
        df = DataLoader.fetch_data(
            source="hyperliquid",
            symbol="BTC",
            interval="1m",
            limit=500
        )
        print("✅ Data fetched successfully!")
        print()
        
        # Print summary information
        print("📊 Data Summary:")
        print("-" * 50)
        print(f"Total rows: {len(df)}")
        print(f"Columns: {list(df.columns)}")
        # print(f"Time range: {df.index[0]} to {df.index[-1]}")
        if df.empty:
            print("❌ DataFrame is EMPTY — no data returned")
            return
        print(f"Index type: {type(df.index)}")
        print()
        
        # Print first 5 rows
        print("📈 First 5 rows:")
        print("-" * 50)
        print(df.head())
        print()
        
        # Print last 5 rows
        print("📉 Last 5 rows:")
        print("-" * 50)
        print(df.tail())
        print()
        
        # Print DataFrame info
        print("ℹ️ DataFrame Info:")
        print("-" * 50)
        df.info()
        print()
        
        # Validate data
        print("✅ Validation Checks:")
        print("-" * 50)
        
        # Check required columns exist
        required_columns = ["Open", "High", "Low", "Close", "Volume"]
        all_columns_present = all(col in df.columns for col in required_columns)
        print(f"All required columns present: {all_columns_present}")
        
        # Check no NaN values
        has_no_nans = not df.isnull().values.any()
        print(f"No NaN values present: {has_no_nans}")
        
        # Check index is datetime
        index_is_datetime = isinstance(df.index, pd.DatetimeIndex)
        print(f"Index is DatetimeIndex: {index_is_datetime}")
        
        # Check price logic
        price_logic_valid = (df["High"] >= df["Low"]).all()
        print(f"All High >= Low: {price_logic_valid}")
        
        print()
        print("🎉 All tests completed successfully!")
        print("=" * 70)
        
    except Exception as e:
        print(f"❌ Error during testing: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_hyperliquid_data_fetch()