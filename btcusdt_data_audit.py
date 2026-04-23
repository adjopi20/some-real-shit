import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import os
from datetime import datetime, timedelta
import pytz

def analyze_chunk_files(directory):
    """Analyze BTCUSDT chunk files for data integrity issues."""
    # List all parquet files
    files = [f for f in os.listdir(directory) if f.endswith('.parquet')]
    files.sort()
    
    # Initialize results storage
    results = {
        'missing_data': [],
        'duplicates': [],
        'order_violations': [],
        'type_mismatches': [],
        'min_timestamp': float('inf'),
        'max_timestamp': float('-inf'),
        'total_trades': 0
    }
    
    # Process each file
    all_ids = set()
    last_timestamp = None
    
    for filename in files:
        filepath = os.path.join(directory, filename)
        table = pq.read_table(filepath)
        df = table.to_pandas()
        
        # Track min/max timestamps
        file_min = df['timestamp'].min()
        file_max = df['timestamp'].max()
        results['min_timestamp'] = min(results['min_timestamp'], file_min)
        results['max_timestamp'] = max(results['max_timestamp'], file_max)
        results['total_trades'] += len(df)
        
        # Check for missing data
        missing = df.isnull().sum()
        if missing.sum() > 0:
            results['missing_data'].append({
                'file': filename,
                'missing_counts': missing[missing > 0].to_dict()
            })
        
        # Check for duplicates (within file and across files)
        file_duplicates = df[df.duplicated('local_id', keep=False)]
        if not file_duplicates.empty:
            results['duplicates'].append({
                'file': filename,
                'duplicate_ids': file_duplicates['local_id'].unique().tolist(),
                'count': len(file_duplicates)
            })
        
        # Check for cross-file duplicates
        new_duplicates = [id_ for id_ in df['local_id'] if id_ in all_ids]
        if new_duplicates:
            results['duplicates'].append({
                'file': filename,
                'cross_file_duplicates': list(set(new_duplicates)),
                'count': len(new_duplicates)
            })
        all_ids.update(df['local_id'])
        
        # Check chronological order
        if last_timestamp is not None and df['timestamp'].min() < last_timestamp:
            results['order_violations'].append({
                'file': filename,
                'violation': f"Min timestamp {df['timestamp'].min()} < previous max {last_timestamp}"
            })
        last_timestamp = df['timestamp'].max()
        
        # Check data types
        type_issues = {}
        if not df['price'].apply(lambda x: isinstance(x, float)).all():
            type_issues['price'] = 'Non-float values found'
        if not df['quantity'].apply(lambda x: isinstance(x, float)).all():
            type_issues['quantity'] = 'Non-float values found'
        if not df['side'].apply(lambda x: x in [-1, 1]).all():
            type_issues['side'] = 'Invalid side values (should be -1 or 1)'
            
        if type_issues:
            results['type_mismatches'].append({
                'file': filename,
                'issues': type_issues
            })
    
    # Convert timestamps to WIB (UTC+7)
    wib = pytz.timezone('Asia/Jakarta')
    min_dt = datetime.fromtimestamp(results['min_timestamp']/1000).astimezone(wib)
    max_dt = datetime.fromtimestamp(results['max_timestamp']/1000).astimezone(wib)
    
    # Filter out non-issue keys
    issue_keys = ['missing_data', 'duplicates', 'order_violations', 'type_mismatches']
    issues = {k: v for k, v in results.items() if k in issue_keys and v}
    
    return {
        'issues': issues,
        'time_range': {
            'start': min_dt.strftime('%Y-%m-%d %H:%M:%S %Z'),
            'end': max_dt.strftime('%Y-%m-%d %H:%M:%S %Z')
        },
        'total_files': len(files),
        'total_trades': results['total_trades']
    }

if __name__ == "__main__":
    directory = "storage/btcusdt"
    print("Analyzing BTCUSDT chunk files...")
    results = analyze_chunk_files(directory)
    
    print("\n=== Analysis Results ===")
    print(f"Time Range (WIB): {results['time_range']['start']} to {results['time_range']['end']}")
    print(f"Total Files: {results['total_files']}")
    print(f"Total Trades: {results['total_trades']}")
    
    if results['issues']:
        print("\n=== Data Integrity Issues ===")
        for issue_type, issues in results['issues'].items():
            # Convert numpy int to native Python int for len()
            issue_count = int(issues) if hasattr(issues, 'item') else len(issues)
            print(f"\n{issue_type.upper()} ({issue_count} files affected):")
            if isinstance(issues, list):
                for issue in issues:
                    print(f"  File: {issue['file']}")
                    for k, v in issue.items():
                        if k != 'file':
                            print(f"    {k}: {v}")
            else:
                print(f"  Details: {issues}")
    else:
        print("\nNo data integrity issues found!")