import pandas as pd
import numpy as np

df = pd.read_csv('event_features_dataset.csv')

# Check autocorrelation of target_direction_50_gap
td = df['target_direction_50_gap']
print('target_direction_50_gap autocorrelation:')
for lag in [1, 5, 10, 20, 50, 100]:
    ac = td.corr(td.shift(lag))
    print(f'  lag={lag}: {ac:.6f}')

# Check how many rows have target_20_gap==0 or target_100_gap==0 that survived
print(f'\ntarget_20_gap == 0: {(df["target_20_gap"]==0).sum()}')
print(f'target_50_gap == 0: {(df["target_50_gap"]==0).sum()}')
print(f'target_100_gap == 0: {(df["target_100_gap"]==0).sum()}')

# Check overall prob_up to see if dataset is balanced
print(f'\nOverall prob_up (target_direction_50_gap): {td.mean():.6f}')
print(f'Total rows: {len(df)}')

# Check cross-correlation between pressure_50 and target_direction_50_gap
print(f'\npressure_50 vs target_direction_50_gap correlation: {df["pressure_50"].corr(td):.6f}')

# Check if the high edge is driven by overlapping windows
# pressure_50 uses trades [i-49, i], target_50_gap uses trades [i+10, i+60]
# Let's check the correlation between pressure_50 and the raw target
print(f'pressure_50 vs target_50_gap correlation: {df["pressure_50"].corr(df["target_50_gap"]):.6f}')

# Effective number of independent observations
# With autocorrelation rho at lag 1, effective N ~ N * (1-rho)/(1+rho)
rho = td.corr(td.shift(1))
n = len(td)
n_eff = n * (1 - rho) / (1 + rho)
print(f'\nAutocorrelation-adjusted effective sample size:')
print(f'  Raw N: {n}')
print(f'  rho(1): {rho:.6f}')
print(f'  N_eff: {n_eff:.0f}')
print(f'  Ratio: {n_eff/n:.4f}')
