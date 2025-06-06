# data_quality_agent.py

import pandas as pd
import numpy as np
from scipy import stats

# 1. Simulate pipeline output (replace with real data when you have it)
np.random.seed(42)
data = {
    'id': range(1, 21),
    'transaction_volume': np.random.normal(loc=100, scale=15, size=20),
    'agent_count': np.append(np.random.randint(5, 15, size=18), [np.nan, 8]),
    'listing_count': np.random.normal(loc=50, scale=10, size=20)
}
df = pd.DataFrame(data)

# Introduce some anomalies to demonstrate detection
df.loc[3, 'transaction_volume'] = 180  # High outlier
df.loc[15, 'listing_count']     = 10   # Low outlier

# 2. Define a function to detect missing values and outliers
def detect_issues(df):
    issues = {}

    # a) Check for missing values
    missing_summary = df.isnull().sum().to_dict()
    issues['missing_values'] = missing_summary

    # b) Detect outliers using z-score (|z| > 2)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    filled = df[numeric_cols].fillna(df[numeric_cols].mean())
    z_scores = np.abs(stats.zscore(filled))

    outliers = {}
    for col, z_col in zip(numeric_cols, z_scores.T):
        indices = np.where(z_col > 2)[0].tolist()
        outliers[col] = indices
    issues['outliers'] = outliers

    return issues

if __name__ == '__main__':
    issues_found = detect_issues(df)
    print("Issues Found:")
    print(issues_found)
