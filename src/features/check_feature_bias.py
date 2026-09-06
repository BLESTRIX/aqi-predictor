# check_feature_bias.py
import pandas as pd
import numpy as np
import hopsworks
from src.config import CONFIG, HOPSWORKS_API_KEY

project = hopsworks.login(
    api_key_value=HOPSWORKS_API_KEY,
    project=CONFIG["feature_store"]["project_name"])
fs = project.get_feature_store()

fg = fs.get_feature_group(
    name=CONFIG["feature_store"]["feature_group_name"],
    version=CONFIG["feature_store"]["feature_group_version"]
)
df = fg.read(read_options={"use_hive": False})
df = df.sort_values("time").reset_index(drop=True)

print(f"Total rows in feature store: {len(df)}")
print(f"Date range: {df['time'].min()} -> {df['time'].max()}\n")

# 1. Check AQI category distribution AFTER 12h resampling (what the model actually sees)
bins = [0, 50, 100, 150, 200, 300, float("inf")]
labels = ["Good", "Moderate", "Unhealthy(Sensitive)", "Unhealthy", "Very Unhealthy", "Hazardous"]
df["category"] = pd.cut(df["us_aqi"], bins=bins, labels=labels)
print("=== Category distribution (post-resampling, what model trains on) ===")
print(df["category"].value_counts().reindex(labels))
print(f"\n% at Very Unhealthy or worse: {(df['us_aqi'] > 200).mean() * 100:.2f}%")
print(f"Max us_aqi in feature store: {df['us_aqi'].max():.1f}")

# 2. Check train/test split distribution mismatch (mimics train.py's actual split)
from sklearn.model_selection import train_test_split
target_cols = [c for c in df.columns if c.startswith("target_")]
df_clean = df.dropna(subset=target_cols)
train_df, test_df = train_test_split(df_clean, test_size=0.2, shuffle=False)

print(f"\n=== Train/test split comparison ===")
print(f"Train: {train_df['time'].min()} -> {train_df['time'].max()}  (n={len(train_df)})")
print(f"Test:  {test_df['time'].min()} -> {test_df['time'].max()}  (n={len(test_df)})")
print(f"\nTrain us_aqi mean/std: {train_df['us_aqi'].mean():.1f} / {train_df['us_aqi'].std():.1f}")
print(f"Test  us_aqi mean/std: {test_df['us_aqi'].mean():.1f} / {test_df['us_aqi'].std():.1f}")

print(f"\nTrain month distribution:\n{train_df['time'].dt.month.value_counts().sort_index()}")
print(f"\nTest month distribution:\n{test_df['time'].dt.month.value_counts().sort_index()}")

# 3. Check how much resampling smoothed out extremes vs raw hourly data
print(f"\n=== Effect of 12h resampling on extremes ===")
print(f"Rows with us_aqi > 150 in feature store: {(df['us_aqi'] > 150).sum()} ({(df['us_aqi'] > 150).mean()*100:.1f}%)")
print(f"Rows with us_aqi > 200 in feature store: {(df['us_aqi'] > 200).sum()} ({(df['us_aqi'] > 200).mean()*100:.1f}%)")