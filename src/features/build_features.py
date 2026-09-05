import pandas as pd
import numpy as np
from src.config import CONFIG

def enforce_12h_cadence(df: pd.DataFrame) -> pd.DataFrame:
    """Resamples the DataFrame to an exact 12-hour grid and interpolates missing values."""
    df = df.sort_values("time").set_index("time")
    
    # Resample to strict 12-hour frequency
    resampled_df = df.resample("12h").mean(numeric_only=True)
    
    # Interpolate ALL numeric columns to prevent dropna() from deleting our grid
    resampled_df = resampled_df.interpolate(method="linear")
    
    return resampled_df.reset_index()

def create_target_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Defines 12-hour target steps:
    shift(-2) = 24 hours ahead (2 * 12h)
    shift(-4) = 48 hours ahead (4 * 12h)
    shift(-6) = 72 hours ahead (6 * 12h)
    """
    df["target_aqi_24h"] = df["us_aqi"].shift(-2)
    df["target_aqi_48h"] = df["us_aqi"].shift(-4)
    df["target_aqi_72h"] = df["us_aqi"].shift(-6)
    return df
def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts calendar features and cyclical encodings."""
    df["day_of_week"] = df["time"].dt.dayofweek
    df["day"] = df["time"].dt.day
    df["month"] = df["time"].dt.month
    
    # Cyclical sine/cosine transforms for month
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12.0)
    return df

def create_daily_lags_and_rolling(df: pd.DataFrame) -> pd.DataFrame:
    """Computes daily lag metrics and rolling statistics."""
    # 1-day historical shift (24 hours ago in daily cadence)
    df["aqi_lag_1d"] = df["us_aqi"].shift(1)
    df["aqi_change_rate_1d"] = (df["us_aqi"] - df["aqi_lag_1d"]) / (df["aqi_lag_1d"] + 1e-5)

    # 3-day rolling window average
    df["aqi_rolling_mean_3d"] = df["us_aqi"].rolling(window=3, min_periods=1).mean()
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Main feature engineering entrypoint."""
    # 1. Standardize to a 12-hour grid first
    df = enforce_12h_cadence(df)

    # 2. Compute features and multi-step target shifts
    df = create_time_features(df)
    df = create_daily_lags_and_rolling(df)
    df = create_target_variables(df)

    # Hopsworks metadata and primary keys
    df["location"] = CONFIG["location"]["name"].lower()
    df["event_timestamp"] = df["time"].astype("int64") // 10**6

    feature_cols = [c for c in df.columns if not c.startswith("target_")]
    return df.dropna(subset=feature_cols).reset_index(drop=True)
if __name__ == "__main__":
    from src.data.api_client import fetch_raw_air_quality

    raw_df = fetch_raw_air_quality()
    processed_df = engineer_features(raw_df)
    
    print(f"Engineered {processed_df.shape[1]} features across {len(processed_df)} daily records for {CONFIG['location']['name']}:")
    print(processed_df[["time", "us_aqi", "aqi_lag_1d", "target_aqi_24h", "target_aqi_48h", "target_aqi_72h"]].head(10))