import pandas as pd
from src.features.build_features import engineer_features
from src.features.feature_store import push_to_feature_store
from src.config import CONFIG


def load_historical_aqicn_data(file_path: str) -> pd.DataFrame:
    """
    Loads a historical PM2.5 CSV downloaded from aqicn.org/historical
    (format: 'date, pm25' with date as YYYY/M/D, one row per day, for the
    Islamabad US Embassy station).
    """
    df = pd.read_csv(file_path)
    df.columns = [c.strip() for c in df.columns]  # strips leading space in " pm25"

    df["time"] = pd.to_datetime(df["date"], format="%Y/%m/%d")
    df = df.drop(columns=["date"])

    # AQICN's historical PM2.5 export is already an AQI sub-index (0-500 scale).
    # PM2.5 is the dominant pollutant driving AQI in Islamabad, so this doubles
    # as us_aqi directly - no separate conversion needed.
    df["us_aqi"] = df["pm25"].astype(float)
    df["pm2_5"] = df["pm25"].astype(float)
    df = df.drop(columns=["pm25"])

    # This source only provides PM2.5 - no other pollutants or weather data.
    # Filled with 0.0 to keep the schema consistent with the live API payload;
    # these columns carry no signal for the historical portion of the dataset
    # and are excluded from model features in train.py / predict.py.
    for col in ["pm10", "nitrogen_dioxide", "sulphur_dioxide", "carbon_monoxide",
                "ozone", "temperature", "humidity", "pressure", "wind_speed"]:
        df[col] = 0.0

    df["record_type"] = "historical"
    return df.sort_values("time").reset_index(drop=True)


if __name__ == "__main__":
    historical_data_path = "data/history_ISL_AQI.csv"

    print("1. Loading historical AQICN PM2.5 data...")
    raw_hist_df = load_historical_aqicn_data(historical_data_path)
    print(f"   {len(raw_hist_df)} daily rows, {raw_hist_df['time'].min()} -> {raw_hist_df['time'].max()}")

    print("2. Engineering multi-step targets and lag features...")
    processed_hist_df = engineer_features(raw_hist_df)

    print(f"Engineered {len(processed_hist_df)} historical records.")

    print("3. Uploading historical batch to Hopsworks Feature Store...")
    push_to_feature_store(processed_hist_df)