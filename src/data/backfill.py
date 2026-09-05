import pandas as pd
from src.features.build_features import engineer_features
from src.features.feature_store import push_to_feature_store
from src.config import CONFIG

def load_historical_aqicn_data(file_path: str) -> pd.DataFrame:
    """
    Loads historical AQICN data from a CSV or API.
    Rename columns to match the live API schema expected by build_features.py.
    """
    # Example for a downloaded AQICN historical CSV
    df = pd.read_csv(file_path)
    
    # Standardize column names to match your live pipeline
    df = df.rename(columns={
        "date": "time",
        "pm25": "pm2_5",
        "pm10": "pm10",
        "o3": "ozone",
        "no2": "nitrogen_dioxide",
        "so2": "sulphur_dioxide",
        "co": "carbon_monoxide"
    })
    
    # Convert time strings to datetime objects
    df["time"] = pd.to_datetime(df["time"])
    
    # Map the primary AQI column (AQICN usually uses PM2.5 as the main AQI driver)
    if "us_aqi" not in df.columns:
        df["us_aqi"] = df["pm2_5"] 
        
    # Ensure missing weather columns exist if your historical data lacks them
    for col in ["temperature", "humidity", "pressure", "wind_speed"]:
        if col not in df.columns:
            df[col] = 0.0
            
    df["record_type"] = "historical"
    return df.sort_values("time").reset_index(drop=True)

if __name__ == "__main__":
    # Replace with your actual historical data retrieval method or file path
    historical_data_path = "data/islamabad_aqicn_historical.csv"
    
    print("1. Loading historical AQICN data...")
    raw_hist_df = load_historical_aqicn_data(historical_data_path)
    
    print("2. Engineering multi-step targets and lag features...")
    # This uses the EXACT SAME logic as your live pipeline
    processed_hist_df = engineer_features(raw_hist_df)
    
    print(f"Engineered {len(processed_hist_df)} historical records.")
    
    print("3. Uploading historical batch to Hopsworks Feature Store...")
    push_to_feature_store(processed_hist_df)