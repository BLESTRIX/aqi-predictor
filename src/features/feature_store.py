import hopsworks
import pandas as pd
from src.config import CONFIG, HOPSWORKS_API_KEY

def push_to_feature_store(df: pd.DataFrame) -> None:
    """Authenticates with Hopsworks and writes the engineered features into the Feature Store."""
    if not HOPSWORKS_API_KEY:
        raise ValueError("HOPSWORKS_API_KEY is missing from your environment or .env file.")

    # 1. Log in to Hopsworks Project
    project = hopsworks.login(
    api_key_value=HOPSWORKS_API_KEY,
    project=CONFIG["feature_store"]["project_name"]
)
    fs = project.get_feature_store()

    fg_name = CONFIG["feature_store"]["feature_group_name"]
    fg_version = CONFIG["feature_store"]["feature_group_version"]

    # 2. Get or create Feature Group
    aqi_fg = fs.get_or_create_feature_group(
        name=fg_name,
        version=fg_version,
        primary_key=["location", "event_timestamp"],
        event_time="event_timestamp",
        description=f"Air Quality features and target variables for {CONFIG['location']['name']}",
        online_enabled=True,
        time_travel_format="HUDI"
    )

    # 3. Upload feature data frame
    aqi_fg.insert(df, write_options={"wait_for_job": False})
    print(f"Successfully inserted {len(df)} records into Feature Group '{fg_name}' (v{fg_version}).")

if __name__ == "__main__":
    from src.data.api_client import fetch_raw_air_quality
    from src.features.build_features import engineer_features

    print("1. Fetching raw AQICN data...")
    raw_df = fetch_raw_air_quality()

    print("2. Engineering features...")
    processed_df = engineer_features(raw_df)

    print("3. Uploading to Hopsworks Feature Store...")
    push_to_feature_store(processed_df)