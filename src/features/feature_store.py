import hopsworks
import pandas as pd
from src.config import CONFIG, HOPSWORKS_API_KEY
from src.utils.logger import logger


def _check_schema_compatibility(feature_group, df: pd.DataFrame) -> None:
    """
    Compares the DataFrame's columns against the feature group's existing
    schema *before* calling insert(), and raises a clear, actionable error
    if they don't match — instead of letting a raw hsfs FeatureStoreException
    surface deep in a stack trace during CI.

    Skipped for a brand-new feature group (no features registered yet), since
    in that case the first insert() call is what defines the schema.
    """
    existing_features = getattr(feature_group, "features", None)
    if not existing_features:
        # Newly created feature group — nothing to compare against yet.
        return

    # Hopsworks lower-cases feature names and replaces spaces with
    # underscores, so normalize both sides the same way before comparing.
    existing_cols = {f.name.lower() for f in existing_features}
    incoming_cols = {c.lower().replace(" ", "_") for c in df.columns}

    extra_cols = incoming_cols - existing_cols
    missing_cols = existing_cols - incoming_cols

    if extra_cols or missing_cols:
        msg_parts = [
            f"DataFrame columns don't match feature group "
            f"'{feature_group.name}' (v{feature_group.version}) schema."
        ]
        if extra_cols:
            msg_parts.append(
                f"Columns in DataFrame but NOT in feature group schema "
                f"(these will be rejected by Hopsworks): {sorted(extra_cols)}"
            )
        if missing_cols:
            msg_parts.append(
                f"Columns in feature group schema but NOT in DataFrame "
                f"(these will be inserted as null): {sorted(missing_cols)}"
            )
        msg_parts.append(
            "If this is a deliberate new feature, bump "
            "feature_group_version in config/config.yaml, re-backfill "
            "history, and update build_features.py accordingly. If it's "
            "not deliberate, drop the extra column(s) before pushing."
        )
        full_msg = "\n".join(msg_parts)
        logger.error(full_msg)
        raise ValueError(full_msg)


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

    # 2b. Fail fast with a clear message if the DataFrame doesn't match the
    # feature group's existing schema, instead of surfacing a raw
    # FeatureStoreException from deep inside hsfs.
    _check_schema_compatibility(aqi_fg, df)

    # 3. Upload feature data frame
    aqi_fg.insert(df, write_options={"wait_for_job": True})
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