import os
import requests
import pandas as pd
from typing import Dict, Any
from src.config import CONFIG, AQICN_API_KEY


def parse_aqicn_payload(payload: Dict[str, Any]) -> pd.DataFrame:
    """
    Parses AQICN JSON payload, extracting real-time pollutant/weather
    metrics and joining daily forecast arrays across target pollutants.
    """
    if payload.get("status") != "ok":
        raise ValueError(f"AQICN API returned error: {payload.get('data')}")

    data = payload["data"]

    # 1. Parse real-time snapshot measurement
    time_str = data.get("time", {}).get("s")
    current_time = pd.to_datetime(time_str) if time_str else pd.Timestamp.now()

    iaqi = data.get("iaqi", {})
    current_row = {
        "time": current_time,
        "us_aqi": float(data.get("aqi", 0)),
        "pm2_5": float(iaqi.get("pm25", {}).get("v", 0.0)),
        "pm10": float(iaqi.get("pm10", {}).get("v", 0.0)),
        "nitrogen_dioxide": float(iaqi.get("no2", {}).get("v", 0.0)),
        "sulphur_dioxide": float(iaqi.get("so2", {}).get("v", 0.0)),
        "carbon_monoxide": float(iaqi.get("co", {}).get("v", 0.0)),
        "ozone": float(iaqi.get("o3", {}).get("v", 0.0)),
        # Weather features provided by AQICN station
        "temperature": float(iaqi.get("t", {}).get("v", 0.0)),
        "humidity": float(iaqi.get("h", {}).get("v", 0.0)),
        "pressure": float(iaqi.get("p", {}).get("v", 0.0)),
        "wind_speed": float(iaqi.get("w", {}).get("v", 0.0)),
        "record_type": "realtime",
    }

    # 2. Parse daily forecast series (PM2.5, PM10, O3)
    daily_forecasts = data.get("forecast", {}).get("daily", {})
    pm25_list = daily_forecasts.get("pm25", [])
    pm10_map = {
        item["day"]: item["avg"] for item in daily_forecasts.get("pm10", [])
    }
    o3_map = {
        item["day"]: item["avg"] for item in daily_forecasts.get("o3", [])
    }
    # NOTE: AQICN's forecast payload also carries a daily "max" alongside "avg"
    # for pm25, which would be useful for a future pm25_daily_max feature.
    # It is intentionally NOT added to forecast_rows below: the Hopsworks
    # feature group schema (islamabad_aqi_features v2) was created from
    # backfill.py's historical loader, which never produced this column, so
    # inserting it here causes a hard schema-mismatch error at push time
    # ("Features are not compatible with Feature Group schema"). To add this
    # feature properly: (1) add it to build_features.py so backfill.py also
    # produces it, (2) bump feature_group_version in config.yaml so Hopsworks
    # creates a new group with the extra column, (3) re-backfill history,
    # then re-add pm2_5_max here.

    forecast_rows = []
    for item in pm25_list:
        day_str = item["day"]
        forecast_rows.append({
            "time": pd.to_datetime(day_str),
            "us_aqi": float(item.get("avg", 0.0)),
            "pm2_5": float(item.get("avg", 0.0)),
            "pm10": float(pm10_map.get(day_str, 0.0)),
            "nitrogen_dioxide": 0.0,
            "sulphur_dioxide": 0.0,
            "carbon_monoxide": 0.0,
            "ozone": float(o3_map.get(day_str, 0.0)),
            "temperature": 0.0,
            "humidity": 0.0,
            "pressure": 0.0,
            "wind_speed": 0.0,
            "record_type": "forecast",
        })

    # Combine real-time observation with forecast series
    df = pd.DataFrame([current_row] + forecast_rows)
    return df.sort_values("time").reset_index(drop=True)


def fetch_raw_air_quality(station: str = None, lat: float = None, lon: float = None) -> pd.DataFrame:
    """
    Queries the AQICN API. Defaults to the pinned station (config.api.station_path)
    so live fetches hit the SAME physical sensor as the historical training data
    (Islamabad US Embassy). Falls back to geo-coordinates only if explicitly passed.
    """
    if not AQICN_API_KEY:
        raise ValueError("AQICN_API_KEY is not set in environment or .env file.")

    if lat is not None and lon is not None:
        path = f"geo:{lat};{lon}"
    else:
        path = station or CONFIG["api"].get("station_path", "pakistan/islamabad/us-embassy")

    url = f"{CONFIG['api']['base_url']}/{path}/?token={AQICN_API_KEY}"

    response = requests.get(url)
    response.raise_for_status()
    return parse_aqicn_payload(response.json())


if __name__ == "__main__":
    df = fetch_raw_air_quality()
    print(f"Successfully processed AQICN payload for {CONFIG['location']['name']}:")
    print(df[["time", "us_aqi", "pm2_5", "temperature", "record_type"]].head(10))