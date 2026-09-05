"""
Fetches historical air quality data for Islamabad from Open-Meteo (free, no API key needed).
Saves to data/islamabad_aqicn_historical.csv in the format expected by backfill.py.

Usage:
    python -m src.data.fetch_historical
"""

import os
import requests
import pandas as pd
from datetime import datetime, timedelta

# ── Config ────────────────────────────────────────────────────────────────────
LATITUDE   = 33.6844
LONGITUDE  = 73.0479
DAYS_BACK  = 730
OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "islamabad_aqicn_historical.csv")

OPENMETEO_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"


def fetch_openmeteo_history(lat: float, lon: float, days_back: int) -> pd.DataFrame:
    """Fetches hourly historical air quality data from Open-Meteo."""
    end_date   = datetime.utcnow().date()
    start_date = end_date - timedelta(days=days_back)

    params = {
        "latitude":  lat,
        "longitude": lon,
        "hourly": ",".join([
            "pm2_5",
            "pm10",
            "nitrogen_dioxide",
            "sulphur_dioxide",
            "ozone",
            "carbon_monoxide",
            "us_aqi",
        ]),
        "start_date": str(start_date),
        "end_date":   str(end_date),
        "timezone":   "UTC",
    }

    print(f"  Requesting {start_date} → {end_date} from Open-Meteo...")
    response = requests.get(OPENMETEO_URL, params=params, timeout=60)
    response.raise_for_status()

    data   = response.json()
    hourly = data.get("hourly", {})

    df = pd.DataFrame({
        "time":              pd.to_datetime(hourly["time"]),
        "us_aqi":            hourly.get("us_aqi",            [0.0] * len(hourly["time"])),
        "pm2_5":             hourly.get("pm2_5",             [0.0] * len(hourly["time"])),
        "pm10":              hourly.get("pm10",               [0.0] * len(hourly["time"])),
        "nitrogen_dioxide":  hourly.get("nitrogen_dioxide",  [0.0] * len(hourly["time"])),
        "sulphur_dioxide":   hourly.get("sulphur_dioxide",   [0.0] * len(hourly["time"])),
        "ozone":             hourly.get("ozone",              [0.0] * len(hourly["time"])),
        "carbon_monoxide":   hourly.get("carbon_monoxide",   [0.0] * len(hourly["time"])),
    })

    # Add placeholder weather columns (not provided by air quality API)
    df["temperature"] = 0.0
    df["humidity"]    = 0.0
    df["pressure"]    = 0.0
    df["wind_speed"]  = 0.0

    # Fill NaNs with 0 and drop rows where all pollutants are 0
    df = df.fillna(0.0)
    df = df[df["us_aqi"] > 0].reset_index(drop=True)

    return df


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Fetching {DAYS_BACK} days of Islamabad air quality data from Open-Meteo...")

    df = fetch_openmeteo_history(LATITUDE, LONGITUDE, DAYS_BACK)

    if df.empty:
        print("No data returned. Check your internet connection.")
        return

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nDone. Saved {len(df)} hourly rows → {OUTPUT_FILE}")
    print(f"\nSample data:")
    print(df[["time", "us_aqi", "pm2_5", "pm10", "ozone"]].head(10).to_string(index=False))
    print(f"\nAQI range: min={df['us_aqi'].min():.1f}  max={df['us_aqi'].max():.1f}  mean={df['us_aqi'].mean():.1f}")


if __name__ == "__main__":
    main()