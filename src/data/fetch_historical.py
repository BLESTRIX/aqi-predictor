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
DAYS_BACK  = 3650
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

    # ── Verification: confirm what we actually got vs. what we asked for ──
    actual_start = df["time"].min()
    actual_end = df["time"].max()
    actual_days = (actual_end - actual_start).days
    requested_start = (datetime.utcnow().date() - timedelta(days=DAYS_BACK))

    print(f"\nDone. Saved {len(df)} hourly rows → {OUTPUT_FILE}")
    print(f"\n--- Coverage check ---")
    print(f"Requested start : {requested_start}")
    print(f"Actual start    : {actual_start.date()}")
    print(f"Actual end      : {actual_end.date()}")
    print(f"Actual span     : {actual_days} days (~{actual_days/365.25:.1f} years)")
    if actual_start.date() > requested_start:
        gap_days = (actual_start.date() - requested_start).days
        print(f"⚠️  API clipped your request by {gap_days} days — "
              f"Open-Meteo's archive doesn't go back as far as requested.")

    print(f"\nSample data:")
    print(df[["time", "us_aqi", "pm2_5", "pm10", "ozone"]].head(10).to_string(index=False))

    print(f"\n--- AQI distribution ---")
    print(f"min={df['us_aqi'].min():.1f}  max={df['us_aqi'].max():.1f}  mean={df['us_aqi'].mean():.1f}")

    # Breakdown by AQI category so you can see if hazardous levels are represented
    bins = [0, 50, 100, 150, 200, 300, float("inf")]
    labels = ["Good", "Moderate", "Unhealthy(Sensitive)", "Unhealthy", "Very Unhealthy", "Hazardous"]
    category_counts = pd.cut(df["us_aqi"], bins=bins, labels=labels).value_counts().sort_index()
    print("\nRows per AQI category:")
    print(category_counts.to_string())
    pct_hazardous_plus = (df["us_aqi"] > 200).mean() * 100
    print(f"\n{pct_hazardous_plus:.1f}% of rows are Very Unhealthy or worse (>200)")

if __name__ == "__main__":
    main()