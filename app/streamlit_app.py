import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.config import settings
from src.data.api_client import OpenMeteoAPIClient
from src.features.build_features import engineer_all_features
from src.models.registry import HopsworksModelRegistry
from app.components.risk_indicators import get_aqi_category
from app.components.plots import create_aqi_gauge, create_forecast_chart, create_pollutants_breakdown_chart

# Page Configuration
st.set_page_config(
    page_title="AQI Predictor Dashboard 🌍💨",
    page_icon="💨",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_data(ttl=1800)
def load_live_data(lat: float, lon: float):
    """Fetches and processes live air quality and weather data from Open-Meteo."""
    client = OpenMeteoAPIClient(latitude=lat, longitude=lon)
    df_raw = client.fetch_combined_data(past_days=3, forecast_days=2)
    if df_raw.empty:
        return pd.DataFrame()
    df_feat = engineer_all_features(df_raw)
    return df_feat


@st.cache_resource
def load_trained_model():
    """Loads trained prediction model from registry or local storage."""
    registry = HopsworksModelRegistry()
    model = registry.load_latest_model()
    return model


def main():
    st.title("🌍 Air Quality Index (AQI) Forecast Dashboard")
    st.markdown("Real-time air pollution metrics, 24-hour ML predictions, and health risk advisories powered by **Hopsworks** & **Open-Meteo**.")

    # Sidebar Parameters
    st.sidebar.header("📍 Location & Settings")
    city = st.sidebar.text_input("City Name", value=settings.location.city)
    lat = st.sidebar.number_input("Latitude", value=settings.location.latitude, format="%.4f")
    lon = st.sidebar.number_input("Longitude", value=settings.location.longitude, format="%.4f")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Pipeline Controls")
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    # Load Data
    with st.spinner(f"Fetching air quality metrics for {city}..."):
        df = load_live_data(lat, lon)

    if df.empty:
        st.error("Failed to load air quality data. Please check your network connection or coordinates.")
        return

    # Extract Latest Metrics
    latest_row = df.iloc[-1]
    current_pm25 = latest_row.get("pm2_5", 0.0)
    current_aqi = latest_row.get("calculated_aqi", 0.0)
    category = get_aqi_category(current_aqi)

    # Top Row Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("City", city)
    with col2:
        st.metric("PM2.5 Level", f"{current_pm25:.1f} μg/m³")
    with col3:
        st.metric("Current AQI", f"{int(current_aqi)}")
    with col4:
        st.markdown(f"**Health Risk:**")
        st.markdown(
            f"<div style='background-color:{category['color']}; color:{category['text_color']}; "
            f"padding:8px 12px; border-radius:5px; font-weight:bold; text-align:center;'>"
            f"{category['badge']}</div>",
            unsafe_allow_html=True
        )

    st.markdown("---")

    # Layout Columns: Gauge & Health Advisory | Pollutant Breakdown
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.plotly_chart(create_aqi_gauge(current_aqi), use_container_width=True)
        st.info(f"💡 **Health Advisory:** {category['advice']}")

    with col_right:
        st.plotly_chart(create_pollutants_breakdown_chart(latest_row), use_container_width=True)

    st.markdown("---")

    # Machine Learning Forecast Section
    st.subheader("🔮 24-48 Hour Machine Learning Forecast")

    model = load_trained_model()
    if model is not None:
        try:
            drop_cols = ["timestamp", "timestamp_ms", "pm2_5"]
            feature_cols = [c for c in df.columns if c not in drop_cols]
            X_pred = df[feature_cols].fillna(0)

            df["predicted_aqi"] = model.predict(X_pred)
            st.plotly_chart(create_forecast_chart(df), use_container_width=True)
        except Exception as e:
            st.warning(f"Could not generate model forecast: {e}")
            st.plotly_chart(create_forecast_chart(df), use_container_width=True)
    else:
        st.warning("⚠️ No trained ML model found in Hopsworks Model Registry or local storage. Showing historical trend.")
        st.plotly_chart(create_forecast_chart(df), use_container_width=True)

    # Data Table View
    with st.expander("📋 View Raw Data Table"):
        st.dataframe(df.tail(24), use_container_width=True)


if __name__ == "__main__":
    main()
