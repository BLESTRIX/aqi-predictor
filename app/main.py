import sys
import os
import json
from datetime import datetime
import streamlit as st
import pandas as pd
import plotly.express as px

# Adjust sys.path to allow importing from the src package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.predict import run_inference

# Page Config
st.set_page_config(page_title="Islamabad AQI Predictor", layout="wide", page_icon="🌤️")

# Per-horizon confidence, based on measured R² from model comparison
# (24h ≈ 0.80, 48h ≈ 0.60, 72h ≈ 0.46-0.50 across RF/XGBoost/Ridge)
HORIZON_CONFIDENCE = {
    "24h": {"r2": 0.80, "label": "High confidence"},
    "48h": {"r2": 0.60, "label": "Moderate confidence"},
    "72h": {"r2": 0.48, "label": "Lower confidence"},
}

HAZARDOUS_THRESHOLD = 200  # "Very Unhealthy" and above triggers an alert banner


def get_aqi_category(aqi_value):
    """Return category name, CSS color, and health recommendation based on US-AQI standard."""
    if aqi_value is None or pd.isna(aqi_value):
        return "Unknown", "gray", "N/A"

    val = float(aqi_value)
    if val <= 50:
        return "Good", "green", "Air quality is satisfactory, and air pollution poses little or no risk."
    elif val <= 100:
        return "Moderate", "gold", "Air quality is acceptable. However, there may be a risk for some people, particularly those who are unusually sensitive to air pollution."
    elif val <= 150:
        return "Unhealthy for Sensitive Groups", "orange", "Members of sensitive groups may experience health effects. The general public is less likely to be affected."
    elif val <= 200:
        return "Unhealthy", "red", "Some members of the general public may experience health effects; members of sensitive groups may experience more serious health effects."
    elif val <= 300:
        return "Very Unhealthy", "purple", "Health alert: The risk of health effects is increased for everyone."
    else:
        return "Hazardous", "maroon", "Health warning of emergency conditions: everyone is more likely to be affected."


def render_hazard_alerts(current_aqi, forecast_24h, forecast_48h, forecast_72h):
    """Displays a prominent alert banner if current or any forecasted AQI crosses the hazardous threshold."""
    checks = [
        ("Current", current_aqi),
        ("24h forecast", forecast_24h),
        ("48h forecast", forecast_48h),
        ("72h forecast", forecast_72h),
    ]

    triggered = [(label, val) for label, val in checks if val is not None and val >= HAZARDOUS_THRESHOLD]

    if triggered:
        lines = "\n".join(
            f"- **{label}:** AQI {val:.1f} ({get_aqi_category(val)[0]})" for label, val in triggered
        )
        st.error(
            f"🚨 **Hazardous Air Quality Alert**\n\n"
            f"The following readings are at or above {HAZARDOUS_THRESHOLD} US-AQI "
            f"(Very Unhealthy or worse):\n\n{lines}\n\n"
            f"Limit outdoor exposure and consider wearing an N95 mask if you must go outside."
        )


# Header Section
st.title("🌤️ Islamabad AQI Predictor")
st.markdown("Real-time air quality forecasts for the next 72 hours powered by Machine Learning and Hopsworks Feature Store.")

if st.button("🔄 Interactive Refresh"):
    st.rerun()

st.divider()

# Main Dashboard Logic
try:
    with st.spinner("Fetching latest feature vectors and generating predictions..."):
        predictions = run_inference()

    current_aqi = predictions.get("current_aqi")
    forecast_24h = predictions.get("forecast_24h")
    forecast_48h = predictions.get("forecast_48h")
    forecast_72h = predictions.get("forecast_72h")
    model_version = predictions.get("model_version")

    # Hazardous AQI alert banner — shown first so it's impossible to miss
    render_hazard_alerts(current_aqi, forecast_24h, forecast_48h, forecast_72h)

    # AQI Health Status Cards
    st.subheader("Current Air Quality")

    cat_name, cat_color, cat_desc = get_aqi_category(current_aqi)

    col1, col2 = st.columns([1, 4])
    with col1:
        st.metric(
            label="Current US-AQI",
            value=f"{current_aqi:.1f}" if current_aqi is not None else "N/A"
        )
    with col2:
        st.markdown(f"**Health Status:** <span style='color:{cat_color}; font-weight:bold; font-size: 1.1em;'>{cat_name}</span>", unsafe_allow_html=True)
        st.info(f"**Recommendation:** {cat_desc}")

    st.divider()

    # Forecast Trends Chart
    st.subheader("Forecast Trends (Next 72 Hours)")

    time_labels = ["Current", "24 Hours", "48 Hours", "72 Hours"]
    aqi_values = [current_aqi, forecast_24h, forecast_48h, forecast_72h]

    df_plot = pd.DataFrame({
        "Timeline": time_labels,
        "Predicted AQI": aqi_values
    })

    fig = px.line(
        df_plot,
        x="Timeline",
        y="Predicted AQI",
        markers=True,
        title="Air Quality Forecast",
        text=[f"{val:.1f}" if val is not None else "" for val in aqi_values]
    )

    fig.update_traces(
        textposition="top center",
        line=dict(width=4, color="#1f77b4"),
        marker=dict(size=12, color="#1f77b4")
    )

    # Warning threshold lines
    fig.add_hline(y=100, line_dash="dash", line_color="orange", annotation_text="Unhealthy for Sensitive Groups (>100)", annotation_position="top left")
    fig.add_hline(y=150, line_dash="dash", line_color="red", annotation_text="Unhealthy (>150)", annotation_position="top left")
    fig.add_hline(y=200, line_dash="dash", line_color="purple", annotation_text="Very Unhealthy (>200)", annotation_position="top left")

    fig.update_layout(
        yaxis_title="US-AQI Value",
        xaxis_title="Forecast Window",
        yaxis=dict(range=[0, max([val for val in aqi_values if val is not None] + [200]) + 20])
    )

    st.plotly_chart(fig, use_container_width=True)

    # Forecast Breakdown Table (with per-horizon confidence)
    st.subheader("Detailed Forecast Breakdown")

    breakdown_data = []
    horizon_keys = ["24h", "48h", "72h"]
    for label, key, val in zip(time_labels[1:], horizon_keys, [forecast_24h, forecast_48h, forecast_72h]):
        c_name, _, c_desc = get_aqi_category(val)
        confidence = HORIZON_CONFIDENCE[key]
        breakdown_data.append({
            "Forecast Window": label,
            "Predicted US-AQI": round(val, 1) if val is not None else "N/A",
            "Category": c_name,
            "Model Confidence": f"{confidence['label']} (R²≈{confidence['r2']:.2f})",
            "Recommendation": c_desc
        })
        st.divider()
    st.subheader("🔍 Why This Forecast (Feature Importance)")

    shap_explanations = predictions.get("shap_explanations", {})
    horizon_map = {
        "target_aqi_24h": "24 Hours",
        "target_aqi_48h": "48 Hours",
        "target_aqi_72h": "72 Hours",
    }

    if not shap_explanations or all(not v for v in shap_explanations.values()):
        st.info("Feature importance data is unavailable for this forecast.")
    else:
        tabs = st.tabs([horizon_map.get(k, k) for k in shap_explanations.keys()])
        for tab, (horizon_key, contributions) in zip(tabs, shap_explanations.items()):
            with tab:
                if not contributions:
                    st.write("No explanation available for this horizon.")
                    continue

                shap_df = pd.DataFrame(contributions)
                shap_df["direction"] = shap_df["shap_value"].apply(
                    lambda v: "Increases AQI" if v > 0 else "Decreases AQI"
                )
                shap_df["abs_value"] = shap_df["shap_value"].abs()

                fig_shap = px.bar(
                    shap_df.sort_values("abs_value"),
                    x="shap_value",
                    y="feature",
                    orientation="h",
                    color="direction",
                    color_discrete_map={"Increases AQI": "#d62728", "Decreases AQI": "#2ca02c"},
                    title=f"Top factors influencing the {horizon_map.get(horizon_key, horizon_key)} forecast",
                )
                fig_shap.update_layout(yaxis_title="", xaxis_title="SHAP value (impact on prediction)")
                st.plotly_chart(fig_shap, use_container_width=True)

        st.caption(
            "Positive values push the predicted AQI higher; negative values pull it lower. "
            "Based on SHAP values computed against a sample of recent historical readings."
        )

    df_breakdown = pd.DataFrame(breakdown_data)
    st.dataframe(df_breakdown, use_container_width=True, hide_index=True)

    st.caption(
        "⚠️ Confidence estimates are based on historical model evaluation. "
        "72-hour forecasts are meaningfully less reliable than 24-hour forecasts — "
        "treat longer-horizon predictions as directional, not precise."
    )

    st.divider()
    footer_col1, footer_col2 = st.columns(2)
    with footer_col1:
        st.caption(f"Last updated: {predictions.get('execution_time')}")
    with footer_col2:
        st.caption(f"Model version: {model_version if model_version is not None else 'N/A'}")

except Exception as e:
    st.error("⚠️ Failed to load predictions or connect to Hopsworks.")
    st.exception(e)

    st.markdown("""
    **Troubleshooting Steps:**
    1. Ensure `HOPSWORKS_API_KEY` is set correctly in your `.env` file.
    2. Verify the model `islamabad_aqi_model_24h` exists in the Hopsworks Model Registry.
    3. Ensure the feature group `islamabad_aqi_features` has been backfilled with data.
    4. Check your internet connection.
    """)