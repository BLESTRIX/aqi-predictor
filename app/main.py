import sys
import os
from datetime import datetime, timezone
import streamlit as st
import pandas as pd
import plotly.express as px

# Adjust sys.path to allow importing from the src package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.predict import run_inference
from src.data.api_client import fetch_raw_air_quality
from app.components.risk_indicators import get_aqi_category
from app.components.plots import (
    create_aqi_gauge,
    create_pollutants_breakdown_chart,
)

# ── Page Config ───────────────────────────────────────────────────────────
st.set_page_config(page_title="Islamabad AQI Predictor", layout="wide", page_icon="🌤️")

# Per-horizon confidence, based on measured R² from model comparison
# (see notebooks/AQI_Model_Comparison.ipynb: Ridge won every horizon there,
# consistently ahead of RandomForest/XGBoost — 24h ≈ 0.87, 48h ≈ 0.80, 72h ≈ 0.75)
HORIZON_CONFIDENCE = {
    "24h": {"r2": 0.87, "label": "High confidence"},
    "48h": {"r2": 0.80, "label": "Moderate confidence"},
    "72h": {"r2": 0.75, "label": "Lower confidence"},
}

HAZARDOUS_THRESHOLD = 200  # "Very Unhealthy" and above triggers an alert banner


# ── Styling ───────────────────────────────────────────────────────────────
def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        #MainMenu, footer, header {visibility: hidden;}

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }

        .app-title {
            font-size: 2.1rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            margin-bottom: 0.15rem;
            color: #0F172A;
        }
        .app-subtitle {
            color: #64748B;
            font-size: 0.98rem;
            margin-bottom: 1.4rem;
        }

        .hero-card {
            border-radius: 20px;
            padding: 28px 32px;
            margin-bottom: 22px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 18px;
        }
        .hero-aqi-value {
            font-size: 3.6rem;
            font-weight: 800;
            line-height: 1;
        }
        .hero-aqi-label {
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            opacity: 0.85;
        }
        .hero-badge {
            display: inline-block;
            padding: 6px 16px;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.95rem;
            background: rgba(255,255,255,0.28);
        }
        .hero-meta {
            font-size: 0.82rem;
            opacity: 0.85;
            margin-top: 6px;
        }

        .forecast-card {
            border-radius: 16px;
            padding: 18px 20px;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
            border-left: 6px solid;
            background: #FFFFFF;
            height: 100%;
        }
        .forecast-window {
            font-size: 0.82rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #64748B;
            margin-bottom: 4px;
        }
        .forecast-value {
            font-size: 2.1rem;
            font-weight: 800;
            color: #0F172A;
            line-height: 1.1;
        }
        .forecast-category {
            font-weight: 700;
            font-size: 0.95rem;
            margin-top: 4px;
        }
        .forecast-confidence {
            font-size: 0.78rem;
            color: #94A3B8;
            margin-top: 8px;
        }

        .section-title {
            font-size: 1.25rem;
            font-weight: 700;
            color: #0F172A;
            margin-top: 8px;
            margin-bottom: 0.6rem;
        }

        .live-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: #ECFDF5;
            color: #047857;
            border: 1px solid #A7F3D0;
            padding: 3px 12px;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
        }
        .live-dot {
            width: 7px; height: 7px; border-radius: 50%;
            background: #10B981;
        }
        .stale-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: #FEF3C7;
            color: #92400E;
            border: 1px solid #FDE68A;
            padding: 3px 12px;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(current_aqi, is_live, source_label):
    cat = get_aqi_category(current_aqi)
    live_html = (
        '<span class="live-pill"><span class="live-dot"></span>LIVE STATION READING</span>'
        if is_live else
        '<span class="stale-pill">FEATURE STORE SNAPSHOT</span>'
    )
    st.markdown(
        f"""
        <div class="hero-card" style="background: linear-gradient(135deg, {cat['color']}E6, {cat['color']}CC); color: {cat['text_color']};">
            <div>
                <div class="hero-aqi-label">Current US-AQI · Islamabad</div>
                <div class="hero-aqi-value">{current_aqi:.0f}</div>
                <div class="hero-meta">{live_html} &nbsp; {source_label}</div>
            </div>
            <div style="text-align:right;">
                <span class="hero-badge">{cat['badge']}</span>
                <div class="hero-meta" style="margin-top:10px; max-width: 320px; text-align:right;">{cat['advice']}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_forecast_card(col, window_label, value, confidence_key):
    cat = get_aqi_category(value) if value is not None else None
    confidence = HORIZON_CONFIDENCE[confidence_key]
    with col:
        if cat is None:
            st.markdown(
                '<div class="forecast-card" style="border-left-color:#CBD5E1;">'
                '<div class="forecast-window">' + window_label + '</div>'
                '<div class="forecast-value">N/A</div></div>',
                unsafe_allow_html=True,
            )
            return
        st.markdown(
            f"""
            <div class="forecast-card" style="border-left-color:{cat['color']};">
                <div class="forecast-window">{window_label}</div>
                <div class="forecast-value">{value:.0f}</div>
                <div class="forecast-category" style="color:{cat['color']};">{cat['level']}</div>
                <div class="forecast-confidence">{confidence['label']} · R² ≈ {confidence['r2']:.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


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
            f"- **{label}:** AQI {val:.1f} ({get_aqi_category(val)['level']})" for label, val in triggered
        )
        st.error(
            f"🚨 **Hazardous Air Quality Alert**\n\n"
            f"The following readings are at or above {HAZARDOUS_THRESHOLD} US-AQI "
            f"(Very Unhealthy or worse):\n\n{lines}\n\n"
            f"Limit outdoor exposure and consider wearing an N95 mask if you must go outside."
        )


def get_live_reading():
    """
    Fetches a fresh reading directly from AQICN, independent of the feature
    store snapshot the model was trained/served on. The feature store is
    only updated hourly and can lag; this gives an honestly "live" number
    for the hero section and pollutant breakdown, while forecasts still
    come from the model via run_inference().
    Returns None if AQICN_API_KEY isn't set or the live fetch fails for any
    reason — callers must handle that gracefully rather than crashing the
    whole dashboard over a secondary data source.
    """
    try:
        df_live = fetch_raw_air_quality()
        realtime_rows = df_live[df_live["record_type"] == "realtime"]
        if realtime_rows.empty:
            return None
        return realtime_rows.iloc[0]
    except Exception:
        return None


# ── App ───────────────────────────────────────────────────────────────────
inject_css()

st.markdown('<div class="app-title">🌤️ Islamabad AQI Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Real-time air quality readings and 72-hour ML forecasts, '
    'powered by a Hopsworks-backed feature store and model registry.</div>',
    unsafe_allow_html=True,
)

top_col1, top_col2 = st.columns([5, 1])
with top_col2:
    if st.button("🔄 Refresh", use_container_width=True):
        st.rerun()

try:
    with st.spinner("Fetching latest feature vectors and generating predictions..."):
        predictions = run_inference()

    live_row = get_live_reading()

    feature_store_aqi = predictions.get("current_aqi")
    forecast_24h = predictions.get("forecast_24h")
    forecast_48h = predictions.get("forecast_48h")
    forecast_72h = predictions.get("forecast_72h")
    model_version = predictions.get("model_version")

    # Prefer the genuinely live AQICN reading for the headline number;
    # fall back to the feature store's snapshot if the live fetch failed
    # (e.g. AQICN_API_KEY not set in this environment).
    if live_row is not None:
        current_aqi = float(live_row["us_aqi"])
        is_live = True
        source_label = f"Station reading as of {live_row['time']}"
    else:
        current_aqi = feature_store_aqi
        is_live = False
        source_label = "Live AQICN fetch unavailable — showing last feature store value"

    # Hazardous AQI alert banner — shown first so it's impossible to miss
    render_hazard_alerts(current_aqi, forecast_24h, forecast_48h, forecast_72h)

    # ── Hero: current AQI ───────────────────────────────────────────────
    render_hero(current_aqi, is_live, source_label)

    # ── Forecast cards ────────────────────────────────────────────────────
    st.markdown('<div class="section-title">72-Hour Forecast</div>', unsafe_allow_html=True)
    fc_col1, fc_col2, fc_col3 = st.columns(3)
    render_forecast_card(fc_col1, "In 24 Hours", forecast_24h, "24h")
    render_forecast_card(fc_col2, "In 48 Hours", forecast_48h, "48h")
    render_forecast_card(fc_col3, "In 72 Hours", forecast_72h, "72h")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Current AQI gauge + live pollutant breakdown ─────────────────────
    gauge_col, pollutant_col = st.columns([1, 1.4])
    with gauge_col:
        st.markdown('<div class="section-title">At a Glance</div>', unsafe_allow_html=True)
        st.plotly_chart(create_aqi_gauge(current_aqi), use_container_width=True)

    with pollutant_col:
        st.markdown('<div class="section-title">Live Pollutant Breakdown</div>', unsafe_allow_html=True)
        if live_row is not None:
            st.plotly_chart(create_pollutants_breakdown_chart(live_row), use_container_width=True)
        else:
            st.info(
                "Live pollutant breakdown needs a direct AQICN reading. "
                "Set `AQICN_API_KEY` in your environment to enable this — "
                "forecasts above still come from the feature store and are unaffected."
            )

    st.divider()

    # ── Forecast trend chart ─────────────────────────────────────────────
    st.markdown('<div class="section-title">Forecast Trend</div>', unsafe_allow_html=True)

    time_labels = ["Now", "+24h", "+48h", "+72h"]
    aqi_values = [current_aqi, forecast_24h, forecast_48h, forecast_72h]

    df_plot = pd.DataFrame({"Timeline": time_labels, "Predicted AQI": aqi_values})

    fig = px.line(
        df_plot, x="Timeline", y="Predicted AQI", markers=True,
        text=[f"{val:.0f}" if val is not None else "" for val in aqi_values],
    )
    fig.update_traces(
        textposition="top center",
        line=dict(width=4, color="#2563EB"),
        marker=dict(size=13, color="#2563EB", line=dict(width=2, color="white")),
    )

    # Shaded AQI category bands instead of plain dashed lines — easier to
    # read at a glance and uses the same EPA color scale as everywhere else.
    band_defs = [
        (0, 50, "#00E400", 0.06), (50, 100, "#FFFF00", 0.08),
        (100, 150, "#FF7E00", 0.08), (150, 200, "#FF0000", 0.08),
        (200, 300, "#8F3F97", 0.08),
    ]
    y_max = max([v for v in aqi_values if v is not None] + [200]) + 30
    for lo, hi, color, opacity in band_defs:
        if lo < y_max:
            fig.add_hrect(y0=lo, y1=min(hi, y_max), fillcolor=color, opacity=opacity, line_width=0)

    fig.update_layout(
        yaxis_title="US-AQI",
        xaxis_title=None,
        yaxis=dict(range=[0, y_max]),
        plot_bgcolor="white",
        margin=dict(l=10, r=10, t=10, b=10),
        height=340,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ── SHAP explainability ────────────────────────────────────────────
    st.markdown('<div class="section-title">🔍 Why This Forecast</div>', unsafe_allow_html=True)

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
                    x="shap_value", y="feature", orientation="h",
                    color="direction",
                    color_discrete_map={"Increases AQI": "#DC2626", "Decreases AQI": "#059669"},
                )
                fig_shap.update_layout(
                    yaxis_title="", xaxis_title="SHAP value (impact on prediction)",
                    plot_bgcolor="white", margin=dict(l=10, r=10, t=10, b=10), height=320,
                    legend_title_text="",
                )
                st.plotly_chart(fig_shap, use_container_width=True)

        st.caption(
            "Positive values push the predicted AQI higher; negative values pull it lower. "
            "Based on SHAP values computed against a sample of recent historical readings."
        )

    # ── Detailed breakdown table ──────────────────────────────────────────
    st.markdown('<div class="section-title">Detailed Breakdown</div>', unsafe_allow_html=True)
    breakdown_data = []
    horizon_keys = ["24h", "48h", "72h"]
    for label, key, val in zip(time_labels[1:], horizon_keys, [forecast_24h, forecast_48h, forecast_72h]):
        cat = get_aqi_category(val) if val is not None else None
        confidence = HORIZON_CONFIDENCE[key]
        breakdown_data.append({
            "Forecast Window": label,
            "Predicted US-AQI": round(val, 1) if val is not None else "N/A",
            "Category": cat["level"] if cat else "N/A",
            "Model Confidence": f"{confidence['label']} (R²≈{confidence['r2']:.2f})",
            "Recommendation": cat["advice"] if cat else "N/A",
        })

    df_breakdown = pd.DataFrame(breakdown_data)
    st.dataframe(df_breakdown, use_container_width=True, hide_index=True)

    st.caption(
        "⚠️ Confidence estimates are based on historical model evaluation "
        "(see notebooks/AQI_Model_Comparison.ipynb). 72-hour forecasts are "
        "meaningfully less reliable than 24-hour forecasts — treat "
        "longer-horizon predictions as directional, not precise."
    )

    st.divider()
    footer_col1, footer_col2, footer_col3 = st.columns(3)
    with footer_col1:
        st.caption(f"Model prediction run: {predictions.get('execution_time')}")
    with footer_col2:
        st.caption(f"Model version: {model_version if model_version is not None else 'N/A'}")
    with footer_col3:
        st.caption(f"Live reading: {'✅ available' if live_row is not None else '⚠️ unavailable'}")

except Exception as e:
    st.error("⚠️ Failed to load predictions or connect to Hopsworks.")
    st.exception(e)

    st.markdown("""
    **Troubleshooting Steps:**
    1. Ensure `HOPSWORKS_API_KEY` is set correctly in your `.env` file.
    2. Verify the model `islamabad_aqi_model_24h` exists in the Hopsworks Model Registry.
    3. Ensure the feature group `islamabad_aqi_features` has been backfilled with data.
    4. Set `AQICN_API_KEY` for live station readings and pollutant breakdown.
    5. Check your internet connection.
    """)