import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from app.components.risk_indicators import get_aqi_category


def create_aqi_gauge(aqi_value: float) -> go.Figure:
    """Generates an interactive gauge chart for current AQI index."""
    cat = get_aqi_category(aqi_value)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=aqi_value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Current AQI ({cat['level']})", 'font': {'size': 20}},
        gauge={
            'axis': {'range': [None, 350], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': cat['color']},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': 'rgba(0, 228, 0, 0.3)'},
                {'range': [50, 100], 'color': 'rgba(255, 255, 0, 0.3)'},
                {'range': [100, 150], 'color': 'rgba(255, 126, 0, 0.3)'},
                {'range': [150, 200], 'color': 'rgba(255, 0, 0, 0.3)'},
                {'range': [200, 300], 'color': 'rgba(143, 63, 151, 0.3)'},
                {'range': [300, 350], 'color': 'rgba(126, 0, 35, 0.3)'},
            ],
        }
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def create_forecast_chart(df_forecast: pd.DataFrame) -> go.Figure:
    """Generates time-series line chart for historical and forecasted AQI values."""
    fig = go.Figure()

    if "timestamp" in df_forecast.columns:
        x_val = df_forecast["timestamp"]
    else:
        x_val = df_forecast.index

    if "calculated_aqi" in df_forecast.columns:
        fig.add_trace(go.Scatter(
            x=x_val,
            y=df_forecast["calculated_aqi"],
            mode="lines+markers",
            name="Historical / Live AQI",
            line=dict(color="#1f77b4", width=3)
        ))

    if "predicted_aqi" in df_forecast.columns:
        fig.add_trace(go.Scatter(
            x=x_val,
            y=df_forecast["predicted_aqi"],
            mode="lines+markers",
            name="Model Prediction (XGBoost)",
            line=dict(color="#ff7f0e", width=3, dash="dash")
        ))

    fig.update_layout(
        title="24-48 Hour Air Quality Index Forecast",
        xaxis_title="Time",
        yaxis_title="AQI Sub-Index",
        template="plotly_white",
        legend=dict(x=0.01, y=0.99),
        height=380,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig


def create_pollutants_breakdown_chart(df_latest: pd.Series) -> go.Figure:
    """Generates bar chart showing concentrations of key air pollutants."""
    pollutant_cols = {
        "PM2.5": df_latest.get("pm2_5", 0),
        "PM10": df_latest.get("pm10", 0),
        "NO2": df_latest.get("nitrogen_dioxide", 0),
        "SO2": df_latest.get("sulphur_dioxide", 0),
        "O3": df_latest.get("ozone", 0),
        "CO": df_latest.get("carbon_monoxide", 0) / 100.0 if df_latest.get("carbon_monoxide") else 0
    }

    df_bar = pd.DataFrame({
        "Pollutant": list(pollutant_cols.keys()),
        "Concentration (μg/m³)": list(pollutant_cols.values())
    })

    fig = px.bar(
        df_bar,
        x="Pollutant",
        y="Concentration (μg/m³)",
        color="Pollutant",
        title="Current Pollutant Concentrations",
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    fig.update_layout(template="plotly_white", height=320, margin=dict(l=20, r=20, t=50, b=20))
    return fig
