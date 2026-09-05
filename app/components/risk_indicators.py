from typing import Dict, Any


def get_aqi_category(aqi_value: float) -> Dict[str, Any]:
    """Returns EPA AQI category, color code, status badge, and health recommendation."""
    val = float(aqi_value) if aqi_value is not None else 0.0

    if val <= 50:
        return {
            "level": "Good",
            "color": "#00E400",
            "text_color": "#000000",
            "badge": "🟢 GOOD",
            "advice": "Air quality is satisfactory, and air pollution poses little or no risk."
        }
    elif val <= 100:
        return {
            "level": "Moderate",
            "color": "#FFFF00",
            "text_color": "#000000",
            "badge": "🟡 MODERATE",
            "advice": "Air quality is acceptable; sensitive individuals should consider limiting prolonged outdoor exertion."
        }
    elif val <= 150:
        return {
            "level": "Unhealthy for Sensitive Groups",
            "color": "#FF7E00",
            "text_color": "#FFFFFF",
            "badge": "🟠 UNHEALTHY FOR SENSITIVE GROUPS",
            "advice": "Members of sensitive groups (children, elderly, asthmatics) may experience health effects."
        }
    elif val <= 200:
        return {
            "level": "Unhealthy",
            "color": "#FF0000",
            "text_color": "#FFFFFF",
            "badge": "🔴 UNHEALTHY",
            "advice": "Everyone may begin to experience health effects; sensitive groups may experience more serious effects."
        }
    elif val <= 300:
        return {
            "level": "Very Unhealthy",
            "color": "#8F3F97",
            "text_color": "#FFFFFF",
            "badge": "🟣 VERY UNHEALTHY",
            "advice": "Health alert: everyone may experience more serious health effects. Avoid outdoor activities."
        }
    else:
        return {
            "level": "Hazardous",
            "color": "#7E0023",
            "text_color": "#FFFFFF",
            "badge": "ALERT: HAZARDOUS",
            "advice": "Health warnings of emergency conditions. The entire population is more likely to be affected."
        }
