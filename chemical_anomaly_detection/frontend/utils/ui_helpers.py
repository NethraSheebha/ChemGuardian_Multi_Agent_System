import streamlit as st

def get_status_color(status):
    """Returns color based on status."""
    colors = {
        "NORMAL": "green",
        "MILD": "yellow",
        "MEDIUM": "orange",
        "HIGH": "red"
    }
    return colors.get(status, "blue")

def style_anomaly(anomaly):
    """Returns CSS style for anomaly."""
    return "background-color: red;" if anomaly else ""

def render_status_indicator(status):
    """Renders a status indicator with emoji and color."""
    emojis = {
        "NORMAL": "🟢",
        "MILD": "🟡",
        "MEDIUM": "🟠",
        "HIGH": "🔴"
    }
    color = get_status_color(status)
    st.markdown(f"<h2 style='color:{color};'>{emojis.get(status, '🟢')} {status}</h2>", unsafe_allow_html=True)

def render_trend_arrow(trend):
    """Renders trend arrow."""
    arrows = {"UP": "↑", "DOWN": "↓", "STABLE": "→"}
    return arrows.get(trend, "→")

@st.cache_data
def cached_plotly_fig(fig):
    """Caches Plotly figures for performance."""
    return fig