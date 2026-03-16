import streamlit as st
import plotly.graph_objects as go
from utils.ui_helpers import cached_plotly_fig

def render(audio_data):
    """Renders the enhanced audio monitoring panel."""
    
    if not audio_data:
        st.info("🔊 Waiting for audio data...")
        return
    
    # Header with status indicator
    anomaly = audio_data.get("anomaly", False)
    status_color = "🔴" if anomaly else "🟢"
    status_text = "ANOMALY DETECTED" if anomaly else "NORMAL"
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"### 🔊 Audio Analysis")
    with col2:
        if anomaly:
            st.markdown(f"<div style='background-color: rgba(255, 68, 68, 0.2); color: #ff4444; padding: 8px; border-radius: 5px; text-align: center; font-weight: bold; border: 2px solid #ff4444; font-size: 14px;'>{status_color} {status_text}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='background-color: rgba(68, 255, 68, 0.2); color: #228822; padding: 8px; border-radius: 5px; text-align: center; font-weight: bold; border: 2px solid #44ff44; font-size: 14px;'>{status_color} {status_text}</div>", unsafe_allow_html=True)
    
    # Waveform visualization
    waveform = audio_data.get("waveform", [])
    if waveform:
        color = "#ff4444" if anomaly else "#4444ff"
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=waveform, 
            mode='lines', 
            line=dict(color=color, width=1.5),
            fill='tozeroy',
            fillcolor=f'rgba({255 if anomaly else 68}, {68}, {255 if not anomaly else 68}, 0.2)'
        ))
        fig.update_layout(
            title="Audio Waveform",
            xaxis_title="Sample",
            yaxis_title="Amplitude",
            height=250,
            margin=dict(l=40, r=40, t=40, b=40),
            plot_bgcolor='rgba(240,240,240,0.5)'
        )
        
        # Add subtle border for anomalies
        if anomaly:
            st.markdown("<div style='border: 2px solid #ff4444; padding: 3px; border-radius: 5px;'>", unsafe_allow_html=True)
            st.plotly_chart(cached_plotly_fig(fig), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.plotly_chart(cached_plotly_fig(fig), use_container_width=True)
    
    # Audio metrics in organized layout
    st.markdown("---")
    metrics = audio_data.get("metrics", {})
    
    col1, col2, col3 = st.columns(3)
    with col1:
        peak = metrics.get('peak', 0.0)
        st.metric("📊 Peak Amplitude", f"{peak:.4f}")
    with col2:
        rms = metrics.get('rms', 0.0)
        st.metric("📈 RMS Level", f"{rms:.4f}")
    with col3:
        zcr = metrics.get('zcr', 0.0)
        st.metric("🌊 Zero Crossing Rate", f"{zcr:.2f}")