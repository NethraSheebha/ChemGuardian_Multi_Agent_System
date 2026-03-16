import streamlit as st
import pandas as pd
from utils.ui_helpers import style_anomaly, render_trend_arrow

def render(sensor_data):
    """Renders the enhanced sensor monitoring panel."""
    
    if not sensor_data:
        st.info("📡 Waiting for sensor data...")
        return
    
    # Header with status indicator
    anomaly = sensor_data.get("anomaly", False)
    status_color = "🔴" if anomaly else "🟢"
    status_text = "ANOMALY DETECTED" if anomaly else "NORMAL"
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"### 📡 Sensor Readings")
    with col2:
        if anomaly:
            st.markdown(f"<div style='background-color: rgba(255, 68, 68, 0.2); color: #ff4444; padding: 8px; border-radius: 5px; text-align: center; font-weight: bold; border: 2px solid #ff4444; font-size: 14px;'>{status_color} {status_text}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='background-color: rgba(68, 255, 68, 0.2); color: #228822; padding: 8px; border-radius: 5px; text-align: center; font-weight: bold; border: 2px solid #44ff44; font-size: 14px;'>{status_color} {status_text}</div>", unsafe_allow_html=True)
    
    values = sensor_data.get("values", {})
    trends = sensor_data.get("trends", {})
    
    # Display sensor metrics in organized grid
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        temp = values.get("temperature", 0.0)
        temp_trend = trends.get("temperature", "STABLE")
        delta_color = "normal" if temp_trend == "STABLE" else ("inverse" if temp_trend == "DOWN" else "off")
        st.metric("🌡️ Temperature", f"{temp:.1f}°C", delta=render_trend_arrow(temp_trend), delta_color=delta_color)
        
        pressure = values.get("pressure", 0.0)
        pressure_trend = trends.get("pressure", "STABLE")
        delta_color = "normal" if pressure_trend == "STABLE" else ("inverse" if pressure_trend == "DOWN" else "off")
        st.metric("⚙️ Pressure", f"{pressure:.1f} bar", delta=render_trend_arrow(pressure_trend), delta_color=delta_color)
    
    with col2:
        gas = values.get("gas_concentration", 0.0)
        gas_trend = trends.get("gas_concentration", "STABLE")
        delta_color = "normal" if gas_trend == "STABLE" else ("inverse" if gas_trend == "DOWN" else "off")
        st.metric("☁️ Gas Concentration", f"{gas:.1f} ppm", delta=render_trend_arrow(gas_trend), delta_color=delta_color)
        
        vibration = values.get("vibration", 0.0)
        vibration_trend = trends.get("vibration", "STABLE")
        delta_color = "normal" if vibration_trend == "STABLE" else ("inverse" if vibration_trend == "DOWN" else "off")
        st.metric("📳 Vibration", f"{vibration:.2f} mm/s", delta=render_trend_arrow(vibration_trend), delta_color=delta_color)
    
    with col3:
        flow = values.get("flow_rate", 0.0)
        flow_trend = trends.get("flow_rate", "STABLE")
        delta_color = "normal" if flow_trend == "STABLE" else ("inverse" if flow_trend == "DOWN" else "off")
        st.metric("💧 Flow Rate", f"{flow:.1f} L/min", delta=render_trend_arrow(flow_trend), delta_color=delta_color)
        
        valve_status = values.get("valve_status", "STABLE")
        valve_color = "#ff4444" if valve_status == "UNSTABLE" else "#44ff44"
        st.markdown(f"<div style='background-color: {valve_color}; color: {'white' if valve_status == 'UNSTABLE' else 'black'}; padding: 15px; border-radius: 5px; text-align: center; font-weight: bold; margin-top: 8px;'>🔧 Valve: {valve_status}</div>", unsafe_allow_html=True)