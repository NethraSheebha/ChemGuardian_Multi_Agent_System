import streamlit as st
import time
from components import video_panel, audio_panel, sensor_panel, alert_panel
from utils import api_client, ui_helpers
from config import REFRESH_INTERVAL, DEFAULT_VIDEO_PATH, DEFAULT_AUDIO_PATH, DEFAULT_SENSOR_PATH

st.set_page_config(layout="wide", page_title="Chemical Leak Monitoring Dashboard")

# Add custom CSS
st.markdown("""
<style>
    /* Smooth transitions for all elements */
    .stApp {
        transition: opacity 0.1s ease-in-out;
    }
    
    /* Reduce flicker on rerun */
    .element-container {
        transition: all 0.1s ease-in-out;
    }
    
    /* Smooth image transitions */
    img {
        transition: opacity 0.2s ease-in-out;
    }
    
    /* Prevent layout shift */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'monitoring' not in st.session_state:
    st.session_state.monitoring = False
if 'data' not in st.session_state:
    st.session_state.data = None
if 'backend_status' not in st.session_state:
    st.session_state.backend_status = None

# Sidebar
with st.sidebar:
    st.header("Controls")
    
    # Data file inputs
    video_path = st.text_input("Video Path", value=DEFAULT_VIDEO_PATH)
    audio_path = st.text_input("Audio Path", value=DEFAULT_AUDIO_PATH)
    sensor_path = st.text_input("Sensor Path", value=DEFAULT_SENSOR_PATH)
    
    # Control buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Monitoring", type="primary"):
            result = api_client.start_monitoring(video_path, audio_path, sensor_path)
            if result and result.get("status") == "started":
                st.session_state.monitoring = True
                st.success("✅ Monitoring started!")
                st.write(f"Video frames: {result.get('video_frames')}")
                st.write(f"Audio duration: {result.get('audio_duration'):.1f}s")
                st.write(f"Sensor samples: {result.get('sensor_samples')}")
            elif result and result.get("status") == "already_running":
                st.session_state.monitoring = True
                st.info("Already running")
            else:
                st.error("❌ Failed to start - is backend running?")
    
    with col2:
        if st.button("Stop Monitoring", type="secondary"):
            result = api_client.stop_monitoring()
            if result:
                st.session_state.monitoring = False
                st.success("⏹️ Monitoring stopped")
            else:
                st.error("Failed to stop")
    
    st.divider()
    
    # Backend status
    st.subheader("Backend Status")
    status = api_client.get_status()
    if status:
        st.session_state.backend_status = status
        st.write(f"🟢 Backend: **Online**")
        st.write(f"Monitoring: **{'Active' if status.get('monitoring') else 'Inactive'}**")
    else:
        st.write(f"🔴 Backend: **Offline**")
        st.write("Start backend with: `python backend_api.py`")
    
    st.divider()
    
    # System Metrics
    st.subheader("System Metrics")
    if st.session_state.data:
        metrics = st.session_state.data.get("system_metrics", {})
        latency = st.session_state.data.get("latency_ms", {})
        st.metric("Samples Processed", metrics.get('samples_processed', 0))
        st.metric("Anomalies Detected", metrics.get('anomalies_detected', 0))
        st.metric("Anomaly Rate", f"{metrics.get('anomaly_rate', 0.0):.1f}%")
        st.metric("Uptime", f"{metrics.get('uptime_seconds', 0)}s")
        st.metric("Total Latency", f"{latency.get('total', 0)}ms")
        st.metric("Qdrant Latency", f"{latency.get('qdrant', 0)}ms")
        st.metric("Embedding Latency", f"{latency.get('embedding', 0)}ms")
    else:
        st.write("No metrics available")

# Main title
st.title("🏭 Chemical Leak Monitoring Dashboard")

# Main layout: Video on left, Audio/Sensor on right (top/bottom), Alert at bottom
col_left, col_right = st.columns([1, 2])  # Left: Video (1/3), Right: Audio/Sensor (2/3)

# Monitoring loop
if st.session_state.monitoring:
    # Fetch data from backend
    data = api_client.get_backend_data()
    
    if data is None:
        st.error("⚠️ Backend unreachable - retrying...")
        time.sleep(REFRESH_INTERVAL)
        st.rerun()
    elif data.get("status") == "not_monitoring":
        st.session_state.monitoring = False
        st.warning("Backend stopped monitoring")
        time.sleep(1)
        st.rerun()
    else:
        # Store data in session state
        st.session_state.data = data
        
        # Render panels directly in columns
        with col_left:
            video_panel.render(data.get("video"))
        
        with col_right:
            audio_panel.render(data.get("audio"))
            sensor_panel.render(data.get("sensors"))
        
        # Alert panel at bottom
        alert_panel.render(
            data.get("alert"),
            data.get("actions", []),
            data.get("timestamp")
        )
        
        # Auto-refresh
        time.sleep(REFRESH_INTERVAL)
        st.rerun()
else:
    # Not monitoring - show instructions
    st.info("👈 Click **Start Monitoring** in the sidebar to begin")
    
    with col_left:
        st.subheader("Video Monitoring")
        st.write("Real-time video feed with anomaly detection")
        st.write("- Displays live camera feed")
        st.write("- Highlights anomalies with bounding boxes")
        st.write("- Shows camera location and status")
    
    with col_right:
        st.subheader("Audio Monitoring")
        st.write("Audio waveform analysis")
        st.write("- Real-time waveform visualization")
        st.write("- Audio metrics (peak, RMS, ZCR)")
        st.write("- Anomaly detection for unusual sounds")
        
        st.divider()
        
        st.subheader("Sensor Monitoring")
        st.write("Multi-sensor data tracking")
        st.write("- Pressure, temperature, gas concentration")
        st.write("- Flow rate and vibration monitoring")
        st.write("- Trend indicators and valve status")