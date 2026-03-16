import streamlit as st
import base64
import io
from PIL import Image, UnidentifiedImageError

def render(video_data):
    """Renders the enhanced video monitoring panel."""
    
    if not video_data:
        st.info("🎥 Waiting for video feed...")
        return
    
    # Header with status indicator
    anomaly = video_data.get("anomaly", False)
    status_color = "🔴" if anomaly else "🟢"
    status_text = "ANOMALY DETECTED" if anomaly else "NORMAL"
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"### 🎥 Video Feed")
    with col2:
        if anomaly:
            st.markdown(f"<div style='background-color: rgba(255, 68, 68, 0.2); color: #ff4444; padding: 8px; border-radius: 5px; text-align: center; font-weight: bold; border: 2px solid #ff4444; font-size: 14px;'>{status_color} {status_text}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='background-color: rgba(68, 255, 68, 0.2); color: #228822; padding: 8px; border-radius: 5px; text-align: center; font-weight: bold; border: 2px solid #44ff44; font-size: 14px;'>{status_color} {status_text}</div>", unsafe_allow_html=True)
    
    # Display frame
    frame_data = video_data.get("frame", "")
    
    if frame_data:
        try:
            if frame_data.startswith("data:image/"):
                header, encoded = frame_data.split(",", 1)
                image_bytes = base64.b64decode(encoded)
            else:
                image_bytes = base64.b64decode(frame_data)
            img = Image.open(io.BytesIO(image_bytes))
            
            # Add subtle border for anomalies
            if anomaly:
                st.markdown("<div style='border: 2px solid #ff4444; padding: 3px; border-radius: 5px;'>", unsafe_allow_html=True)
                st.image(img, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.image(img, use_container_width=True)
                
        except Exception as e:
            st.error(f"⚠️ Error loading frame: {str(e)}")
    else:
        st.warning("📷 No frame available")
    
    # Metadata in organized layout
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📍 Location", video_data.get('location', 'Unknown'))
    with col2:
        st.metric("📹 Camera", video_data.get('camera_id', 'N/A'))
    with col3:
        timestamp = video_data.get('timestamp', 'N/A')
        if timestamp != 'N/A' and 'T' in timestamp:
            # Format timestamp nicely
            time_part = timestamp.split('T')[1].split('.')[0]
            st.metric("🕐 Time", time_part)
        else:
            st.metric("🕐 Time", timestamp)