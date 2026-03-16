import streamlit as st
from utils.ui_helpers import render_status_indicator

def render(alert_data, actions, timestamp):
    """Renders the enhanced alerts and actions panel."""
    
    if not alert_data:
        st.info("⚠️ Waiting for alert data...")
        return
    
    system_status = alert_data.get("system_status", "NORMAL")
    
    # Compact status banner with smaller font
    if system_status == "HIGH":
        st.markdown("<div style='background-color: rgba(204, 0, 0, 0.15); color: #cc0000; padding: 12px; border-radius: 8px; text-align: center; font-size: 18px; font-weight: bold; margin-bottom: 15px; border: 2px solid #cc0000;'>🚨 HIGH SEVERITY ALERT</div>", unsafe_allow_html=True)
    elif system_status == "MEDIUM":
        st.markdown("<div style='background-color: rgba(255, 136, 0, 0.15); color: #ff8800; padding: 12px; border-radius: 8px; text-align: center; font-size: 18px; font-weight: bold; margin-bottom: 15px; border: 2px solid #ff8800;'>⚠️ MEDIUM SEVERITY ALERT</div>", unsafe_allow_html=True)
    elif system_status == "MILD":
        st.markdown("<div style='background-color: rgba(255, 204, 0, 0.15); color: #cc9900; padding: 12px; border-radius: 8px; text-align: center; font-size: 18px; font-weight: bold; margin-bottom: 15px; border: 2px solid #ffcc00;'>⚡ MILD SEVERITY ALERT</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='background-color: rgba(68, 255, 68, 0.15); color: #228822; padding: 12px; border-radius: 8px; text-align: center; font-size: 18px; font-weight: bold; margin-bottom: 15px; border: 2px solid #44ff44;'>✅ SYSTEM NORMAL</div>", unsafe_allow_html=True)
    
    # Alert details in organized layout with smaller fonts
    col1, col2 = st.columns(2)
    
    with col1:
        risk_level = alert_data.get('risk_level', 'NORMAL')
        st.markdown(f"<div style='font-size: 14px; font-weight: bold; margin-bottom: 5px;'>📊 Risk Level</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size: 16px; font-weight: bold; color: {'#cc0000' if risk_level == 'HIGH' else '#ff8800' if risk_level == 'MEDIUM' else '#cc9900' if risk_level == 'MILD' else '#228822'}; margin-bottom: 10px;'>{risk_level}</div>", unsafe_allow_html=True)
        
        confidence = alert_data.get('confidence', 0.0)
        if confidence > 0:
            st.markdown(f"<div style='font-size: 14px; font-weight: bold; margin-bottom: 5px;'>🎯 Confidence</div>", unsafe_allow_html=True)
            st.progress(confidence)
            st.markdown(f"<div style='text-align: center; font-weight: bold; font-size: 13px;'>{confidence*100:.1f}%</div>", unsafe_allow_html=True)
    
    with col2:
        affected = alert_data.get('affected_modalities', [])
        st.markdown(f"<div style='font-size: 14px; font-weight: bold; margin-bottom: 5px;'>🎯 Affected Modalities</div>", unsafe_allow_html=True)
        if affected:
            for modality in affected:
                icon = "🎥" if modality == "video" else "🔊" if modality == "audio" else "📡"
                st.markdown(f"<div style='background-color: rgba(255, 68, 68, 0.15); color: #ff4444; padding: 6px; border-radius: 4px; margin: 4px 0; font-weight: bold; font-size: 13px; border: 1px solid #ff4444;'>{icon} {modality.upper()}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='color: #888; font-size: 13px;'>None</div>", unsafe_allow_html=True)
        
        st.markdown(f"<div style='font-size: 14px; font-weight: bold; margin-top: 10px; margin-bottom: 5px;'>🕐 Timestamp</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size: 12px;'>{timestamp}</div>", unsafe_allow_html=True)
    
    # Cause analysis - only show if there's actual content
    cause = alert_data.get('cause', None)
    explanation = alert_data.get('explanation', '')
    
    if cause and cause != "None" and system_status != "NORMAL":
        st.markdown("---")
        st.markdown(f"<div style='font-size: 15px; font-weight: bold; margin-bottom: 8px;'>🔍 Root Cause Analysis</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='background-color: rgba(240, 240, 240, 0.5); padding: 10px; border-radius: 5px; border-left: 3px solid #ff8800; font-size: 13px;'><strong>Primary Cause:</strong> {cause}</div>", unsafe_allow_html=True)
        
        if explanation and explanation != "No anomaly detected" and len(explanation) > 10:
            st.markdown(f"<div style='background-color: rgba(248, 248, 248, 0.5); padding: 10px; border-radius: 5px; margin-top: 8px; font-size: 12px; line-height: 1.4;'><strong>Analysis:</strong> {explanation}</div>", unsafe_allow_html=True)
    
    # Recommended actions - only show if there are actions
    if actions and len(actions) > 0:
        st.markdown("---")
        st.markdown(f"<div style='font-size: 15px; font-weight: bold; margin-bottom: 8px;'>🎬 Recommended Actions ({len(actions)})</div>", unsafe_allow_html=True)
        
        for i, action in enumerate(actions, 1):
            # Determine urgency color
            action_upper = action.upper()
            if "IMMEDIATE" in action_upper or "EMERGENCY" in action_upper:
                bg_color = "rgba(204, 0, 0, 0.15)"
                text_color = "#cc0000"
                border_color = "#cc0000"
                icon = "🚨"
            elif "URGENT" in action_upper:
                bg_color = "rgba(255, 136, 0, 0.15)"
                text_color = "#ff8800"
                border_color = "#ff8800"
                icon = "⚠️"
            else:
                bg_color = "rgba(68, 68, 255, 0.1)"
                text_color = "#4444ff"
                border_color = "#4444ff"
                icon = "ℹ️"
            
            st.markdown(f"<div style='background-color: {bg_color}; color: {text_color}; padding: 8px; border-radius: 5px; margin: 6px 0; font-size: 12px; border: 1px solid {border_color};'><strong>{icon} Action {i}:</strong> {action}</div>", unsafe_allow_html=True)