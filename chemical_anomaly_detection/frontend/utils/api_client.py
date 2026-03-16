import json
import time
import requests
import base64
from PIL import Image
import io
from config import API_ENDPOINT, START_ENDPOINT, STOP_ENDPOINT, STATUS_ENDPOINT


def start_monitoring(video_path, audio_path, sensor_path):
    """
    Start monitoring with specified data files
    
    Args:
        video_path: Path to video file
        audio_path: Path to audio file
        sensor_path: Path to sensor CSV file
        
    Returns:
        Response dict or None if failed
    """
    try:
        response = requests.post(
            START_ENDPOINT,
            params={
                "video_path": video_path,
                "audio_path": audio_path,
                "sensor_path": sensor_path
            },
            timeout=30  # Increased for cloud processing
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Failed to start monitoring: {e}")
        return None


def stop_monitoring():
    """
    Stop monitoring
    
    Returns:
        Response dict or None if failed
    """
    try:
        response = requests.post(STOP_ENDPOINT, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Failed to stop monitoring: {e}")
        return None


def get_backend_data():
    """
    Polls the backend for data
    
    Returns:
        The JSON structure as a dict, or None if unreachable
    """
    try:
        response = requests.get(API_ENDPOINT, timeout=30)  # Increased for cloud processing
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        print("Backend request timed out")
        return None
    except requests.exceptions.ConnectionError:
        print("Backend connection error - is the backend running?")
        return None
    except Exception as e:
        print(f"Backend error: {e}")
        return None


def get_status():
    """
    Get backend status
    
    Returns:
        Status dict or None if failed
    """
    try:
        response = requests.get(STATUS_ENDPOINT, timeout=30)  # Increased for cloud processing
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Failed to get status: {e}")
        return None