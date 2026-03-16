# Configuration constants
API_BASE_URL = "http://localhost:8000/api"
API_ENDPOINT = f"{API_BASE_URL}/data"
START_ENDPOINT = f"{API_BASE_URL}/start"
STOP_ENDPOINT = f"{API_BASE_URL}/stop"
STATUS_ENDPOINT = f"{API_BASE_URL}/status"

REFRESH_INTERVAL = 10  # seconds - display each result for 10s while next processes

# Default data paths (relative to backend root)
DEFAULT_VIDEO_PATH = "anomalous_1.mp4"
DEFAULT_AUDIO_PATH = "anomalous_audio.wav"
DEFAULT_SENSOR_PATH = "anomalous_sensor.csv"