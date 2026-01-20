"""
Sensor Stream Simulator
Simulates real-time sensor data streams from CSV files
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Iterator, Optional
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class SensorStreamSimulator:
    """Simulates streaming sensor data from CSV"""
    
    def __init__(self, csv_path: str = "data/sensors/sensor_stream.csv"):
        """
        Initialize sensor stream simulator
        
        Args:
            csv_path: Path to sensor data CSV
        """
        self.csv_path = Path(csv_path)
        self.data = None
        self.sensors = {}
        
    def generate_synthetic_data(
        self,
        num_sensors: int = 5,
        num_readings: int = 1000,
        save_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Generate synthetic sensor data
        
        Args:
            num_sensors: Number of sensors to simulate
            num_readings: Number of readings per sensor
            save_path: Optional path to save generated data
            
        Returns:
            DataFrame with synthetic sensor data
        """
        logger.info(f"Generating synthetic data: {num_sensors} sensors, {num_readings} readings")
        
        zones = ['ZONE_A', 'ZONE_B', 'ZONE_C', 'ZONE_D']
        data_list = []
        
        base_time = datetime.now()
        
        for sensor_idx in range(num_sensors):
            sensor_id = f"SENSOR_{sensor_idx:03d}"
            zone = zones[sensor_idx % len(zones)]
            
            for reading_idx in range(num_readings):
                timestamp = base_time + timedelta(seconds=reading_idx * 10)
                
                # Normal patterns with occasional anomalies
                is_anomaly = np.random.random() < 0.05  # 5% anomaly rate
                
                if is_anomaly:
                    # Anomalous readings
                    gas_ppm = np.random.uniform(300, 1000)
                    pressure = np.random.uniform(90, 130)
                    temp = np.random.uniform(20, 60)
                    humidity = np.random.uniform(20, 90)
                else:
                    # Normal readings with some noise
                    gas_ppm = np.random.normal(50, 10)
                    pressure = np.random.normal(101.3, 2)
                    temp = np.random.normal(25, 3)
                    humidity = np.random.normal(50, 10)
                
                # Ensure positive values
                gas_ppm = max(0, gas_ppm)
                pressure = max(0, pressure)
                temp = max(-50, min(100, temp))
                humidity = max(0, min(100, humidity))
                
                data_list.append({
                    'timestamp': timestamp.isoformat(),
                    'sensor_id': sensor_id,
                    'zone': zone,
                    'gas_ppm': round(gas_ppm, 2),
                    'pressure': round(pressure, 2),
                    'temp': round(temp, 2),
                    'humidity': round(humidity, 2)
                })
        
        df = pd.DataFrame(data_list)
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(save_path, index=False)
            logger.info(f"Synthetic data saved to {save_path}")
        
        return df
    
    def load_data(self) -> bool:
        """
        Load sensor data from CSV
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.csv_path.exists():
                logger.warning(f"CSV file not found: {self.csv_path}")
                logger.info("Generating synthetic data...")
                
                self.data = self.generate_synthetic_data(
                    num_sensors=5,
                    num_readings=1000,
                    save_path=str(self.csv_path)
                )
            else:
                logger.info(f"Loading sensor data from {self.csv_path}")
                self.data = pd.read_csv(self.csv_path)
            
            # Validate columns
            required_cols = ['timestamp', 'sensor_id', 'zone', 'gas_ppm', 
                           'pressure', 'temp', 'humidity']
            
            missing_cols = set(required_cols) - set(self.data.columns)
            if missing_cols:
                logger.error(f"Missing required columns: {missing_cols}")
                return False
            
            # Group by sensor
            for sensor_id in self.data['sensor_id'].unique():
                sensor_data = self.data[self.data['sensor_id'] == sensor_id]
                zone = sensor_data['zone'].iloc[0]
                self.sensors[sensor_id] = {
                    'zone': zone,
                    'data': sensor_data.to_dict('records')
                }
            
            logger.info(f"Loaded {len(self.data)} readings from {len(self.sensors)} sensors")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return False
    
    def get_training_windows(
        self,
        num_windows: int = 10,
        window_size: int = 50
    ) -> List[Tuple[str, str, List[Dict]]]:
        """
        Generate training windows from initial data
        
        Args:
            num_windows: Number of windows to generate
            window_size: Size of each window
            
        Returns:
            List of (sensor_id, zone, window_data) tuples
        """
        if self.data is None:
            logger.error("No data loaded")
            return []
        
        training_windows = []
        
        for sensor_id, sensor_info in self.sensors.items():
            readings = sensor_info['data']
            zone = sensor_info['zone']
            
            # Take first N windows from each sensor
            windows_per_sensor = max(1, num_windows // len(self.sensors))
            
            for i in range(windows_per_sensor):
                start_idx = i * window_size
                end_idx = start_idx + window_size
                
                if end_idx <= len(readings):
                    window = readings[start_idx:end_idx]
                    training_windows.append((sensor_id, zone, window))
        
        logger.info(f"Generated {len(training_windows)} training windows")
        return training_windows
    
    def stream_windows(
        self,
        window_size: int = 50,
        batch_size: int = 5
    ) -> Iterator[List[Tuple[str, str, List[Dict]]]]:
        """
        Stream sensor windows in batches (simulates real-time)
        
        Args:
            window_size: Size of sliding window
            batch_size: Number of windows per batch
            
        Yields:
            Batches of (sensor_id, zone, window_data) tuples
        """
        if self.data is None:
            logger.error("No data loaded")
            return
        
        # Track current position for each sensor
        positions = {sensor_id: window_size * 10 for sensor_id in self.sensors.keys()}
        
        batch = []
        
        while True:
            all_exhausted = True
            
            for sensor_id, sensor_info in self.sensors.items():
                readings = sensor_info['data']
                zone = sensor_info['zone']
                
                start_idx = positions[sensor_id]
                end_idx = start_idx + window_size
                
                if end_idx <= len(readings):
                    all_exhausted = False
                    window = readings[start_idx:end_idx]
                    
                    batch.append((sensor_id, zone, window))
                    positions[sensor_id] += 1
                    
                    if len(batch) >= batch_size:
                        yield batch
                        batch = []
            
            if all_exhausted:
                if batch:
                    yield batch
                break