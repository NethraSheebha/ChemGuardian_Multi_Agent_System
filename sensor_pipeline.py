"""
Sensor Data Processing Pipeline
Handles feature extraction, anomaly detection, and embedding generation
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters

logger = logging.getLogger(__name__)


class SensorPipeline:
    """Complete pipeline for sensor data analysis"""
    
    def __init__(
        self,
        window_size: int = 50,
        anomaly_threshold: float = -0.5,
        model_dir: str = "models"
    ):
        """
        Initialize sensor processing pipeline
        
        Args:
            window_size: Number of readings per window
            anomaly_threshold: Threshold for anomaly classification
            model_dir: Directory for model persistence
        """
        self.window_size = window_size
        self.anomaly_threshold = anomaly_threshold
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        # Models (initialized during training)
        self.iforest = None
        self.pca = None
        self.scaler = None
        
        # Training state
        self.is_trained = False
        
        # Feature extraction settings
        self.fc_parameters = EfficientFCParameters()
        
        logger.info(f"Pipeline initialized (window_size={window_size})")
    
    def _window_to_dataframe(
        self,
        window_data: List[Dict],
        sensor_id: str
    ) -> pd.DataFrame:
        """
        Convert window data to DataFrame for tsfresh
        
        Args:
            window_data: List of sensor readings
            sensor_id: Sensor identifier
            
        Returns:
            DataFrame with id and time columns
        """
        df = pd.DataFrame(window_data)
        
        # Add required columns for tsfresh
        df['id'] = sensor_id
        df['time'] = range(len(df))
        
        return df
    
    def extract_features(
        self,
        window_data: List[Dict],
        sensor_id: str
    ) -> Optional[np.ndarray]:
        """
        Extract time-series features using tsfresh
        
        Args:
            window_data: List of sensor readings
            sensor_id: Sensor identifier
            
        Returns:
            Feature vector as numpy array, or None on error
        """
        try:
            # Convert to DataFrame
            df = self._window_to_dataframe(window_data, sensor_id)
            
            # Select numeric columns for feature extraction
            value_columns = ['gas_ppm', 'pressure', 'temp', 'humidity']
            
            # Extract features
            features = extract_features(
                df[['id', 'time'] + value_columns],
                column_id='id',
                column_sort='time',
                default_fc_parameters=self.fc_parameters,
                disable_progressbar=True
            )
            
            # Remove NaN and inf values
            features = features.fillna(0)
            features = features.replace([np.inf, -np.inf], 0)
            
            return features.values.flatten()
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return None
    
    def compute_metrics(self, window_data: List[Dict]) -> Dict:
        """
        Compute basic statistical metrics from window
        
        Args:
            window_data: List of sensor readings
            
        Returns:
            Dictionary of metrics
        """
        df = pd.DataFrame(window_data)
        
        metrics = {
            'gas_ppm_mean': df['gas_ppm'].mean(),
            'gas_ppm_std': df['gas_ppm'].std(),
            'gas_ppm_max': df['gas_ppm'].max(),
            'pressure_mean': df['pressure'].mean(),
            'pressure_std': df['pressure'].std(),
            'temp_mean': df['temp'].mean(),
            'temp_std': df['temp'].std(),
            'humidity_mean': df['humidity'].mean(),
            'humidity_std': df['humidity'].std()
        }
        
        return metrics
    
    def train_models(self, training_windows: List[Tuple]):
        """
        Train anomaly detection and dimensionality reduction models
        
        Args:
            training_windows: List of (sensor_id, zone, window_data) tuples
        """
        logger.info(f"Training on {len(training_windows)} windows...")
        
        # Extract features from all training windows
        feature_matrix = []
        
        for sensor_id, zone, window_data in training_windows:
            features = self.extract_features(window_data, sensor_id)
            if features is not None:
                feature_matrix.append(features)
        
        if not feature_matrix:
            raise ValueError("No features extracted from training data")
        
        # Convert to numpy array
        X_train = np.array(feature_matrix)
        logger.info(f"Training feature matrix shape: {X_train.shape}")
        
        # Train scaler
        logger.info("Training StandardScaler...")
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Train Isolation Forest
        logger.info("Training IsolationForest...")
        self.iforest = IsolationForest(
            contamination=0.02,
            random_state=42,
            n_estimators=100
        )
        self.iforest.fit(X_scaled)
        
        # Train PCA for dimensionality reduction to 128
        logger.info("Training PCA...")
        n_samples, n_features = X_scaled.shape
        n_components = min(128, n_samples, n_features)
        self.pca = PCA(n_components=n_components)
        self.pca.fit(X_scaled)
        
        logger.info(f"PCA explained variance ratio: {self.pca.explained_variance_ratio_.sum():.4f}")
        
        # Save models
        self._save_models()
        
        self.is_trained = True
        logger.info("Model training complete")
    
    def _save_models(self):
        """Save trained models to disk"""
        try:
            with open(self.model_dir / "iforest.pkl", 'wb') as f:
                pickle.dump(self.iforest, f)
            
            with open(self.model_dir / "pca.pkl", 'wb') as f:
                pickle.dump(self.pca, f)
            
            with open(self.model_dir / "scaler.pkl", 'wb') as f:
                pickle.dump(self.scaler, f)
            
            logger.info(f"Models saved to {self.model_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save models: {e}")
    
    def _load_models(self) -> bool:
        """
        Load trained models from disk
        
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(self.model_dir / "iforest.pkl", 'rb') as f:
                self.iforest = pickle.load(f)
            
            with open(self.model_dir / "pca.pkl", 'rb') as f:
                self.pca = pickle.load(f)
            
            with open(self.model_dir / "scaler.pkl", 'rb') as f:
                self.scaler = pickle.load(f)
            
            self.is_trained = True
            logger.info("Models loaded successfully")
            return True
            
        except FileNotFoundError:
            logger.warning("Model files not found - training required")
            return False
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return False
    
    def analyze_window(
        self,
        window_data: List[Dict],
        sensor_id: str
    ) -> Optional[Tuple[bool, float, List[float], Dict]]:
        """
        Analyze a sensor window for anomalies
        
        Args:
            window_data: List of sensor readings
            sensor_id: Sensor identifier
            
        Returns:
            Tuple of (is_anomaly, score, embedding, metrics) or None on error
        """
        if not self.is_trained:
            # Try to load models
            if not self._load_models():
                logger.error("Models not trained - call train_models() first")
                return None
        
        try:
            # Extract features
            features = self.extract_features(window_data, sensor_id)
            if features is None:
                return None
            
            # Reshape for sklearn
            features = features.reshape(1, -1)
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Compute anomaly score
            score = self.iforest.decision_function(features_scaled)[0]
            
            # Generate 128-dim embedding using PCA
            embedding = self.pca.transform(features_scaled)[0]
            
            # Compute metrics
            metrics = self.compute_metrics(window_data)
            
            # Determine if anomaly
            is_anomaly = score <= self.anomaly_threshold
            
            return (is_anomaly, score, embedding.tolist(), metrics)
            
        except Exception as e:
            logger.error(f"Window analysis failed: {e}")
            return None