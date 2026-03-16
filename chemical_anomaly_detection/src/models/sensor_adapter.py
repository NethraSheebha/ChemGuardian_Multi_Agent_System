"""Sensor embedding adapter neural network"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional
import logging


logger = logging.getLogger(__name__)


class SensorEmbeddingAdapter(nn.Module):
    """
    Neural network adapter for sensor data embeddings
    
    Architecture: Input(5) -> Dense(64, ReLU) -> Dense(128, tanh)
    
    Learns non-linear relationships between sensor features and generates
    compact 128-dimensional embeddings for similarity search.
    
    Input features:
        - temperature_celsius
        - pressure_bar
        - gas_concentration_ppm
        - vibration_mm_s
        - flow_rate_lpm
    """
    
    # Normalization parameters (computed from normal_sensor_data.csv)
    SENSOR_MEANS = {
        'temperature_celsius': 74.94,
        'pressure_bar': 5.208,
        'gas_concentration_ppm': 318.2,
        'vibration_mm_s': 1.199,
        'flow_rate_lpm': 144.98
    }
    
    SENSOR_STDS = {
        'temperature_celsius': 1.60,
        'pressure_bar': 0.075,
        'gas_concentration_ppm': 11.1,
        'vibration_mm_s': 0.090,
        'flow_rate_lpm': 3.91
    }
    
    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 64,
        embed_dim: int = 128
    ):
        """
        Initialize sensor embedding adapter
        
        Args:
            input_dim: Number of input features (default: 5)
            hidden_dim: Hidden layer dimension (default: 64)
            embed_dim: Output embedding dimension (default: 128)
        """
        super(SensorEmbeddingAdapter, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        
        # Network layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.tanh = nn.Tanh()
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(
            f"Initialized SensorEmbeddingAdapter: "
            f"input={input_dim}, hidden={hidden_dim}, embed={embed_dim}"
        )
        
    def _initialize_weights(self) -> None:
        """Initialize network weights using Xavier initialization"""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
               Expected to be normalized sensor readings
               
        Returns:
            Embedding tensor of shape (batch_size, embed_dim)
        """
        # First layer with ReLU activation
        x = self.fc1(x)
        x = self.relu(x)
        
        # Second layer with tanh activation
        x = self.fc2(x)
        x = self.tanh(x)
        
        return x
        
    @staticmethod
    def normalize_sensor_data(
        sensor_data: Dict[str, float],
        means: Optional[Dict[str, float]] = None,
        stds: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """
        Normalize sensor data using z-score normalization
        
        Args:
            sensor_data: Dictionary with sensor readings
            means: Optional custom means (uses class defaults if None)
            stds: Optional custom stds (uses class defaults if None)
            
        Returns:
            Normalized numpy array of shape (5,)
            
        Raises:
            ValueError: If required sensor fields are missing
        """
        if means is None:
            means = SensorEmbeddingAdapter.SENSOR_MEANS
        if stds is None:
            stds = SensorEmbeddingAdapter.SENSOR_STDS
            
        # Required fields
        required_fields = [
            'temperature_celsius',
            'pressure_bar',
            'gas_concentration_ppm',
            'vibration_mm_s',
            'flow_rate_lpm'
        ]
        
        # Check for missing fields
        missing_fields = [f for f in required_fields if f not in sensor_data]
        if missing_fields:
            raise ValueError(f"Missing required sensor fields: {missing_fields}")
        
        # Normalize each feature
        normalized = np.array([
            (sensor_data['temperature_celsius'] - means['temperature_celsius']) / stds['temperature_celsius'],
            (sensor_data['pressure_bar'] - means['pressure_bar']) / stds['pressure_bar'],
            (sensor_data['gas_concentration_ppm'] - means['gas_concentration_ppm']) / stds['gas_concentration_ppm'],
            (sensor_data['vibration_mm_s'] - means['vibration_mm_s']) / stds['vibration_mm_s'],
            (sensor_data['flow_rate_lpm'] - means['flow_rate_lpm']) / stds['flow_rate_lpm']
        ], dtype=np.float32)
        
        return normalized
        
    def embed(self, sensor_data: Dict[str, float]) -> np.ndarray:
        """
        Generate embedding for sensor data
        
        Args:
            sensor_data: Dictionary with sensor readings
            
        Returns:
            Embedding numpy array of shape (128,)
        """
        # Normalize sensor data
        normalized = self.normalize_sensor_data(sensor_data)
        
        # Convert to tensor
        x = torch.from_numpy(normalized).unsqueeze(0)  # Add batch dimension
        
        # Generate embedding
        with torch.no_grad():
            embedding = self.forward(x)
            
        # Convert back to numpy and remove batch dimension
        return embedding.squeeze(0).numpy()
        
    def save(self, path: str) -> None:
        """
        Save model weights to file
        
        Args:
            path: Path to save model weights
        """
        torch.save(self.state_dict(), path)
        logger.info(f"Saved model weights to {path}")
        
    def load(self, path: str) -> None:
        """
        Load model weights from file
        
        Args:
            path: Path to load model weights from
        """
        self.load_state_dict(torch.load(path))
        self.eval()  # Set to evaluation mode
        logger.info(f"Loaded model weights from {path}")
