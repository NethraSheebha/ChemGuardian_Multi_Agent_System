"""Sensor data processor with async support and noise filtering"""

import asyncio
import logging
import numpy as np
from typing import Dict, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field, field_validator

from src.models.sensor_adapter import SensorEmbeddingAdapter


logger = logging.getLogger(__name__)


class SensorReading(BaseModel):
    """Pydantic model for sensor reading validation"""
    
    timestamp: datetime = Field(..., description="Reading timestamp")
    temperature_celsius: float = Field(..., ge=-50.0, le=200.0, description="Temperature in Celsius")
    pressure_bar: float = Field(..., ge=0.0, le=50.0, description="Pressure in bar")
    gas_concentration_ppm: float = Field(..., ge=0.0, le=10000.0, description="Gas concentration in ppm")
    vibration_mm_s: float = Field(..., ge=0.0, le=100.0, description="Vibration in mm/s")
    flow_rate_lpm: float = Field(..., ge=0.0, le=500.0, description="Flow rate in liters per minute")
    
    @field_validator('temperature_celsius', 'pressure_bar', 'gas_concentration_ppm', 
                     'vibration_mm_s', 'flow_rate_lpm')
    @classmethod
    def validate_not_nan(cls, v):
        """Validate that values are not NaN"""
        if np.isnan(v):
            raise ValueError("Sensor value cannot be NaN")
        return v


class SensorProcessor:
    """
    Processes sensor readings with noise filtering and embedding generation
    
    Features:
        - Async processing for non-blocking operations
        - Pydantic validation for sensor data
        - Noise filtering for outliers (>3 std devs)
        - Embedding generation using SensorEmbeddingAdapter
    """
    
    def __init__(
        self,
        adapter: SensorEmbeddingAdapter,
        noise_threshold: float = 3.0,
        enable_noise_filtering: bool = True
    ):
        """
        Initialize sensor processor
        
        Args:
            adapter: SensorEmbeddingAdapter instance
            noise_threshold: Number of standard deviations for outlier detection (default: 3.0)
            enable_noise_filtering: Whether to enable noise filtering (default: True)
        """
        self.adapter = adapter
        self.noise_threshold = noise_threshold
        self.enable_noise_filtering = enable_noise_filtering
        
        logger.info(
            f"Initialized SensorProcessor: "
            f"noise_threshold={noise_threshold}, "
            f"filtering_enabled={enable_noise_filtering}"
        )
        
    def _is_outlier(self, value: float, mean: float, std: float) -> bool:
        """
        Check if a value is an outlier using z-score
        
        Args:
            value: Value to check
            mean: Mean of the distribution
            std: Standard deviation of the distribution
            
        Returns:
            True if value is an outlier (>3 std devs from mean)
        """
        if std == 0:
            return False
        z_score = abs((value - mean) / std)
        return z_score > self.noise_threshold
        
    def filter_noise(self, sensor_data: Dict[str, float]) -> Dict[str, float]:
        """
        Filter noise from sensor readings
        
        Replaces outliers (>3 std devs) with the mean value.
        
        Args:
            sensor_data: Dictionary with sensor readings
            
        Returns:
            Filtered sensor data dictionary
        """
        if not self.enable_noise_filtering:
            return sensor_data
            
        filtered_data = sensor_data.copy()
        
        # Check each sensor value for outliers
        for field, value in sensor_data.items():
            if field == 'timestamp':
                continue
                
            # Get mean and std for this field
            mean = self.adapter.SENSOR_MEANS.get(field)
            std = self.adapter.SENSOR_STDS.get(field)
            
            if mean is not None and std is not None:
                if self._is_outlier(value, mean, std):
                    logger.warning(
                        f"Outlier detected in {field}: {value:.2f} "
                        f"(mean={mean:.2f}, std={std:.2f}). Replacing with mean."
                    )
                    filtered_data[field] = mean
                    
        return filtered_data
        
    def validate_sensor_data(self, sensor_data: Dict[str, Any]) -> SensorReading:
        """
        Validate sensor data using Pydantic model
        
        Args:
            sensor_data: Dictionary with sensor readings
            
        Returns:
            Validated SensorReading instance
            
        Raises:
            ValueError: If validation fails
        """
        try:
            return SensorReading(**sensor_data)
        except Exception as e:
            logger.error(f"Sensor data validation failed: {e}")
            raise ValueError(f"Invalid sensor data: {e}") from e
            
    async def process(
        self,
        sensor_data: Dict[str, Any],
        validate: bool = True,
        filter_noise: bool = True
    ) -> Optional[np.ndarray]:
        """
        Process sensor reading and generate embedding
        
        Args:
            sensor_data: Dictionary with sensor readings
            validate: Whether to validate sensor data (default: True)
            filter_noise: Whether to filter noise (default: True)
            
        Returns:
            Embedding numpy array of shape (128,) or None if processing fails
        """
        try:
            # Validate sensor data
            if validate:
                validated = self.validate_sensor_data(sensor_data)
                sensor_dict = validated.model_dump()
            else:
                sensor_dict = sensor_data
                
            # Filter noise
            if filter_noise and self.enable_noise_filtering:
                sensor_dict = self.filter_noise(sensor_dict)
                
            # Generate embedding (run in executor to avoid blocking)
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None,
                self.adapter.embed,
                sensor_dict
            )
            
            logger.debug(
                f"Generated sensor embedding: shape={embedding.shape}, "
                f"mean={embedding.mean():.4f}, std={embedding.std():.4f}"
            )
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to process sensor data: {e}")
            return None
            
    async def process_batch(
        self,
        sensor_readings: list[Dict[str, Any]],
        validate: bool = True,
        filter_noise: bool = True
    ) -> list[Optional[np.ndarray]]:
        """
        Process multiple sensor readings in parallel
        
        Args:
            sensor_readings: List of sensor reading dictionaries
            validate: Whether to validate sensor data (default: True)
            filter_noise: Whether to filter noise (default: True)
            
        Returns:
            List of embedding arrays (or None for failed readings)
        """
        tasks = [
            self.process(reading, validate, filter_noise)
            for reading in sensor_readings
        ]
        
        embeddings = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to None
        results = []
        for i, emb in enumerate(embeddings):
            if isinstance(emb, Exception):
                logger.error(f"Failed to process reading {i}: {emb}")
                results.append(None)
            else:
                results.append(emb)
                
        return results
