"""Unit tests for sensor processor"""

import pytest
import asyncio
import numpy as np
from datetime import datetime
from src.models.sensor_adapter import SensorEmbeddingAdapter
from src.models.sensor_processor import SensorProcessor, SensorReading


class TestSensorReading:
    """Test SensorReading Pydantic model"""
    
    def test_valid_sensor_reading(self):
        """Test valid sensor reading"""
        data = {
            'timestamp': datetime.now(),
            'temperature_celsius': 95.0,
            'pressure_bar': 18.5,
            'gas_concentration_ppm': 450.0,
            'vibration_mm_s': 12.0,
            'flow_rate_lpm': 70.0
        }
        
        reading = SensorReading(**data)
        assert reading.temperature_celsius == 95.0
        assert reading.pressure_bar == 18.5
        
    def test_invalid_temperature_range(self):
        """Test invalid temperature range"""
        data = {
            'timestamp': datetime.now(),
            'temperature_celsius': 250.0,  # > 200
            'pressure_bar': 18.5,
            'gas_concentration_ppm': 450.0,
            'vibration_mm_s': 12.0,
            'flow_rate_lpm': 70.0
        }
        
        with pytest.raises(ValueError):
            SensorReading(**data)
            
    def test_invalid_pressure_range(self):
        """Test invalid pressure range"""
        data = {
            'timestamp': datetime.now(),
            'temperature_celsius': 95.0,
            'pressure_bar': -5.0,  # < 0
            'gas_concentration_ppm': 450.0,
            'vibration_mm_s': 12.0,
            'flow_rate_lpm': 70.0
        }
        
        with pytest.raises(ValueError):
            SensorReading(**data)
            
    def test_nan_value(self):
        """Test NaN value rejection"""
        data = {
            'timestamp': datetime.now(),
            'temperature_celsius': float('nan'),
            'pressure_bar': 18.5,
            'gas_concentration_ppm': 450.0,
            'vibration_mm_s': 12.0,
            'flow_rate_lpm': 70.0
        }
        
        # Pydantic catches NaN at validation level
        with pytest.raises(ValueError):
            SensorReading(**data)


class TestSensorProcessor:
    """Test sensor processor"""
    
    @pytest.fixture
    def adapter(self):
        """Create adapter instance"""
        return SensorEmbeddingAdapter(input_dim=5, hidden_dim=64, embed_dim=128)
        
    @pytest.fixture
    def processor(self, adapter):
        """Create processor instance"""
        return SensorProcessor(adapter, noise_threshold=3.0, enable_noise_filtering=True)
        
    @pytest.fixture
    def sample_sensor_data(self):
        """Create sample sensor data"""
        return {
            'timestamp': datetime.now(),
            'temperature_celsius': 95.0,
            'pressure_bar': 18.5,
            'gas_concentration_ppm': 450.0,
            'vibration_mm_s': 12.0,
            'flow_rate_lpm': 70.0
        }
        
    def test_processor_initialization(self, processor):
        """Test processor initialization"""
        assert processor.noise_threshold == 3.0
        assert processor.enable_noise_filtering is True
        
    def test_is_outlier(self, processor):
        """Test outlier detection"""
        # Value 3.5 std devs away is an outlier
        assert processor._is_outlier(100.0, 50.0, 10.0) is True
        
        # Value 2 std devs away is not an outlier
        assert processor._is_outlier(70.0, 50.0, 10.0) is False
        
        # Edge case: std = 0
        assert processor._is_outlier(50.0, 50.0, 0.0) is False
        
    def test_filter_noise_no_outliers(self, processor, sample_sensor_data):
        """Test noise filtering with no outliers"""
        filtered = processor.filter_noise(sample_sensor_data)
        
        # Data should be unchanged
        assert filtered['temperature_celsius'] == sample_sensor_data['temperature_celsius']
        assert filtered['pressure_bar'] == sample_sensor_data['pressure_bar']
        
    def test_filter_noise_with_outlier(self, processor):
        """Test noise filtering with outlier"""
        data = {
            'timestamp': datetime.now(),
            'temperature_celsius': 150.0,  # Outlier (mean=93.5, std=5.8)
            'pressure_bar': 18.5,
            'gas_concentration_ppm': 450.0,
            'vibration_mm_s': 12.0,
            'flow_rate_lpm': 70.0
        }
        
        filtered = processor.filter_noise(data)
        
        # Outlier should be replaced with mean
        assert filtered['temperature_celsius'] == 93.5
        assert filtered['pressure_bar'] == 18.5  # Not an outlier
        
    def test_filter_noise_disabled(self, adapter):
        """Test noise filtering when disabled"""
        processor = SensorProcessor(adapter, enable_noise_filtering=False)
        
        data = {
            'timestamp': datetime.now(),
            'temperature_celsius': 150.0,  # Would be outlier
            'pressure_bar': 18.5,
            'gas_concentration_ppm': 450.0,
            'vibration_mm_s': 12.0,
            'flow_rate_lpm': 70.0
        }
        
        filtered = processor.filter_noise(data)
        
        # Data should be unchanged
        assert filtered['temperature_celsius'] == 150.0
        
    def test_validate_sensor_data_valid(self, processor, sample_sensor_data):
        """Test sensor data validation with valid data"""
        validated = processor.validate_sensor_data(sample_sensor_data)
        
        assert isinstance(validated, SensorReading)
        assert validated.temperature_celsius == 95.0
        
    def test_validate_sensor_data_invalid(self, processor):
        """Test sensor data validation with invalid data"""
        invalid_data = {
            'timestamp': datetime.now(),
            'temperature_celsius': 250.0,  # Out of range
            'pressure_bar': 18.5,
            'gas_concentration_ppm': 450.0,
            'vibration_mm_s': 12.0,
            'flow_rate_lpm': 70.0
        }
        
        with pytest.raises(ValueError, match="Invalid sensor data"):
            processor.validate_sensor_data(invalid_data)
            
    @pytest.mark.asyncio
    async def test_process_valid_data(self, processor, sample_sensor_data):
        """Test processing valid sensor data"""
        embedding = await processor.process(sample_sensor_data)
        
        assert embedding is not None
        assert embedding.shape == (128,)
        assert np.all(embedding >= -1.0)
        assert np.all(embedding <= 1.0)
        
    @pytest.mark.asyncio
    async def test_process_invalid_data(self, processor):
        """Test processing invalid sensor data"""
        invalid_data = {
            'timestamp': datetime.now(),
            'temperature_celsius': 250.0,  # Out of range
            'pressure_bar': 18.5,
            'gas_concentration_ppm': 450.0,
            'vibration_mm_s': 12.0,
            'flow_rate_lpm': 70.0
        }
        
        embedding = await processor.process(invalid_data)
        
        # Should return None for invalid data
        assert embedding is None
        
    @pytest.mark.asyncio
    async def test_process_without_validation(self, processor, sample_sensor_data):
        """Test processing without validation"""
        embedding = await processor.process(sample_sensor_data, validate=False)
        
        assert embedding is not None
        assert embedding.shape == (128,)
        
    @pytest.mark.asyncio
    async def test_process_without_noise_filtering(self, processor, sample_sensor_data):
        """Test processing without noise filtering"""
        embedding = await processor.process(sample_sensor_data, filter_noise=False)
        
        assert embedding is not None
        assert embedding.shape == (128,)
        
    @pytest.mark.asyncio
    async def test_process_batch(self, processor):
        """Test batch processing"""
        readings = [
            {
                'timestamp': datetime.now(),
                'temperature_celsius': 95.0 + i,
                'pressure_bar': 18.5,
                'gas_concentration_ppm': 450.0,
                'vibration_mm_s': 12.0,
                'flow_rate_lpm': 70.0
            }
            for i in range(5)
        ]
        
        embeddings = await processor.process_batch(readings)
        
        assert len(embeddings) == 5
        assert all(emb is not None for emb in embeddings)
        assert all(emb.shape == (128,) for emb in embeddings)
        
    @pytest.mark.asyncio
    async def test_process_batch_with_failures(self, processor):
        """Test batch processing with some failures"""
        readings = [
            {
                'timestamp': datetime.now(),
                'temperature_celsius': 95.0,
                'pressure_bar': 18.5,
                'gas_concentration_ppm': 450.0,
                'vibration_mm_s': 12.0,
                'flow_rate_lpm': 70.0
            },
            {
                'timestamp': datetime.now(),
                'temperature_celsius': 250.0,  # Invalid
                'pressure_bar': 18.5,
                'gas_concentration_ppm': 450.0,
                'vibration_mm_s': 12.0,
                'flow_rate_lpm': 70.0
            },
            {
                'timestamp': datetime.now(),
                'temperature_celsius': 90.0,
                'pressure_bar': 18.5,
                'gas_concentration_ppm': 450.0,
                'vibration_mm_s': 12.0,
                'flow_rate_lpm': 70.0
            }
        ]
        
        embeddings = await processor.process_batch(readings)
        
        assert len(embeddings) == 3
        assert embeddings[0] is not None
        assert embeddings[1] is None  # Failed
        assert embeddings[2] is not None
