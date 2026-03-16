"""Unit tests for sensor embedding adapter"""

import pytest
import torch
import numpy as np
from src.models.sensor_adapter import SensorEmbeddingAdapter


class TestSensorEmbeddingAdapter:
    """Test sensor embedding adapter"""
    
    @pytest.fixture
    def adapter(self):
        """Create adapter instance"""
        return SensorEmbeddingAdapter(input_dim=5, hidden_dim=64, embed_dim=128)
        
    @pytest.fixture
    def sample_sensor_data(self):
        """Create sample sensor data"""
        return {
            'temperature_celsius': 95.0,
            'pressure_bar': 18.5,
            'gas_concentration_ppm': 450.0,
            'vibration_mm_s': 12.0,
            'flow_rate_lpm': 70.0
        }
        
    def test_adapter_initialization(self, adapter):
        """Test adapter initialization"""
        assert adapter.input_dim == 5
        assert adapter.hidden_dim == 64
        assert adapter.embed_dim == 128
        assert isinstance(adapter.fc1, torch.nn.Linear)
        assert isinstance(adapter.fc2, torch.nn.Linear)
        
    def test_forward_pass_shape(self, adapter):
        """Test forward pass output shape"""
        x = torch.randn(1, 5)
        output = adapter.forward(x)
        
        assert output.shape == (1, 128)
        
    def test_forward_pass_batch(self, adapter):
        """Test forward pass with batch"""
        x = torch.randn(10, 5)
        output = adapter.forward(x)
        
        assert output.shape == (10, 128)
        
    def test_forward_pass_range(self, adapter):
        """Test forward pass output range (tanh activation)"""
        x = torch.randn(1, 5)
        output = adapter.forward(x)
        
        # Tanh output should be in [-1, 1]
        assert torch.all(output >= -1.0)
        assert torch.all(output <= 1.0)
        
    def test_normalize_sensor_data(self, sample_sensor_data):
        """Test sensor data normalization"""
        normalized = SensorEmbeddingAdapter.normalize_sensor_data(sample_sensor_data)
        
        assert normalized.shape == (5,)
        assert normalized.dtype == np.float32
        
        # Check normalization (should be close to 0 for values near mean)
        # Temperature: (95 - 93.5) / 5.8 ≈ 0.26
        assert abs(normalized[0] - 0.26) < 0.1
        
    def test_normalize_missing_field(self):
        """Test normalization with missing field"""
        incomplete_data = {
            'temperature_celsius': 95.0,
            'pressure_bar': 18.5
            # Missing other fields
        }
        
        with pytest.raises(ValueError, match="Missing required sensor fields"):
            SensorEmbeddingAdapter.normalize_sensor_data(incomplete_data)
            
    def test_normalize_custom_params(self, sample_sensor_data):
        """Test normalization with custom means and stds"""
        custom_means = {k: 0.0 for k in sample_sensor_data.keys()}
        custom_stds = {k: 1.0 for k in sample_sensor_data.keys()}
        
        normalized = SensorEmbeddingAdapter.normalize_sensor_data(
            sample_sensor_data,
            means=custom_means,
            stds=custom_stds
        )
        
        # With mean=0 and std=1, normalized values should equal original values
        assert abs(normalized[0] - sample_sensor_data['temperature_celsius']) < 0.01
        
    def test_embed(self, adapter, sample_sensor_data):
        """Test embedding generation"""
        embedding = adapter.embed(sample_sensor_data)
        
        assert embedding.shape == (128,)
        assert embedding.dtype == np.float32
        
        # Check embedding is in valid range (tanh output)
        assert np.all(embedding >= -1.0)
        assert np.all(embedding <= 1.0)
        
    def test_embed_deterministic(self, adapter, sample_sensor_data):
        """Test that embedding is deterministic"""
        embedding1 = adapter.embed(sample_sensor_data)
        embedding2 = adapter.embed(sample_sensor_data)
        
        np.testing.assert_array_almost_equal(embedding1, embedding2)
        
    def test_embed_different_inputs(self, adapter):
        """Test that different inputs produce different embeddings"""
        data1 = {
            'temperature_celsius': 90.0,
            'pressure_bar': 15.0,
            'gas_concentration_ppm': 400.0,
            'vibration_mm_s': 10.0,
            'flow_rate_lpm': 70.0
        }
        
        data2 = {
            'temperature_celsius': 100.0,
            'pressure_bar': 20.0,
            'gas_concentration_ppm': 500.0,
            'vibration_mm_s': 15.0,
            'flow_rate_lpm': 80.0
        }
        
        embedding1 = adapter.embed(data1)
        embedding2 = adapter.embed(data2)
        
        # Embeddings should be different
        assert not np.allclose(embedding1, embedding2)
        
    def test_normalization_constants(self):
        """Test that normalization constants are defined"""
        assert 'temperature_celsius' in SensorEmbeddingAdapter.SENSOR_MEANS
        assert 'pressure_bar' in SensorEmbeddingAdapter.SENSOR_MEANS
        assert 'gas_concentration_ppm' in SensorEmbeddingAdapter.SENSOR_MEANS
        assert 'vibration_mm_s' in SensorEmbeddingAdapter.SENSOR_MEANS
        assert 'flow_rate_lpm' in SensorEmbeddingAdapter.SENSOR_MEANS
        
        assert 'temperature_celsius' in SensorEmbeddingAdapter.SENSOR_STDS
        assert 'pressure_bar' in SensorEmbeddingAdapter.SENSOR_STDS
        assert 'gas_concentration_ppm' in SensorEmbeddingAdapter.SENSOR_STDS
        assert 'vibration_mm_s' in SensorEmbeddingAdapter.SENSOR_STDS
        assert 'flow_rate_lpm' in SensorEmbeddingAdapter.SENSOR_STDS
        
    def test_save_load(self, adapter, sample_sensor_data, tmp_path):
        """Test model save and load"""
        # Generate embedding before save
        embedding_before = adapter.embed(sample_sensor_data)
        
        # Save model
        model_path = tmp_path / "model.pth"
        adapter.save(str(model_path))
        
        # Create new adapter and load weights
        new_adapter = SensorEmbeddingAdapter(input_dim=5, hidden_dim=64, embed_dim=128)
        new_adapter.load(str(model_path))
        
        # Generate embedding after load
        embedding_after = new_adapter.embed(sample_sensor_data)
        
        # Embeddings should be identical
        np.testing.assert_array_almost_equal(embedding_before, embedding_after)
