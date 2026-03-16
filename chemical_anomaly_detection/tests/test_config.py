"""Unit tests for configuration validation"""

import pytest
import os
from src.config.settings import (
    SystemConfig,
    QdrantConfig,
    ModelConfig,
    ThresholdConfig,
    AgentConfig,
    LoggingConfig
)


class TestQdrantConfig:
    """Test Qdrant configuration validation"""
    
    def test_valid_config(self):
        """Test valid Qdrant configuration"""
        config = QdrantConfig(
            host="localhost",
            port=6333,
            api_key=None,
            timeout=30
        )
        assert config.host == "localhost"
        assert config.port == 6333
        assert config.timeout == 30
        
    def test_invalid_port(self):
        """Test invalid port number"""
        with pytest.raises(ValueError):
            QdrantConfig(host="localhost", port=70000)
            
    def test_empty_host(self):
        """Test empty host string"""
        with pytest.raises(ValueError):
            QdrantConfig(host="", port=6333)


class TestModelConfig:
    """Test model configuration validation"""
    
    def test_valid_config(self):
        """Test valid model configuration"""
        config = ModelConfig(
            video_model="mobilenet_v3_small",
            audio_model="panns_cnn14",
            device="cpu"
        )
        assert config.video_model == "mobilenet_v3_small"
        assert config.device == "cpu"
        
    def test_invalid_device(self):
        """Test invalid device"""
        with pytest.raises(ValueError):
            ModelConfig(
                video_model="mobilenet_v3_small",
                audio_model="panns_cnn14",
                device="gpu"
            )


class TestThresholdConfig:
    """Test threshold configuration validation"""
    
    def test_valid_config(self):
        """Test valid threshold configuration"""
        config = ThresholdConfig(
            video_initial=0.7,
            audio_initial=0.65,
            sensor_initial=2.5,
            learning_rate=0.05
        )
        assert config.video_initial == 0.7
        assert config.learning_rate == 0.05
        
    def test_invalid_threshold_range(self):
        """Test threshold out of valid range"""
        with pytest.raises(ValueError):
            ThresholdConfig(
                video_initial=1.5,  # > 1.0
                audio_initial=0.65,
                sensor_initial=2.5
            )


class TestLoggingConfig:
    """Test logging configuration validation"""
    
    def test_valid_config(self):
        """Test valid logging configuration"""
        config = LoggingConfig(
            level="INFO",
            format="json",
            log_dir="logs"
        )
        assert config.level == "INFO"
        assert config.format == "json"
        
    def test_invalid_level(self):
        """Test invalid log level"""
        with pytest.raises(ValueError):
            LoggingConfig(level="INVALID", format="json", log_dir="logs")
            
    def test_invalid_format(self):
        """Test invalid log format"""
        with pytest.raises(ValueError):
            LoggingConfig(level="INFO", format="xml", log_dir="logs")


class TestSystemConfig:
    """Test system configuration loading from environment"""
    
    def test_from_env_missing_required(self):
        """Test loading config with missing required variables"""
        # Clear environment
        os.environ.pop("QDRANT_HOST", None)
        os.environ.pop("QDRANT_PORT", None)
        
        with pytest.raises(ValueError, match="QDRANT_HOST"):
            SystemConfig.from_env()
            
    def test_from_env_valid(self, monkeypatch):
        """Test loading valid config from environment"""
        monkeypatch.setenv("QDRANT_HOST", "localhost")
        monkeypatch.setenv("QDRANT_PORT", "6333")
        
        config = SystemConfig.from_env()
        assert config.qdrant.host == "localhost"
        assert config.qdrant.port == 6333
        
    def test_from_env_invalid_port(self, monkeypatch):
        """Test loading config with invalid port"""
        monkeypatch.setenv("QDRANT_HOST", "localhost")
        monkeypatch.setenv("QDRANT_PORT", "not_a_number")
        
        with pytest.raises(ValueError, match="must be an integer"):
            SystemConfig.from_env()
