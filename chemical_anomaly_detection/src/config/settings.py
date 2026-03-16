"""Configuration settings with Pydantic validation"""

import os
from typing import Dict, Optional
from pydantic import BaseModel, Field, field_validator
from pathlib import Path


class QdrantConfig(BaseModel):
    """Qdrant database configuration"""
    host: str = Field(..., description="Qdrant server host")
    port: int = Field(..., ge=1, le=65535, description="Qdrant server port")
    api_key: Optional[str] = Field(None, description="Optional API key")
    timeout: int = Field(30, ge=1, description="Connection timeout in seconds")
    
    @field_validator('host')
    @classmethod
    def validate_host(cls, v):
        if not v or v.strip() == "":
            raise ValueError("Qdrant host cannot be empty")
        return v


class ModelConfig(BaseModel):
    """Model configuration for embedding generation"""
    video_model: str = Field("mobilenet_v3_small", description="Video embedding model")
    audio_model: str = Field("panns_cnn14", description="Audio embedding model")
    sensor_model_path: Optional[str] = Field(None, description="Path to sensor embedding model")
    device: str = Field("cpu", description="Device for model inference (cpu/cuda)")
    
    @field_validator('device')
    @classmethod
    def validate_device(cls, v):
        if v not in ['cpu', 'cuda']:
            raise ValueError("Device must be 'cpu' or 'cuda'")
        return v


class ThresholdConfig(BaseModel):
    """Adaptive threshold configuration"""
    video_initial: float = Field(0.7, ge=0.0, le=1.0, description="Initial video threshold")
    audio_initial: float = Field(0.65, ge=0.0, le=1.0, description="Initial audio threshold")
    sensor_initial: float = Field(2.5, ge=0.0, description="Initial sensor threshold")
    learning_rate: float = Field(0.05, ge=0.0, le=1.0, description="Threshold adaptation rate")
    window_size: int = Field(100, ge=1, description="Feedback window size")


class AgentConfig(BaseModel):
    """Agent-specific configuration"""
    processing_interval: float = Field(1.0, ge=0.1, description="Processing interval in seconds")
    queue_max_depth: int = Field(1000, ge=1, description="Maximum queue depth")
    batch_mode_latency_threshold: float = Field(2.0, ge=0.0, description="Latency threshold for batch mode")


class LoggingConfig(BaseModel):
    """Logging configuration"""
    level: str = Field("INFO", description="Logging level")
    format: str = Field("json", description="Logging format (json/text)")
    log_dir: str = Field("logs", description="Directory for log files")
    
    @field_validator('level')
    @classmethod
    def validate_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()
        
    @field_validator('format')
    @classmethod
    def validate_format(cls, v):
        if v not in ['json', 'text']:
            raise ValueError("Log format must be 'json' or 'text'")
        return v


class SystemConfig(BaseModel):
    """Complete system configuration"""
    qdrant: QdrantConfig
    models: ModelConfig
    thresholds: ThresholdConfig
    agents: AgentConfig
    logging: LoggingConfig
    
    @classmethod
    def from_env(cls) -> "SystemConfig":
        """
        Load configuration from environment variables
        
        Returns:
            SystemConfig instance
            
        Raises:
            ValueError: If required environment variables are missing or invalid
        """
        # Qdrant configuration
        qdrant_host = os.getenv("QDRANT_HOST")
        qdrant_port = os.getenv("QDRANT_PORT")
        
        if not qdrant_host:
            raise ValueError("QDRANT_HOST environment variable is required")
        if not qdrant_port:
            raise ValueError("QDRANT_PORT environment variable is required")
            
        try:
            qdrant_port_int = int(qdrant_port)
        except ValueError:
            raise ValueError(f"QDRANT_PORT must be an integer, got: {qdrant_port}")
        
        qdrant_config = QdrantConfig(
            host=qdrant_host,
            port=qdrant_port_int,
            api_key=os.getenv("QDRANT_API_KEY"),
            timeout=int(os.getenv("QDRANT_TIMEOUT", "30"))
        )
        
        # Model configuration
        models_config = ModelConfig(
            video_model=os.getenv("VIDEO_MODEL", "mobilenet_v3_small"),
            audio_model=os.getenv("AUDIO_MODEL", "panns_cnn14"),
            sensor_model_path=os.getenv("SENSOR_MODEL_PATH"),
            device=os.getenv("DEVICE", "cpu")
        )
        
        # Threshold configuration
        thresholds_config = ThresholdConfig(
            video_initial=float(os.getenv("THRESHOLD_VIDEO", "0.7")),
            audio_initial=float(os.getenv("THRESHOLD_AUDIO", "0.65")),
            sensor_initial=float(os.getenv("THRESHOLD_SENSOR", "2.5")),
            learning_rate=float(os.getenv("THRESHOLD_LEARNING_RATE", "0.05")),
            window_size=int(os.getenv("THRESHOLD_WINDOW_SIZE", "100"))
        )
        
        # Agent configuration
        agents_config = AgentConfig(
            processing_interval=float(os.getenv("PROCESSING_INTERVAL", "1.0")),
            queue_max_depth=int(os.getenv("QUEUE_MAX_DEPTH", "1000")),
            batch_mode_latency_threshold=float(os.getenv("BATCH_MODE_LATENCY", "2.0"))
        )
        
        # Logging configuration
        logging_config = LoggingConfig(
            level=os.getenv("LOG_LEVEL", "INFO"),
            format=os.getenv("LOG_FORMAT", "json"),
            log_dir=os.getenv("LOG_DIR", "logs")
        )
        
        return cls(
            qdrant=qdrant_config,
            models=models_config,
            thresholds=thresholds_config,
            agents=agents_config,
            logging=logging_config
        )
    
    def validate_config(self) -> None:
        """
        Validate configuration and fail fast if invalid
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Check if sensor model path exists if provided
        if self.models.sensor_model_path:
            model_path = Path(self.models.sensor_model_path)
            if not model_path.exists():
                raise ValueError(
                    f"Sensor model path does not exist: {self.models.sensor_model_path}"
                )
        
        # Ensure log directory exists
        log_dir = Path(self.logging.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
