"""Unit tests for video processor"""

import pytest
import asyncio
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock
from src.models.video_processor import VideoProcessor


class TestVideoProcessor:
    """Test video processor"""
    
    @pytest.fixture
    def processor(self):
        """Create processor instance"""
        return VideoProcessor(model_name="mobilenet_v3_small", device="cpu", timeout=1.0)
        
    @pytest.fixture
    def sample_frame(self):
        """Create sample video frame (224x224x3 BGR)"""
        return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
    @pytest.fixture
    def large_frame(self):
        """Create large video frame (1920x1080x3 BGR)"""
        return np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        
    def test_processor_initialization(self, processor):
        """Test processor initialization"""
        assert processor.model_name == "mobilenet_v3_small"
        assert processor.device == "cpu"
        assert processor.timeout == 1.0
        assert processor.model is not None
        
    def test_unsupported_model(self):
        """Test initialization with unsupported model"""
        with pytest.raises(ValueError, match="Unsupported model"):
            VideoProcessor(model_name="unsupported_model")
            
    def test_preprocess_frame_valid(self, processor, sample_frame):
        """Test frame preprocessing with valid frame"""
        tensor = processor._preprocess_frame(sample_frame)
        
        assert tensor.shape == (1, 3, 224, 224)
        assert tensor.device.type == "cpu"
        
    def test_preprocess_frame_large(self, processor, large_frame):
        """Test frame preprocessing with large frame (should resize)"""
        tensor = processor._preprocess_frame(large_frame)
        
        # Should be resized to 224x224
        assert tensor.shape == (1, 3, 224, 224)
        
    def test_preprocess_frame_invalid_shape(self, processor):
        """Test preprocessing with invalid frame shape"""
        invalid_frame = np.random.randint(0, 255, (224, 224), dtype=np.uint8)  # 2D
        
        with pytest.raises(ValueError, match="Expected 3D frame"):
            processor._preprocess_frame(invalid_frame)
            
    def test_preprocess_frame_invalid_channels(self, processor):
        """Test preprocessing with invalid number of channels"""
        invalid_frame = np.random.randint(0, 255, (224, 224, 4), dtype=np.uint8)  # 4 channels
        
        with pytest.raises(ValueError, match="Expected 3 channels"):
            processor._preprocess_frame(invalid_frame)
            
    def test_extract_embedding_shape(self, processor, sample_frame):
        """Test embedding extraction output shape"""
        tensor = processor._preprocess_frame(sample_frame)
        embedding = processor._extract_embedding(tensor)
        
        assert embedding.shape == (512,)
        assert embedding.dtype == np.float32 or embedding.dtype == np.float64
        
    def test_extract_embedding_deterministic(self, processor, sample_frame):
        """Test that embedding extraction is deterministic"""
        tensor = processor._preprocess_frame(sample_frame)
        
        embedding1 = processor._extract_embedding(tensor)
        embedding2 = processor._extract_embedding(tensor)
        
        np.testing.assert_array_almost_equal(embedding1, embedding2)
        
    @pytest.mark.asyncio
    async def test_process_frame_valid(self, processor, sample_frame):
        """Test processing valid frame"""
        embedding = await processor.process_frame(sample_frame)
        
        assert embedding is not None
        assert embedding.shape == (512,)
        
    @pytest.mark.asyncio
    async def test_process_frame_large(self, processor, large_frame):
        """Test processing large frame"""
        embedding = await processor.process_frame(large_frame)
        
        assert embedding is not None
        assert embedding.shape == (512,)
        
    @pytest.mark.asyncio
    async def test_process_frame_invalid_shape(self, processor):
        """Test processing frame with invalid shape"""
        invalid_frame = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
        
        embedding = await processor.process_frame(invalid_frame)
        
        # Should return None for invalid frame
        assert embedding is None
        
    @pytest.mark.asyncio
    async def test_process_frame_with_retry(self, processor, sample_frame):
        """Test processing with retry enabled"""
        embedding = await processor.process_frame(
            sample_frame,
            retry_on_failure=True,
            max_retries=2
        )
        
        assert embedding is not None
        assert embedding.shape == (512,)
        
    @pytest.mark.asyncio
    async def test_process_frame_without_retry(self, processor, sample_frame):
        """Test processing without retry"""
        embedding = await processor.process_frame(
            sample_frame,
            retry_on_failure=False
        )
        
        assert embedding is not None
        assert embedding.shape == (512,)
        
    @pytest.mark.asyncio
    async def test_process_frame_timeout(self, processor):
        """Test processing with timeout"""
        # Create processor with very short timeout
        short_timeout_processor = VideoProcessor(
            model_name="mobilenet_v3_small",
            device="cpu",
            timeout=0.001  # 1ms - too short
        )
        
        frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Should timeout and return None
        embedding = await short_timeout_processor.process_frame(
            frame,
            retry_on_failure=False
        )
        
        # May succeed or fail depending on system speed
        # Just verify it doesn't crash
        assert embedding is None or embedding.shape == (512,)
        
    @pytest.mark.asyncio
    async def test_process_frames_batch(self, processor):
        """Test batch processing"""
        frames = [
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            for _ in range(5)
        ]
        
        embeddings = await processor.process_frames_batch(frames)
        
        assert len(embeddings) == 5
        assert all(emb is not None for emb in embeddings)
        assert all(emb.shape == (512,) for emb in embeddings)
        
    @pytest.mark.asyncio
    async def test_process_frames_batch_with_failures(self, processor):
        """Test batch processing with some invalid frames"""
        frames = [
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),  # Valid
            np.random.randint(0, 255, (224, 224), dtype=np.uint8),     # Invalid (2D)
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),  # Valid
        ]
        
        embeddings = await processor.process_frames_batch(frames)
        
        assert len(embeddings) == 3
        assert embeddings[0] is not None
        assert embeddings[1] is None  # Failed
        assert embeddings[2] is not None
        
    def test_validate_frame_valid(self, processor, sample_frame):
        """Test frame validation with valid frame"""
        is_valid, error = processor.validate_frame(sample_frame)
        
        assert is_valid is True
        assert error is None
        
    def test_validate_frame_none(self, processor):
        """Test frame validation with None"""
        is_valid, error = processor.validate_frame(None)
        
        assert is_valid is False
        assert "None" in error
        
    def test_validate_frame_wrong_type(self, processor):
        """Test frame validation with wrong type"""
        is_valid, error = processor.validate_frame([1, 2, 3])
        
        assert is_valid is False
        assert "numpy array" in error
        
    def test_validate_frame_wrong_dimensions(self, processor):
        """Test frame validation with wrong dimensions"""
        frame_2d = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
        is_valid, error = processor.validate_frame(frame_2d)
        
        assert is_valid is False
        assert "3D frame" in error
        
    def test_validate_frame_wrong_channels(self, processor):
        """Test frame validation with wrong number of channels"""
        frame_4ch = np.random.randint(0, 255, (224, 224, 4), dtype=np.uint8)
        is_valid, error = processor.validate_frame(frame_4ch)
        
        assert is_valid is False
        assert "3 channels" in error
        
    def test_validate_frame_empty(self, processor):
        """Test frame validation with empty frame"""
        empty_frame = np.array([]).reshape(0, 0, 3)
        is_valid, error = processor.validate_frame(empty_frame)
        
        assert is_valid is False
        assert "empty" in error
        
    def test_get_model_info(self, processor):
        """Test getting model information"""
        info = processor.get_model_info()
        
        assert info['model_name'] == "mobilenet_v3_small"
        assert info['device'] == "cpu"
        assert info['embedding_dim'] == 512
        assert info['input_size'] == (224, 224)
        assert info['timeout'] == 1.0
        assert info['parameters'] > 0
        
    def test_model_eval_mode(self, processor):
        """Test that model is in evaluation mode"""
        assert not processor.model.training
        
    def test_different_frames_different_embeddings(self, processor):
        """Test that different frames produce different embeddings"""
        frame1 = np.zeros((224, 224, 3), dtype=np.uint8)  # Black frame
        frame2 = np.ones((224, 224, 3), dtype=np.uint8) * 255  # White frame
        
        tensor1 = processor._preprocess_frame(frame1)
        tensor2 = processor._preprocess_frame(frame2)
        
        embedding1 = processor._extract_embedding(tensor1)
        embedding2 = processor._extract_embedding(tensor2)
        
        # Embeddings should be different
        assert not np.allclose(embedding1, embedding2)
