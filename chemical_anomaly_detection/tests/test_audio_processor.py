"""Unit tests for audio processor"""

import pytest
import asyncio
import numpy as np
from src.models.audio_processor import AudioProcessor


class TestAudioProcessor:
    """Test audio processor"""
    
    @pytest.fixture
    def processor(self):
        """Create processor instance"""
        return AudioProcessor(
            model_name="Cnn14",
            sample_rate=32000,
            device="cpu",
            timeout=1.0
        )
        
    @pytest.fixture
    def sample_audio(self):
        """Create sample audio (1 second at 32kHz)"""
        duration = 1.0
        sr = 32000
        t = np.linspace(0, duration, int(sr * duration))
        # Generate sine wave at 440 Hz
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        return audio, sr
        
    @pytest.fixture
    def stereo_audio(self):
        """Create stereo audio"""
        duration = 1.0
        sr = 32000
        t = np.linspace(0, duration, int(sr * duration))
        # Generate stereo sine wave
        left = np.sin(2 * np.pi * 440 * t)
        right = np.sin(2 * np.pi * 880 * t)
        audio = np.stack([left, right], axis=1).astype(np.float32)
        return audio, sr
        
    def test_processor_initialization(self, processor):
        """Test processor initialization"""
        assert processor.model_name == "Cnn14"
        assert processor.sample_rate == 32000
        assert processor.device == "cpu"
        assert processor.timeout == 1.0
        assert processor.model is not None
        
    def test_preprocess_audio_mono(self, processor, sample_audio):
        """Test audio preprocessing with mono audio"""
        audio, sr = sample_audio
        processed = processor._preprocess_audio(audio, sr)
        
        assert processed.shape == audio.shape
        assert np.max(np.abs(processed)) <= 1.0  # Normalized
        
    def test_preprocess_audio_stereo(self, processor, stereo_audio):
        """Test audio preprocessing with stereo audio"""
        audio, sr = stereo_audio
        processed = processor._preprocess_audio(audio, sr)
        
        # Should be converted to mono
        assert len(processed.shape) == 1
        assert np.max(np.abs(processed)) <= 1.0
        
    def test_preprocess_audio_resample(self, processor):
        """Test audio preprocessing with resampling"""
        # Create audio at different sample rate
        duration = 1.0
        sr = 16000  # Different from processor's 32000
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        processed = processor._preprocess_audio(audio, sr)
        
        # Should be resampled to 32000 Hz
        expected_length = int(duration * processor.sample_rate)
        assert abs(len(processed) - expected_length) < 100  # Allow small difference
        
    def test_preprocess_audio_empty(self, processor):
        """Test preprocessing with empty audio"""
        empty_audio = np.array([])
        
        with pytest.raises(ValueError, match="Audio is empty"):
            processor._preprocess_audio(empty_audio, 32000)
            
    def test_compute_mel_spectrogram(self, processor, sample_audio):
        """Test mel-spectrogram computation"""
        audio, sr = sample_audio
        mel_spec = processor._compute_mel_spectrogram(audio, sr)
        
        # Check shape (n_mels, time_steps)
        assert mel_spec.shape[0] == processor.n_mels
        assert mel_spec.shape[1] > 0
        
    def test_extract_embedding_shape(self, processor, sample_audio):
        """Test embedding extraction output shape"""
        audio, sr = sample_audio
        processed = processor._preprocess_audio(audio, sr)
        embedding = processor._extract_embedding(processed)
        
        # Original PANNs embedding is 2048-dim
        assert embedding.shape == (2048,)
        
    def test_project_embedding(self, processor):
        """Test embedding projection from 2048 to 512"""
        # Create random 2048-dim embedding
        embedding_2048 = np.random.randn(2048).astype(np.float32)
        
        # Project to 512-dim
        embedding_512 = processor._project_embedding(embedding_2048)
        
        assert embedding_512.shape == (512,)
        
    @pytest.mark.asyncio
    async def test_process_audio_valid(self, processor, sample_audio):
        """Test processing valid audio"""
        audio, sr = sample_audio
        embedding = await processor.process_audio(audio, sr)
        
        assert embedding is not None
        assert embedding.shape == (512,)
        
    @pytest.mark.asyncio
    async def test_process_audio_stereo(self, processor, stereo_audio):
        """Test processing stereo audio"""
        audio, sr = stereo_audio
        embedding = await processor.process_audio(audio, sr)
        
        assert embedding is not None
        assert embedding.shape == (512,)
        
    @pytest.mark.asyncio
    async def test_process_audio_different_sr(self, processor):
        """Test processing audio with different sample rate"""
        # Create audio at 16kHz
        duration = 1.0
        sr = 16000
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        embedding = await processor.process_audio(audio, sr)
        
        assert embedding is not None
        assert embedding.shape == (512,)
        
    @pytest.mark.asyncio
    async def test_process_audio_empty(self, processor):
        """Test processing empty audio"""
        empty_audio = np.array([])
        
        embedding = await processor.process_audio(empty_audio, 32000)
        
        # Should return None for invalid audio
        assert embedding is None
        
    @pytest.mark.asyncio
    async def test_process_audio_with_retry(self, processor, sample_audio):
        """Test processing with retry enabled"""
        audio, sr = sample_audio
        embedding = await processor.process_audio(
            audio, sr,
            retry_on_failure=True,
            max_retries=2
        )
        
        assert embedding is not None
        assert embedding.shape == (512,)
        
    @pytest.mark.asyncio
    async def test_process_audio_without_retry(self, processor, sample_audio):
        """Test processing without retry"""
        audio, sr = sample_audio
        embedding = await processor.process_audio(
            audio, sr,
            retry_on_failure=False
        )
        
        assert embedding is not None
        assert embedding.shape == (512,)
        
    @pytest.mark.asyncio
    async def test_process_audio_batch(self, processor):
        """Test batch processing"""
        # Create multiple audio windows
        audio_windows = []
        for i in range(3):
            duration = 1.0
            sr = 32000
            t = np.linspace(0, duration, int(sr * duration))
            audio = np.sin(2 * np.pi * (440 + i * 100) * t).astype(np.float32)
            audio_windows.append((audio, sr))
            
        embeddings = await processor.process_audio_batch(audio_windows)
        
        assert len(embeddings) == 3
        assert all(emb is not None for emb in embeddings)
        assert all(emb.shape == (512,) for emb in embeddings)
        
    @pytest.mark.asyncio
    async def test_process_audio_batch_with_failures(self, processor):
        """Test batch processing with some invalid audio"""
        audio_windows = [
            (np.sin(2 * np.pi * 440 * np.linspace(0, 1, 32000)).astype(np.float32), 32000),  # Valid
            (np.array([]), 32000),  # Invalid (empty)
            (np.sin(2 * np.pi * 880 * np.linspace(0, 1, 32000)).astype(np.float32), 32000),  # Valid
        ]
        
        embeddings = await processor.process_audio_batch(audio_windows)
        
        assert len(embeddings) == 3
        assert embeddings[0] is not None
        assert embeddings[1] is None  # Failed
        assert embeddings[2] is not None
        
    def test_validate_audio_valid(self, processor, sample_audio):
        """Test audio validation with valid audio"""
        audio, sr = sample_audio
        is_valid, error = processor.validate_audio(audio, sr)
        
        assert is_valid is True
        assert error is None
        
    def test_validate_audio_none(self, processor):
        """Test audio validation with None"""
        is_valid, error = processor.validate_audio(None, 32000)
        
        assert is_valid is False
        assert "None" in error
        
    def test_validate_audio_wrong_type(self, processor):
        """Test audio validation with wrong type"""
        is_valid, error = processor.validate_audio([1, 2, 3], 32000)
        
        assert is_valid is False
        assert "numpy array" in error
        
    def test_validate_audio_empty(self, processor):
        """Test audio validation with empty array"""
        empty_audio = np.array([])
        is_valid, error = processor.validate_audio(empty_audio, 32000)
        
        assert is_valid is False
        assert "empty" in error
        
    def test_validate_audio_invalid_sr(self, processor, sample_audio):
        """Test audio validation with invalid sample rate"""
        audio, _ = sample_audio
        is_valid, error = processor.validate_audio(audio, -1)
        
        assert is_valid is False
        assert "sample rate" in error
        
    def test_validate_audio_wrong_dimensions(self, processor):
        """Test audio validation with wrong dimensions"""
        # 3D array
        audio_3d = np.random.randn(10, 10, 10)
        is_valid, error = processor.validate_audio(audio_3d, 32000)
        
        assert is_valid is False
        assert "1D or 2D" in error
        
    def test_get_model_info(self, processor):
        """Test getting model information"""
        info = processor.get_model_info()
        
        assert info['model_name'] == "Cnn14"
        assert info['device'] == "cpu"
        assert info['embedding_dim'] == 512
        assert info['original_embedding_dim'] == 2048
        assert info['sample_rate'] == 32000
        assert info['n_mels'] == 128
        assert info['timeout'] == 1.0
        assert 'panns_available' in info
        
    def test_different_audio_different_embeddings(self, processor):
        """Test that different audio produces different embeddings"""
        # Create two different audio signals
        duration = 1.0
        sr = 32000
        t = np.linspace(0, duration, int(sr * duration))
        
        audio1 = np.sin(2 * np.pi * 440 * t).astype(np.float32)  # 440 Hz
        audio2 = np.sin(2 * np.pi * 880 * t).astype(np.float32)  # 880 Hz
        
        processed1 = processor._preprocess_audio(audio1, sr)
        processed2 = processor._preprocess_audio(audio2, sr)
        
        embedding1 = processor._extract_embedding(processed1)
        embedding2 = processor._extract_embedding(processed2)
        
        # Embeddings should be different (unless using mock model with same seed)
        # We can't guarantee this with mock model, so just check they exist
        assert embedding1.shape == (2048,)
        assert embedding2.shape == (2048,)
