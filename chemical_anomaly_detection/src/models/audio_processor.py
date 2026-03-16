"""Audio processor with PANNs CNN14 for audio embedding generation"""

import asyncio
import logging
import os
import numpy as np
import torch
import torch.nn as nn
import librosa
from typing import Optional, Tuple
try:
    from panns_inference import AudioTagging
except ImportError:
    AudioTagging = None


logger = logging.getLogger(__name__)


class AudioProcessor:
    """
    Processes audio using PANNs CNN14 for embedding generation
    
    Model: PANNs CNN14 (pre-trained on AudioSet)
    - Trained on AudioSet (527 classes including industrial sounds)
    - Latency: ~50ms per 1-second window
    - Original embedding: 2048-dim
    - Output: 512-dim (projected from 2048-dim)
    
    Detection targets:
        - Hissing sounds (gas leaks)
        - Alarm patterns
        - Silence anomalies
        - Mechanical sounds
    """
    
    def __init__(
        self,
        model_name: str = "Cnn14",
        sample_rate: int = 32000,
        device: str = "cpu",
        timeout: float = 1.0,
        checkpoint_path: Optional[str] = None
    ):
        """
        Initialize audio processor
        
        Args:
            model_name: Model to use (default: Cnn14)
            sample_rate: Audio sample rate in Hz (default: 32000)
            device: Device for inference (cpu or cuda)
            timeout: Maximum processing time per audio window in seconds
            checkpoint_path: Path to PANNs checkpoint file (optional, will load from 
                           PANNS_CHECKPOINT_PATH env var if None, will download if not found)
        """
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.device = device
        self.timeout = timeout
        
        # Load checkpoint path from environment if not provided
        if checkpoint_path is None:
            checkpoint_path = os.getenv("PANNS_CHECKPOINT_PATH")
        
        self.checkpoint_path = checkpoint_path
        
        # Load model
        self.model = self._load_model()
        
        # Mel-spectrogram parameters
        self.n_mels = 128
        self.hop_length = 320
        self.n_fft = 1024
        
        # Projection layer to reduce from 2048 to 512 dimensions
        self.projection = nn.Linear(2048, 512)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        self.projection = self.projection.to(device)
        self.projection.eval()
        
        logger.info(
            f"Initialized AudioProcessor: model={model_name}, "
            f"sample_rate={sample_rate}Hz, device={device}, timeout={timeout}s"
        )
        
    def _load_model(self):
        """
        Load pre-trained PANNs CNN14 model
        
        Returns:
            Model instance or None if PANNs not available
        """
        if AudioTagging is None:
            logger.warning(
                "PANNs not available. Install with: pip install panns_inference"
            )
            # Return a mock model for testing
            return self._create_mock_model()
            
        try:
            # Initialize PANNs model
            model = AudioTagging(
                checkpoint_path=self.checkpoint_path,  # Use provided path or download
                device=self.device
            )
            model.model.eval()
            logger.info(f"Loaded PANNs model from: {self.checkpoint_path or 'auto-download'}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load PANNs model: {e}")
            logger.warning("Falling back to mock model")
            return self._create_mock_model()
            
    def _create_mock_model(self):
        """
        Create a mock model for testing when PANNs is not available
        
        Returns:
            Mock model that generates random embeddings
        """
        class MockPANNs:
            def __init__(self, device):
                self.device = device
                
            def inference(self, audio):
                # Return random 2048-dim embedding
                batch_size = audio.shape[0] if len(audio.shape) > 1 else 1
                return {
                    'embedding': np.random.randn(batch_size, 2048).astype(np.float32)
                }
                
        logger.warning("Using mock PANNs model (random embeddings)")
        return MockPANNs(self.device)
        
    def _compute_mel_spectrogram(
        self,
        audio: np.ndarray,
        sr: int
    ) -> np.ndarray:
        """
        Compute mel-spectrogram from audio
        
        Args:
            audio: Audio waveform as numpy array
            sr: Sample rate
            
        Returns:
            Mel-spectrogram as numpy array
        """
        # Resample if needed
        if sr != self.sample_rate:
            audio = librosa.resample(
                audio,
                orig_sr=sr,
                target_sr=self.sample_rate
            )
            
        # Compute mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=50,  # Minimum frequency (Hz)
            fmax=14000  # Maximum frequency (Hz)
        )
        
        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec_db
        
    def _preprocess_audio(
        self,
        audio: np.ndarray,
        sr: int
    ) -> np.ndarray:
        """
        Preprocess audio for model input
        
        Args:
            audio: Audio waveform as numpy array
            sr: Sample rate
            
        Returns:
            Preprocessed audio array
            
        Raises:
            ValueError: If audio is invalid
        """
        # Validate audio
        if audio is None or len(audio) == 0:
            raise ValueError("Audio is empty")
            
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
            
        # Resample if needed
        if sr != self.sample_rate:
            audio = librosa.resample(
                audio,
                orig_sr=sr,
                target_sr=self.sample_rate
            )
            
        # Normalize audio
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
            
        return audio
        
    def _extract_embedding(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract embedding from preprocessed audio
        
        Args:
            audio: Preprocessed audio array
            
        Returns:
            Embedding numpy array of shape (2048,)
        """
        # PANNs expects audio in specific format
        if isinstance(self.model, AudioTagging):
            # Use PANNs inference
            # PANNs returns a tuple: (clipwise_output, embedding)
            result = self.model.inference(audio[np.newaxis, :])
            
            # Handle both dict and tuple return types
            if isinstance(result, tuple):
                # result is (clipwise_output, embedding)
                embedding = result[1][0]  # Get embedding, remove batch dimension
            elif isinstance(result, dict):
                # result is {'embedding': ...}
                embedding = result['embedding'][0]
            else:
                raise ValueError(f"Unexpected PANNs result type: {type(result)}")
        else:
            # Use mock model
            result = self.model.inference(audio)
            embedding = result['embedding'][0]
            
        return embedding
        
    def _project_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """
        Project embedding from 2048-dim to 512-dim
        
        Args:
            embedding: Original embedding (2048-dim)
            
        Returns:
            Projected embedding (512-dim)
        """
        # Convert to tensor
        embedding_tensor = torch.from_numpy(embedding).unsqueeze(0).to(self.device)
        
        # Project
        with torch.no_grad():
            projected = self.projection(embedding_tensor)
            
        # Convert back to numpy
        return projected.cpu().numpy().squeeze(0)
        
    async def process_audio(
        self,
        audio: np.ndarray,
        sr: int,
        retry_on_failure: bool = True,
        max_retries: int = 2
    ) -> Optional[np.ndarray]:
        """
        Process audio window and generate embedding
        
        Args:
            audio: Audio waveform as numpy array
            sr: Sample rate in Hz
            retry_on_failure: Whether to retry on failure
            max_retries: Maximum number of retries
            
        Returns:
            Embedding numpy array of shape (512,) or None if processing fails
        """
        for attempt in range(max_retries if retry_on_failure else 1):
            try:
                # Run preprocessing and embedding extraction in executor
                loop = asyncio.get_event_loop()
                
                # Preprocess audio
                audio_processed = await asyncio.wait_for(
                    loop.run_in_executor(None, self._preprocess_audio, audio, sr),
                    timeout=self.timeout / 3
                )
                
                # Extract embedding
                embedding = await asyncio.wait_for(
                    loop.run_in_executor(None, self._extract_embedding, audio_processed),
                    timeout=self.timeout / 3
                )
                
                # Project to 512-dim
                embedding_512 = await asyncio.wait_for(
                    loop.run_in_executor(None, self._project_embedding, embedding),
                    timeout=self.timeout / 3
                )
                
                logger.debug(
                    f"Generated audio embedding: shape={embedding_512.shape}, "
                    f"mean={embedding_512.mean():.4f}, std={embedding_512.std():.4f}"
                )
                
                return embedding_512
                
            except asyncio.TimeoutError:
                logger.warning(
                    f"Audio processing timeout (attempt {attempt + 1}/{max_retries})"
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.1)
                    
            except ValueError as e:
                logger.error(f"Invalid audio: {e}")
                return None  # Don't retry for invalid audio
                
            except Exception as e:
                logger.error(
                    f"Audio processing failed (attempt {attempt + 1}/{max_retries}): {e}"
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.1)
                    
        # All retries failed
        logger.error(f"Failed to process audio after {max_retries} attempts")
        return None
        
    async def process_audio_batch(
        self,
        audio_windows: list[Tuple[np.ndarray, int]],
        retry_on_failure: bool = True
    ) -> list[Optional[np.ndarray]]:
        """
        Process multiple audio windows in parallel
        
        Args:
            audio_windows: List of (audio, sample_rate) tuples
            retry_on_failure: Whether to retry on failure
            
        Returns:
            List of embedding arrays (or None for failed windows)
        """
        tasks = [
            self.process_audio(audio, sr, retry_on_failure)
            for audio, sr in audio_windows
        ]
        
        embeddings = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to None
        results = []
        for i, emb in enumerate(embeddings):
            if isinstance(emb, Exception):
                logger.error(f"Failed to process audio window {i}: {emb}")
                results.append(None)
            else:
                results.append(emb)
                
        return results
        
    def validate_audio(
        self,
        audio: np.ndarray,
        sr: int
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate audio format
        
        Args:
            audio: Audio waveform
            sr: Sample rate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if audio is None
        if audio is None:
            return False, "Audio is None"
            
        # Check if audio is numpy array
        if not isinstance(audio, np.ndarray):
            return False, f"Audio must be numpy array, got {type(audio)}"
            
        # Check if audio is empty
        if audio.size == 0:
            return False, "Audio is empty"
            
        # Check dimensions (should be 1D or 2D)
        if len(audio.shape) > 2:
            return False, f"Audio must be 1D or 2D, got shape {audio.shape}"
            
        # Check sample rate
        if sr <= 0:
            return False, f"Invalid sample rate: {sr}"
            
        # Check data type
        if audio.dtype not in [np.float32, np.float64, np.int16, np.int32]:
            return False, f"Unsupported audio dtype: {audio.dtype}"
            
        return True, None
        
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "embedding_dim": 512,
            "original_embedding_dim": 2048,
            "sample_rate": self.sample_rate,
            "n_mels": self.n_mels,
            "timeout": self.timeout,
            "panns_available": AudioTagging is not None
        }
