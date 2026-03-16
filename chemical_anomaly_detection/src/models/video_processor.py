"""Video processor with MobileNetV3-Small for frame embedding generation"""

import asyncio
import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple
from torchvision import models, transforms
from PIL import Image
import cv2


logger = logging.getLogger(__name__)


class VideoProcessor:
    """
    Processes video frames using MobileNetV3-Small for embedding generation
    
    Model: MobileNetV3-Small (pre-trained on ImageNet)
    - Latency: ~15ms per frame on CPU
    - Accuracy: 67.7% ImageNet top-1
    - Size: 2.5M parameters (927K after feature extraction)
    - Output: 512-dimensional embedding (projected from 576-dim avgpool)
    
    Detection targets:
        - Gas plumes
        - Sparks
        - PPE violations
        - Human panic behavior
    """
    
    def __init__(
        self,
        model_name: str = "mobilenet_v3_small",
        device: str = "cpu",
        timeout: float = 1.0
    ):
        """
        Initialize video processor
        
        Args:
            model_name: Model to use (default: mobilenet_v3_small)
            device: Device for inference (cpu or cuda)
            timeout: Maximum processing time per frame in seconds
        """
        self.model_name = model_name
        self.device = device
        self.timeout = timeout
        
        # Load model
        self.model = self._load_model()
        self.model.eval()  # Set to evaluation mode
        
        # Preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet means
                std=[0.229, 0.224, 0.225]     # ImageNet stds
            )
        ])
        
        logger.info(
            f"Initialized VideoProcessor: model={model_name}, "
            f"device={device}, timeout={timeout}s"
        )
        
    def _load_model(self) -> nn.Module:
        """
        Load pre-trained MobileNetV3-Small model
        
        Returns:
            Modified model with embedding extraction
        """
        if self.model_name == "mobilenet_v3_small":
            # Load pre-trained MobileNetV3-Small
            base_model = models.mobilenet_v3_small(weights='DEFAULT')
            
            # MobileNetV3-Small structure:
            # - features: convolutional layers
            # - avgpool: adaptive average pooling (outputs 576-dim)
            # - classifier: Linear(576, 1000)
            
            # We want 512-dim output, so add a projection layer
            
            # Create a wrapper to extract features and project to 512-dim
            class FeatureExtractor(nn.Module):
                def __init__(self, base_model):
                    super().__init__()
                    self.features = base_model.features
                    self.avgpool = base_model.avgpool
                    # Project from 576 to 512 dimensions
                    self.projection = nn.Linear(576, 512)
                    # Initialize projection layer
                    nn.init.xavier_uniform_(self.projection.weight)
                    nn.init.zeros_(self.projection.bias)
                    
                def forward(self, x):
                    x = self.features(x)
                    x = self.avgpool(x)
                    x = torch.flatten(x, 1)
                    x = self.projection(x)
                    return x
                    
            model = FeatureExtractor(base_model)
            
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
            
        # Move to device
        model = model.to(self.device)
        
        return model
        
    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """
        Preprocess video frame for model input
        
        Args:
            frame: Frame as numpy array (H, W, C) in BGR format
            
        Returns:
            Preprocessed tensor of shape (1, 3, 224, 224)
            
        Raises:
            ValueError: If frame shape is invalid
        """
        # Validate frame shape
        if len(frame.shape) != 3:
            raise ValueError(f"Expected 3D frame (H, W, C), got shape {frame.shape}")
            
        if frame.shape[2] != 3:
            raise ValueError(f"Expected 3 channels, got {frame.shape[2]}")
            
        # Convert BGR to RGB (OpenCV uses BGR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        image = Image.fromarray(frame_rgb)
        
        # Apply transforms
        tensor = self.transform(image)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor.to(self.device)
        
    def _extract_embedding(self, frame_tensor: torch.Tensor) -> np.ndarray:
        """
        Extract embedding from preprocessed frame
        
        Args:
            frame_tensor: Preprocessed frame tensor (1, 3, 224, 224)
            
        Returns:
            Embedding numpy array of shape (512,)
        """
        with torch.no_grad():
            # Forward pass through model
            embedding = self.model(frame_tensor)
            
            # Flatten if needed
            if len(embedding.shape) > 2:
                embedding = embedding.flatten(start_dim=1)
                
        # Convert to numpy and remove batch dimension
        return embedding.cpu().numpy().squeeze(0)
        
    async def process_frame(
        self,
        frame: np.ndarray,
        retry_on_failure: bool = True,
        max_retries: int = 2
    ) -> Optional[np.ndarray]:
        """
        Process video frame and generate embedding
        
        Args:
            frame: Frame as numpy array (H, W, C) in BGR format
            retry_on_failure: Whether to retry on failure
            max_retries: Maximum number of retries
            
        Returns:
            Embedding numpy array of shape (512,) or None if processing fails
        """
        for attempt in range(max_retries if retry_on_failure else 1):
            try:
                # Run preprocessing and embedding extraction in executor
                loop = asyncio.get_event_loop()
                
                # Preprocess frame
                frame_tensor = await asyncio.wait_for(
                    loop.run_in_executor(None, self._preprocess_frame, frame),
                    timeout=self.timeout / 2
                )
                
                # Extract embedding
                embedding = await asyncio.wait_for(
                    loop.run_in_executor(None, self._extract_embedding, frame_tensor),
                    timeout=self.timeout / 2
                )
                
                logger.debug(
                    f"Generated video embedding: shape={embedding.shape}, "
                    f"mean={embedding.mean():.4f}, std={embedding.std():.4f}"
                )
                
                return embedding
                
            except asyncio.TimeoutError:
                logger.warning(
                    f"Frame processing timeout (attempt {attempt + 1}/{max_retries})"
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.1)  # Brief delay before retry
                    
            except ValueError as e:
                logger.error(f"Invalid frame: {e}")
                return None  # Don't retry for invalid frames
                
            except Exception as e:
                logger.error(
                    f"Frame processing failed (attempt {attempt + 1}/{max_retries}): {e}"
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.1)
                    
        # All retries failed
        logger.error(f"Failed to process frame after {max_retries} attempts")
        return None
        
    async def process_frames_batch(
        self,
        frames: list[np.ndarray],
        retry_on_failure: bool = True
    ) -> list[Optional[np.ndarray]]:
        """
        Process multiple frames in parallel
        
        Args:
            frames: List of frames as numpy arrays
            retry_on_failure: Whether to retry on failure
            
        Returns:
            List of embedding arrays (or None for failed frames)
        """
        tasks = [
            self.process_frame(frame, retry_on_failure)
            for frame in frames
        ]
        
        embeddings = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to None
        results = []
        for i, emb in enumerate(embeddings):
            if isinstance(emb, Exception):
                logger.error(f"Failed to process frame {i}: {emb}")
                results.append(None)
            else:
                results.append(emb)
                
        return results
        
    def validate_frame(self, frame: np.ndarray) -> Tuple[bool, Optional[str]]:
        """
        Validate frame shape and format
        
        Args:
            frame: Frame as numpy array
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if frame is None
        if frame is None:
            return False, "Frame is None"
            
        # Check if frame is numpy array
        if not isinstance(frame, np.ndarray):
            return False, f"Frame must be numpy array, got {type(frame)}"
            
        # Check dimensions
        if len(frame.shape) != 3:
            return False, f"Expected 3D frame (H, W, C), got shape {frame.shape}"
            
        # Check channels
        if frame.shape[2] != 3:
            return False, f"Expected 3 channels, got {frame.shape[2]}"
            
        # Check if frame is empty
        if frame.size == 0:
            return False, "Frame is empty"
            
        # Check data type
        if frame.dtype not in [np.uint8, np.float32, np.float64]:
            return False, f"Unsupported frame dtype: {frame.dtype}"
            
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
            "embedding_dim": 512,  # Projected from 576-dim avgpool
            "input_size": (224, 224),
            "timeout": self.timeout,
            "parameters": sum(p.numel() for p in self.model.parameters())
        }
