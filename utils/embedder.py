from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List
import logging
import torch

logger = logging.getLogger(__name__)


class ChemicalEmbedder:
    """Generates embeddings for chemical safety text using SentenceTransformers"""
    
    def __init__(self, model_name: str = "BAAI/bge-base-en"):
        """
        Initialize embedding model
        
        Args:
            model_name: HuggingFace model identifier
        """
        self.model_name = model_name
        
        # Use GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading embedding model {model_name} on {device}")
        
        self.model = SentenceTransformer(model_name, device=device)
        
        # Verify output dimension
        test_embedding = self.model.encode("test")
        self.dimension = len(test_embedding)
        logger.info(f"Model loaded. Embedding dimension: {self.dimension}")
        
        if self.dimension != 768:
            logger.warning(f"Expected 768 dimensions, got {self.dimension}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector as numpy array
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for multiple texts efficiently
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            
        Returns:
            Array of embeddings
        """
        logger.info(f"Embedding {len(texts)} texts in batches of {batch_size}")
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        
        return embeddings
    
    def get_dimension(self) -> int:
        """
        Get embedding dimension
        
        Returns:
            Embedding dimension
        """
        return self.dimension