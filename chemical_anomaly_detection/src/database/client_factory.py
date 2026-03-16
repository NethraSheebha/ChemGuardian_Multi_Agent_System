"""Factory for creating Qdrant client based on environment configuration"""

import os
import logging
from typing import Optional
from qdrant_client import QdrantClient


logger = logging.getLogger(__name__)


def create_qdrant_client(
    mode: Optional[str] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
    url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: Optional[int] = None
) -> QdrantClient:
    """
    Create Qdrant client based on configuration
    
    Args:
        mode: 'cloud' or 'local' (defaults to env QDRANT_MODE)
        host: Qdrant host for local mode (defaults to env QDRANT_HOST)
        port: Qdrant port for local mode (defaults to env QDRANT_PORT)
        url: Qdrant Cloud URL (defaults to env QDRANT_URL)
        api_key: API key for cloud mode (defaults to env QDRANT_API_KEY)
        timeout: Connection timeout (defaults to env QDRANT_TIMEOUT)
        
    Returns:
        QdrantClient instance configured for cloud or local deployment
    """
    # Get configuration from environment if not provided
    mode = mode or os.getenv("QDRANT_MODE", "local")
    timeout = timeout or int(os.getenv("QDRANT_TIMEOUT", "30"))
    
    if mode == "cloud":
        # Cloud deployment
        url = url or os.getenv("QDRANT_URL")
        api_key = api_key or os.getenv("QDRANT_API_KEY")
        
        if not url:
            raise ValueError("QDRANT_URL must be set for cloud mode")
        if not api_key:
            raise ValueError("QDRANT_API_KEY must be set for cloud mode")
        
        logger.info(f"Creating Qdrant Cloud client: {url}")
        client = QdrantClient(
            url=url,
            api_key=api_key,
            timeout=timeout
        )
        
    else:
        # Local deployment
        host = host or os.getenv("QDRANT_HOST", "localhost")
        port = port or int(os.getenv("QDRANT_PORT", "6333"))
        
        logger.info(f"Creating Qdrant local client: {host}:{port}")
        client = QdrantClient(
            host=host,
            port=port,
            timeout=timeout
        )
    
    # Test connection
    try:
        collections = client.get_collections()
        logger.info(f"Successfully connected to Qdrant ({mode} mode)")
        logger.info(f"Available collections: {[c.name for c in collections.collections]}")
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {e}")
        raise
    
    return client
