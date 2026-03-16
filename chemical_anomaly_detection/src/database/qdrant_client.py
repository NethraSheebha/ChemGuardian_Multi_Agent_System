"""Qdrant client wrapper with connection management"""

import asyncio
import logging
from typing import Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.http.exceptions import UnexpectedResponse


logger = logging.getLogger(__name__)


class QdrantClientManager:
    """Manages Qdrant client connection with retry logic"""
    
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3
    ):
        """
        Initialize Qdrant client manager
        
        Args:
            host: Qdrant server host (for local deployment)
            port: Qdrant server port (for local deployment)
            url: Qdrant Cloud URL (for cloud deployment)
            api_key: API key for authentication (required for cloud)
            timeout: Connection timeout in seconds
            max_retries: Maximum number of connection retries
        """
        self.host = host
        self.port = port
        self.url = url
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: Optional[QdrantClient] = None
        
    def connect(self) -> QdrantClient:
        """
        Establish connection to Qdrant server with retry logic
        
        Returns:
            QdrantClient instance
            
        Raises:
            ConnectionError: If connection fails after max retries
        """
        for attempt in range(self.max_retries):
            try:
                if self.url:
                    # Cloud deployment
                    logger.info(
                        f"Connecting to Qdrant Cloud at {self.url} "
                        f"(attempt {attempt + 1}/{self.max_retries})"
                    )
                    self._client = QdrantClient(
                        url=self.url,
                        api_key=self.api_key,
                        timeout=self.timeout
                    )
                else:
                    # Local deployment
                    logger.info(
                        f"Connecting to Qdrant at {self.host}:{self.port} "
                        f"(attempt {attempt + 1}/{self.max_retries})"
                    )
                    self._client = QdrantClient(
                        host=self.host,
                        port=self.port,
                        api_key=self.api_key,
                        timeout=self.timeout
                    )
                
                # Test connection
                self._client.get_collections()
                logger.info("Successfully connected to Qdrant")
                return self._client
                
            except Exception as e:
                logger.error(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.info(f"Retrying in {wait_time} seconds...")
                    asyncio.sleep(wait_time)
                else:
                    raise ConnectionError(
                        f"Failed to connect to Qdrant after {self.max_retries} attempts"
                    ) from e
                    
    def get_client(self) -> QdrantClient:
        """
        Get the Qdrant client instance
        
        Returns:
            QdrantClient instance
            
        Raises:
            RuntimeError: If client is not connected
        """
        if self._client is None:
            raise RuntimeError("Qdrant client not connected. Call connect() first.")
        return self._client
        
    def disconnect(self) -> None:
        """Close the Qdrant client connection"""
        if self._client:
            logger.info("Disconnecting from Qdrant")
            self._client.close()
            self._client = None
