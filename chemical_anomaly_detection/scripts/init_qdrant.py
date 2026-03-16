"""Initialize Qdrant collections"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import SystemConfig
from src.database.qdrant_client import QdrantClientManager
from src.database.schemas import QdrantSchemas
from src.utils.logging import setup_logging
import logging


def main():
    """Initialize all Qdrant collections"""
    
    # Setup logging
    setup_logging(level="INFO", format_type="text", log_dir="logs")
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        logger.info("Loading configuration...")
        config = SystemConfig.from_env()
        config.validate_config()
        
        # Connect to Qdrant
        logger.info(f"Connecting to Qdrant at {config.qdrant.host}:{config.qdrant.port}...")
        manager = QdrantClientManager(
            host=config.qdrant.host,
            port=config.qdrant.port,
            api_key=config.qdrant.api_key,
            timeout=config.qdrant.timeout
        )
        
        client = manager.connect()
        
        # Initialize schemas
        logger.info("Initializing collections...")
        schemas = QdrantSchemas(client)
        schemas.initialize_all_collections()
        
        # Display collection info
        logger.info("\n" + "="*60)
        logger.info("Collection Information")
        logger.info("="*60)
        
        for collection_name in [
            schemas.BASELINES,
            schemas.DATA,
            schemas.LABELED_ANOMALIES,
            schemas.RESPONSE_STRATEGIES
        ]:
            info = schemas.get_collection_info(collection_name)
            logger.info(f"\n{collection_name}:")
            logger.info(f"  Points: {info['points_count']}")
            logger.info(f"  Vectors: {info['vectors_count']}")
            logger.info(f"  Status: {info['status']}")
        
        logger.info("\n" + "="*60)
        logger.info("✅ All collections initialized successfully!")
        logger.info("="*60)
        
        # Disconnect
        manager.disconnect()
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize collections: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
