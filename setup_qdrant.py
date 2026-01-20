from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PayloadSchemaType,
    CreateCollection,
)
import logging
from typing import Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QdrantCollectionManager:
    """Manages creation and configuration of Qdrant collections"""
    
    def __init__(self, host: str = "localhost", port: int = 6333):
        """
        Initialize Qdrant client connection
        
        Args:
            host: Qdrant server hostname (default: localhost)
            port: Qdrant server port (default: 6333)
        """
        self.client = QdrantClient(host=host, port=port)
        logger.info(f"Connected to Qdrant at {host}:{port}")
        
    def collection_exists(self, collection_name: str) -> bool:
        """
        Check if a collection already exists
        
        Args:
            collection_name: Name of the collection to check
            
        Returns:
            True if collection exists, False otherwise
        """
        try:
            collections = self.client.get_collections().collections
            return any(col.name == collection_name for col in collections)
        except Exception as e:
            logger.error(f"Error checking collection {collection_name}: {e}")
            return False
    
    def create_collection_safe(
        self, 
        collection_name: str, 
        vector_size: int,
        distance_metric: Distance = Distance.COSINE
    ) -> bool:
        """
        Create a collection only if it doesn't exist (idempotent)
        
        Args:
            collection_name: Name of the collection
            vector_size: Dimension of the vectors
            distance_metric: Distance calculation method (default: COSINE)
            
        Returns:
            True if created or already exists, False on error
        """
        try:
            if self.collection_exists(collection_name):
                logger.info(f"Collection '{collection_name}' already exists, skipping creation")
                return True
            
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=distance_metric
                )
            )
            logger.info(f"Created collection '{collection_name}' with dimension {vector_size}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create collection '{collection_name}': {e}")
            return False
    
    def payload_index_exists(self, collection_name: str, field_name: str) -> bool:
        try:
            info = self.client.get_collection(collection_name)
            schema = info.payload_schema
            return field_name in schema
        except Exception:
            return False

    def setup_payload_index(self, collection_name, field_name, field_type):
        try:
            if self.payload_index_exists(collection_name, field_name):
                logger.info(f"Index on '{field_name}' already exists in '{collection_name}', skipping")
                return True
        
            self.client.create_payload_index(collection_name=collection_name, field_name=field_name, field_schema=field_type)
            logger.info(f"Created index on '{field_name}' in '{collection_name}'")
            return True
        except Exception as e:
            logger.error(f"Failed index '{field_name}': {e}")
            return False
    
    def setup_all_collections(self) -> Dict[str, bool]:
        """
        Create all required collections for the chemical leak detection system
        
        Returns:
            Dictionary with collection names and their creation status
        """
        results = {}
        
        logger.info("\n" + "="*60)
        logger.info("SETTING UP QDRANT COLLECTIONS")
        logger.info("="*60 + "\n")
        
        # Collection 1: Chemical Safety Documents (Text Embeddings)
        logger.info("1. Creating chemical_docs collection...")
        results['chemical_docs'] = self.create_collection_safe(
            collection_name="chemical_docs",
            vector_size=768  # SentenceTransformer all-MiniLM-L6-v2 dimension
        )
        if results['chemical_docs']:
            # Create indexes for filtering
            self.setup_payload_index("chemical_docs", "chemical", PayloadSchemaType.KEYWORD)
            self.setup_payload_index("chemical_docs", "severity", PayloadSchemaType.KEYWORD)
            self.setup_payload_index("chemical_docs", "source", PayloadSchemaType.KEYWORD)
            self.setup_payload_index("chemical_docs", "timestamp", PayloadSchemaType.FLOAT)
            self.setup_payload_index("chemical_docs", "factory_zone", PayloadSchemaType.KEYWORD)
        
        # Collection 2: Visual Leak Detection (Image Embeddings)
        logger.info("\n2. Creating visual_leaks collection...")
        results['visual_leaks'] = self.create_collection_safe(
            collection_name="visual_leaks",
            vector_size=512  # OpenCLIP ViT-B/32 dimension
        )
        if results['visual_leaks']:
            self.setup_payload_index("visual_leaks", "chemical", PayloadSchemaType.KEYWORD)
            self.setup_payload_index("visual_leaks", "severity", PayloadSchemaType.KEYWORD)
            self.setup_payload_index("visual_leaks", "source", PayloadSchemaType.KEYWORD)
            self.setup_payload_index("visual_leaks", "timestamp", PayloadSchemaType.FLOAT)
            self.setup_payload_index("visual_leaks", "factory_zone", PayloadSchemaType.KEYWORD)
        
        # Collection 3: Audio Event Detection (Audio Embeddings)
        logger.info("\n3. Creating audio_events collection...")
        results['audio_events'] = self.create_collection_safe(
            collection_name="audio_events",
            vector_size=512  # CLAP audio embedding dimension
        )
        if results['audio_events']:
            self.setup_payload_index("audio_events", "chemical", PayloadSchemaType.KEYWORD)
            self.setup_payload_index("audio_events", "severity", PayloadSchemaType.KEYWORD)
            self.setup_payload_index("audio_events", "source", PayloadSchemaType.KEYWORD)
            self.setup_payload_index("audio_events", "timestamp", PayloadSchemaType.FLOAT)
            self.setup_payload_index("audio_events", "factory_zone", PayloadSchemaType.KEYWORD)
        
        # Collection 4: Sensor Pattern Recognition (Numeric Embeddings)
        logger.info("\n4. Creating sensor_patterns collection...")
        results['sensor_patterns'] = self.create_collection_safe(
            collection_name="sensor_patterns",
            vector_size=128  # Custom numeric embedding dimension
        )
        if results['sensor_patterns']:
            self.setup_payload_index("sensor_patterns", "chemical", PayloadSchemaType.KEYWORD)
            self.setup_payload_index("sensor_patterns", "severity", PayloadSchemaType.KEYWORD)
            self.setup_payload_index("sensor_patterns", "source", PayloadSchemaType.KEYWORD)
            self.setup_payload_index("sensor_patterns", "timestamp", PayloadSchemaType.FLOAT)
            self.setup_payload_index("sensor_patterns", "factory_zone", PayloadSchemaType.KEYWORD)
        
        # Collection 5: Incident Memory (Multimodal Fused Embeddings)
        logger.info("\n5. Creating incident_memory collection...")
        results['incident_memory'] = self.create_collection_safe(
            collection_name="incident_memory",
            vector_size=768  # Fused multimodal embedding dimension
        )
        if results['incident_memory']:
            self.setup_payload_index("incident_memory", "chemical", PayloadSchemaType.KEYWORD)
            self.setup_payload_index("incident_memory", "severity", PayloadSchemaType.KEYWORD)
            self.setup_payload_index("incident_memory", "source", PayloadSchemaType.KEYWORD)
            self.setup_payload_index("incident_memory", "timestamp", PayloadSchemaType.FLOAT)
            self.setup_payload_index("incident_memory", "factory_zone", PayloadSchemaType.KEYWORD)
        
        return results
    
    def verify_setup(self) -> None:
        """
        Verify all collections are created and display summary
        """
        logger.info("\n" + "="*60)
        logger.info("VERIFICATION SUMMARY")
        logger.info("="*60 + "\n")
        
        collections = self.client.get_collections().collections
        
        expected_collections = [
            ("chemical_docs", 768),
            ("visual_leaks", 512),
            ("audio_events", 512),
            ("sensor_patterns", 128),
            ("incident_memory", 768)
        ]
        
        for name, expected_dim in expected_collections:
            collection = next((c for c in collections if c.name == name), None)
            if collection:
                info = self.client.get_collection(name)
                actual_dim = info.config.params.vectors.size
                status = "" if actual_dim == expected_dim else "⚠"
                logger.info(f"{status} {name}: dimension={actual_dim}, points={info.points_count}")
            else:
                logger.warning(f"  {name}: NOT FOUND")
        
        logger.info("\n" + "="*60)
        logger.info(f"Total collections: {len(collections)}")
        logger.info("="*60 + "\n")


def main():
    """Main execution function"""
    try:
        # Initialize manager
        manager = QdrantCollectionManager(host="localhost", port=6333)
        
        # Create all collections
        results = manager.setup_all_collections()
        
        # Verify setup
        manager.verify_setup()
        
        # Check if all succeeded
        if all(results.values()):
            logger.info("All collections created successfully!")
        else:
            failed = [name for name, success in results.items() if not success]
            logger.warning(f"Some collections failed: {failed}")
            
    except Exception as e:
        logger.error(f"Setup failed with error: {e}")
        raise


if __name__ == "__main__":
    main()