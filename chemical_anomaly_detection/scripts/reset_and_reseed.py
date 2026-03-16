"""Script to reset collections and re-seed all data"""

import asyncio
import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.database.qdrant_client import QdrantClientManager
from src.database.schemas import QdrantSchemas
from src.config.settings import SystemConfig
import logging

# Import the generator classes
from seed_baselines import BaselineGenerator
from seed_labeled_anomalies import LabeledAnomalyGenerator
from seed_response_strategies import ResponseStrategyGenerator
from src.models.sensor_adapter import SensorEmbeddingAdapter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Reset collections and re-seed all data"""
    try:
        # Load settings
        settings = SystemConfig.from_env()
        
        # Connect to Qdrant
        logger.info("Connecting to Qdrant...")
        manager = QdrantClientManager(
            host=settings.qdrant.host,
            port=settings.qdrant.port,
            timeout=settings.qdrant.timeout
        )
        client = manager.connect()
        schemas = QdrantSchemas(client)
        
        # Delete all collections
        logger.info("\n" + "="*60)
        logger.info("STEP 1: Deleting existing collections...")
        logger.info("="*60)
        schemas.delete_all_collections()
        logger.info("✅ All collections deleted")
        
        # Recreate collections
        logger.info("\n" + "="*60)
        logger.info("STEP 2: Recreating collections...")
        logger.info("="*60)
        schemas.initialize_all_collections()
        logger.info("✅ All collections recreated")
        
        # Initialize sensor adapter
        logger.info("\n" + "="*60)
        logger.info("STEP 3: Seeding baselines...")
        logger.info("="*60)
        sensor_adapter = SensorEmbeddingAdapter()
        baseline_gen = BaselineGenerator(client, sensor_adapter)
        await baseline_gen.generate_and_store_baselines('normal_sensor_data.csv')
        logger.info("✅ Baselines seeded")
        
        # Seed labeled anomalies
        logger.info("\n" + "="*60)
        logger.info("STEP 4: Seeding labeled anomalies...")
        logger.info("="*60)
        anomaly_gen = LabeledAnomalyGenerator(client, sensor_adapter)
        await anomaly_gen.generate_and_store_labeled_anomalies('anomalous_sensor.csv')
        logger.info("✅ Labeled anomalies seeded")
        
        # Seed response strategies
        logger.info("\n" + "="*60)
        logger.info("STEP 5: Seeding response strategies...")
        logger.info("="*60)
        response_gen = ResponseStrategyGenerator(client)
        await response_gen.generate_and_store_response_strategies()
        logger.info("✅ Response strategies seeded")
        
        # Verify
        logger.info("\n" + "="*60)
        logger.info("STEP 6: Verifying seeded data...")
        logger.info("="*60)
        
        collections = [
            (schemas.BASELINES, 66),
            (schemas.LABELED_ANOMALIES, 60),
            (schemas.RESPONSE_STRATEGIES, 186),
            (schemas.DATA, 0)
        ]
        
        all_good = True
        for collection_name, expected_count in collections:
            info = schemas.get_collection_info(collection_name)
            actual_count = info['points_count']
            
            if actual_count == expected_count:
                logger.info(f"✅ {collection_name}: {actual_count} points (expected {expected_count})")
            else:
                logger.error(f"❌ {collection_name}: {actual_count} points (expected {expected_count})")
                all_good = False
        
        logger.info("\n" + "="*60)
        if all_good:
            logger.info("✅ ALL DATA SUCCESSFULLY RESET AND RESEEDED!")
        else:
            logger.error("❌ SOME ISSUES DETECTED")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Reset and reseed failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
