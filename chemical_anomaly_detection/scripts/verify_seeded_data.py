"""Script to verify all seeded data in Qdrant"""

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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Verify all seeded data"""
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
        
        # Check all collections
        logger.info("\n" + "="*60)
        logger.info("SEEDED DATA VERIFICATION")
        logger.info("="*60)
        
        collections = [
            (schemas.BASELINES, 66, 66, "Baseline embeddings from normal sensor data"),
            (schemas.LABELED_ANOMALIES, 60, 60, "Labeled anomalies for training"),
            (schemas.RESPONSE_STRATEGIES, 150, 200, "Response strategies with effectiveness (varies due to randomization)"),
            (schemas.DATA, 0, 0, "Real-time data storage (empty until system runs)")
        ]
        
        all_good = True
        for collection_name, min_count, max_count, description in collections:
            info = schemas.get_collection_info(collection_name)
            actual_count = info['points_count']
            status = info['status']
            
            logger.info(f"\n{collection_name}:")
            logger.info(f"  Description: {description}")
            logger.info(f"  Expected: {min_count}-{max_count} points" if min_count != max_count else f"  Expected: {min_count} points")
            logger.info(f"  Actual: {actual_count} points")
            logger.info(f"  Status: {status}")
            
            if min_count <= actual_count <= max_count:
                logger.info(f"  ✅ PASS")
            else:
                logger.error(f"  ❌ FAIL - Expected {min_count}-{max_count}, got {actual_count}")
                all_good = False
                
        # Check MSDS and SOP files
        logger.info(f"\nMSDS Database:")
        msds_path = "data/msds_database.json"
        if os.path.exists(msds_path):
            logger.info(f"  ✅ PASS - {msds_path} exists")
        else:
            logger.error(f"  ❌ FAIL - {msds_path} not found")
            all_good = False
            
        logger.info(f"\nSOP Database:")
        sop_path = "data/sop_database.json"
        if os.path.exists(sop_path):
            logger.info(f"  ✅ PASS - {sop_path} exists")
        else:
            logger.error(f"  ❌ FAIL - {sop_path} not found")
            all_good = False
        
        logger.info("\n" + "="*60)
        if all_good:
            logger.info("✅ ALL VERIFICATION CHECKS PASSED!")
            logger.info("="*60)
            logger.info("\nThe system is ready for:")
            logger.info("  - Task 24: Integration testing and validation")
            logger.info("  - Task 25: Documentation and deployment guide")
            logger.info("  - End-to-end testing with real data")
        else:
            logger.error("❌ SOME VERIFICATION CHECKS FAILED!")
            logger.error("="*60)
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Verification failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
