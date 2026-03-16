"""Script to generate baseline embeddings from normal sensor data"""

import asyncio
import logging
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict
import uuid
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.database.client_factory import create_qdrant_client
from src.database.schemas import QdrantSchemas
from src.models.sensor_adapter import SensorEmbeddingAdapter
from src.config.settings import SystemConfig
from qdrant_client.models import PointStruct


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BaselineGenerator:
    """Generates baseline embeddings from normal sensor data"""
    
    def __init__(self, qdrant_client, sensor_adapter: SensorEmbeddingAdapter):
        self.client = qdrant_client
        self.adapter = sensor_adapter
        self.schemas = QdrantSchemas(qdrant_client)
        
    def load_normal_sensor_data(self, csv_path: str) -> pd.DataFrame:
        """Load normal sensor data from CSV"""
        logger.info(f"Loading normal sensor data from {csv_path}")
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} sensor readings")
        return df
        
    def assign_shifts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign shifts based on timestamp (simulated)"""
        # Simulate shifts: morning (0-20), afternoon (21-40), night (41-60)
        df['shift'] = df['timestamp_sec'].apply(
            lambda x: 'morning' if x <= 20 else ('afternoon' if x <= 40 else 'night')
        )
        return df
        
    def assign_equipment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign equipment IDs based on timestamp (simulated)"""
        # Simulate 3 equipment groups
        df['equipment_id'] = df['timestamp_sec'].apply(
            lambda x: f"EQUIP_{(x % 3) + 1:03d}"
        )
        return df
        
    def assign_plant_zones(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign plant zones based on timestamp (simulated)"""
        # Simulate 4 plant zones
        zones = ['Zone_A', 'Zone_B', 'Zone_C', 'Zone_D']
        df['plant_zone'] = df['timestamp_sec'].apply(
            lambda x: zones[x % 4]
        )
        return df
        
    def generate_embeddings(self, df: pd.DataFrame) -> List[Dict]:
        """Generate sensor embeddings for all readings"""
        logger.info("Generating sensor embeddings...")
        embeddings = []
        
        for idx, row in df.iterrows():
            sensor_data = {
                'temperature_celsius': row['temperature_celsius'],
                'pressure_bar': row['pressure_bar'],
                'gas_concentration_ppm': row['gas_concentration_ppm'],
                'vibration_mm_s': row['vibration_mm_s'],
                'flow_rate_lpm': row['flow_rate_lpm']
            }
            
            try:
                embedding = self.adapter.embed(sensor_data)
                embeddings.append({
                    'embedding': embedding,
                    'timestamp': datetime.now() - timedelta(seconds=len(df) - idx),
                    'shift': row['shift'],
                    'equipment_id': row['equipment_id'],
                    'plant_zone': row['plant_zone']
                })
            except Exception as e:
                logger.error(f"Failed to generate embedding for row {idx}: {e}")
                
        logger.info(f"Generated {len(embeddings)} embeddings")
        return embeddings
        
    def create_baseline_points(self, embeddings: List[Dict]) -> List[PointStruct]:
        """Create Qdrant points for baseline embeddings"""
        logger.info("Creating baseline points...")
        points = []
        
        # Generate dummy video and audio embeddings (zeros for now)
        video_dim = self.schemas.VIDEO_DIM
        audio_dim = self.schemas.AUDIO_DIM
        
        for i, emb_data in enumerate(embeddings):
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    'video': np.zeros(video_dim, dtype=np.float32).tolist(),
                    'audio': np.zeros(audio_dim, dtype=np.float32).tolist(),
                    'sensor': emb_data['embedding'].tolist()
                },
                payload={
                    'timestamp': emb_data['timestamp'].isoformat(),
                    'shift': emb_data['shift'],
                    'equipment_id': emb_data['equipment_id'],
                    'plant_zone': emb_data['plant_zone'],
                    'baseline_type': 'global_baseline'
                }
            )
            points.append(point)
            
        logger.info(f"Created {len(points)} baseline points")
        return points
        
    def create_shift_specific_baselines(self, embeddings: List[Dict]) -> List[PointStruct]:
        """Create shift-specific baseline points"""
        logger.info("Creating shift-specific baselines...")
        points = []
        
        # Group by shift
        shifts = {}
        for emb_data in embeddings:
            shift = emb_data['shift']
            if shift not in shifts:
                shifts[shift] = []
            shifts[shift].append(emb_data)
            
        # Create average baseline for each shift
        video_dim = self.schemas.VIDEO_DIM
        audio_dim = self.schemas.AUDIO_DIM
        
        for shift, shift_embeddings in shifts.items():
            # Average sensor embeddings for this shift
            sensor_embs = np.array([e['embedding'] for e in shift_embeddings])
            avg_sensor_emb = sensor_embs.mean(axis=0)
            
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    'video': np.zeros(video_dim, dtype=np.float32).tolist(),
                    'audio': np.zeros(audio_dim, dtype=np.float32).tolist(),
                    'sensor': avg_sensor_emb.tolist()
                },
                payload={
                    'timestamp': datetime.now().isoformat(),
                    'shift': shift,
                    'equipment_id': 'ALL',
                    'plant_zone': 'ALL',
                    'baseline_type': 'shift_baseline'
                }
            )
            points.append(point)
            
        logger.info(f"Created {len(points)} shift-specific baseline points")
        return points
        
    def create_equipment_specific_baselines(self, embeddings: List[Dict]) -> List[PointStruct]:
        """Create equipment-specific baseline points"""
        logger.info("Creating equipment-specific baselines...")
        points = []
        
        # Group by equipment
        equipment = {}
        for emb_data in embeddings:
            equip_id = emb_data['equipment_id']
            if equip_id not in equipment:
                equipment[equip_id] = []
            equipment[equip_id].append(emb_data)
            
        # Create average baseline for each equipment
        video_dim = self.schemas.VIDEO_DIM
        audio_dim = self.schemas.AUDIO_DIM
        
        for equip_id, equip_embeddings in equipment.items():
            # Average sensor embeddings for this equipment
            sensor_embs = np.array([e['embedding'] for e in equip_embeddings])
            avg_sensor_emb = sensor_embs.mean(axis=0)
            
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    'video': np.zeros(video_dim, dtype=np.float32).tolist(),
                    'audio': np.zeros(audio_dim, dtype=np.float32).tolist(),
                    'sensor': avg_sensor_emb.tolist()
                },
                payload={
                    'timestamp': datetime.now().isoformat(),
                    'shift': 'ALL',
                    'equipment_id': equip_id,
                    'plant_zone': 'ALL',
                    'baseline_type': 'equipment_baseline'
                }
            )
            points.append(point)
            
        logger.info(f"Created {len(points)} equipment-specific baseline points")
        return points
        
    def store_baselines(self, points: List[PointStruct]):
        """Store baseline points in Qdrant"""
        logger.info(f"Storing {len(points)} baseline points in Qdrant...")
        
        # Store in batches of 100
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i+batch_size]
            self.client.upsert(
                collection_name=self.schemas.BASELINES,
                points=batch
            )
            logger.info(f"Stored batch {i//batch_size + 1}/{(len(points)-1)//batch_size + 1}")
            
        logger.info("Successfully stored all baseline points")
        
    async def generate_and_store_baselines(self, csv_path: str):
        """Main method to generate and store baselines"""
        logger.info("Starting baseline generation process...")
        
        # Load data
        df = self.load_normal_sensor_data(csv_path)
        
        # Assign metadata
        df = self.assign_shifts(df)
        df = self.assign_equipment(df)
        df = self.assign_plant_zones(df)
        
        # Generate embeddings
        embeddings = self.generate_embeddings(df)
        
        # Create baseline points
        global_points = self.create_baseline_points(embeddings)
        shift_points = self.create_shift_specific_baselines(embeddings)
        equipment_points = self.create_equipment_specific_baselines(embeddings)
        
        # Combine all points
        all_points = global_points + shift_points + equipment_points
        
        # Store in Qdrant
        self.store_baselines(all_points)
        
        logger.info(f"Baseline generation complete! Stored {len(all_points)} total points")
        logger.info(f"  - Global baselines: {len(global_points)}")
        logger.info(f"  - Shift-specific baselines: {len(shift_points)}")
        logger.info(f"  - Equipment-specific baselines: {len(equipment_points)}")


async def main():
    """Main entry point"""
    try:
        # Connect to Qdrant (cloud or local based on env)
        logger.info("Connecting to Qdrant...")
        client = create_qdrant_client()
        
        # Initialize collections
        schemas = QdrantSchemas(client)
        schemas.initialize_all_collections()
        
        # Initialize sensor adapter
        logger.info("Initializing sensor adapter...")
        sensor_adapter = SensorEmbeddingAdapter()
        
        # Generate and store baselines
        generator = BaselineGenerator(client, sensor_adapter)
        await generator.generate_and_store_baselines('normal_sensor_data.csv')
        
        # Get collection info
        info = schemas.get_collection_info(schemas.BASELINES)
        logger.info(f"Baselines collection info: {info}")
        
        logger.info("Baseline seeding completed successfully!")
        
    except Exception as e:
        logger.error(f"Failed to seed baselines: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
