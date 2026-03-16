"""Script to create sample labeled anomalies"""

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


class LabeledAnomalyGenerator:
    """Generates sample labeled anomalies for training"""
    
    # Anomaly cause types
    CAUSES = [
        'gas_plume',
        'audio_anomaly',
        'pressure_spike',
        'valve_malfunction',
        'ppe_violation',
        'human_panic'
    ]
    
    # Severity levels
    SEVERITIES = ['mild', 'medium', 'high']
    
    # Chemical types
    CHEMICALS = ['Chlorine', 'Ammonia', 'MIC', 'Acidic Gas', 'Toxic Gas', 'Unknown']
    
    def __init__(self, qdrant_client, sensor_adapter: SensorEmbeddingAdapter):
        self.client = qdrant_client
        self.adapter = sensor_adapter
        self.schemas = QdrantSchemas(qdrant_client)
        
    def load_anomalous_sensor_data(self, csv_path: str) -> pd.DataFrame:
        """Load anomalous sensor data from CSV"""
        logger.info(f"Loading anomalous sensor data from {csv_path}")
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} anomalous sensor readings")
        return df
        
    def assign_causes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign causes to anomalies based on patterns"""
        causes = []
        for idx, row in df.iterrows():
            # Assign cause based on sensor patterns
            if row['gas_concentration_ppm'] > 500:
                cause = 'gas_plume'
            elif row['pressure_bar'] > 20 or row['pressure_bar'] < 12:
                cause = 'pressure_spike'
            elif row['vibration_mm_s'] > 15:
                cause = 'valve_malfunction'
            elif row['temperature_celsius'] > 100:
                cause = 'audio_anomaly'
            else:
                # Rotate through remaining causes
                cause = self.CAUSES[idx % len(self.CAUSES)]
            causes.append(cause)
        df['ground_truth_cause'] = causes
        return df
        
    def assign_severities(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign severity levels based on sensor values"""
        severities = []
        for idx, row in df.iterrows():
            # Compute severity score
            score = 0.0
            
            # Gas concentration factor
            if row['gas_concentration_ppm'] > 500:
                score += 0.4
            elif row['gas_concentration_ppm'] > 400:
                score += 0.2
                
            # Pressure factor
            if row['pressure_bar'] > 21 or row['pressure_bar'] < 12:
                score += 0.3
            elif row['pressure_bar'] > 19 or row['pressure_bar'] < 15:
                score += 0.15
                
            # Temperature factor
            if row['temperature_celsius'] > 100:
                score += 0.3
            elif row['temperature_celsius'] > 95:
                score += 0.15
                
            # Classify severity
            if score > 0.8:
                severity = 'high'
            elif score > 0.5:
                severity = 'medium'
            else:
                severity = 'mild'
                
            severities.append(severity)
        df['ground_truth_severity'] = severities
        return df
        
    def assign_chemicals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign detected chemicals based on cause"""
        chemicals = []
        for idx, row in df.iterrows():
            if row['ground_truth_cause'] == 'gas_plume':
                # Rotate through chemical types
                chemical = self.CHEMICALS[idx % len(self.CHEMICALS)]
            else:
                chemical = 'Unknown'
            chemicals.append(chemical)
        df['chemical_detected'] = chemicals
        return df
        
    def generate_operator_notes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate operator notes for each anomaly"""
        notes = []
        for idx, row in df.iterrows():
            cause = row['ground_truth_cause']
            severity = row['ground_truth_severity']
            
            note_templates = {
                'gas_plume': f"Gas leak detected with {severity} severity. Concentration: {row['gas_concentration_ppm']:.1f} ppm.",
                'audio_anomaly': f"Unusual audio pattern detected. {severity.capitalize()} severity alert.",
                'pressure_spike': f"Pressure anomaly: {row['pressure_bar']:.1f} bar. {severity.capitalize()} severity.",
                'valve_malfunction': f"Valve malfunction suspected. Vibration: {row['vibration_mm_s']:.1f} mm/s.",
                'ppe_violation': f"PPE violation detected via video. {severity.capitalize()} severity.",
                'human_panic': f"Human panic behavior detected. {severity.capitalize()} response required."
            }
            
            notes.append(note_templates.get(cause, f"{cause} detected with {severity} severity."))
        df['operator_notes'] = notes
        return df
        
    def assign_training_weights(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign training weights based on severity and rarity"""
        weights = []
        cause_counts = df['ground_truth_cause'].value_counts()
        
        for idx, row in df.iterrows():
            # Base weight on severity
            if row['ground_truth_severity'] == 'high':
                weight = 1.5
            elif row['ground_truth_severity'] == 'medium':
                weight = 1.0
            else:
                weight = 0.7
                
            # Increase weight for rare causes
            cause_frequency = cause_counts[row['ground_truth_cause']] / len(df)
            if cause_frequency < 0.1:
                weight *= 1.5
                
            weights.append(weight)
        df['training_weight'] = weights
        return df
        
    def generate_embeddings(self, df: pd.DataFrame) -> List[Dict]:
        """Generate sensor embeddings for all anomalies"""
        logger.info("Generating sensor embeddings for anomalies...")
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
                    'timestamp': datetime.now() - timedelta(days=30) + timedelta(hours=idx),
                    'ground_truth_cause': row['ground_truth_cause'],
                    'ground_truth_severity': row['ground_truth_severity'],
                    'chemical_detected': row['chemical_detected'],
                    'operator_notes': row['operator_notes'],
                    'training_weight': row['training_weight']
                })
            except Exception as e:
                logger.error(f"Failed to generate embedding for row {idx}: {e}")
                
        logger.info(f"Generated {len(embeddings)} anomaly embeddings")
        return embeddings
        
    def create_labeled_anomaly_points(self, embeddings: List[Dict]) -> List[PointStruct]:
        """Create Qdrant points for labeled anomalies"""
        logger.info("Creating labeled anomaly points...")
        points = []
        
        # Generate dummy video and audio embeddings (zeros for now)
        video_dim = self.schemas.VIDEO_DIM
        audio_dim = self.schemas.AUDIO_DIM
        
        for i, emb_data in enumerate(embeddings):
            # Add some variation to video/audio embeddings based on cause
            video_emb = np.random.randn(video_dim).astype(np.float32) * 0.1
            audio_emb = np.random.randn(audio_dim).astype(np.float32) * 0.1
            
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    'video': video_emb.tolist(),
                    'audio': audio_emb.tolist(),
                    'sensor': emb_data['embedding'].tolist()
                },
                payload={
                    'timestamp': emb_data['timestamp'].isoformat(),
                    'ground_truth_cause': emb_data['ground_truth_cause'],
                    'ground_truth_severity': emb_data['ground_truth_severity'],
                    'chemical_detected': emb_data['chemical_detected'],
                    'operator_notes': emb_data['operator_notes'],
                    'training_weight': emb_data['training_weight']
                }
            )
            points.append(point)
            
        logger.info(f"Created {len(points)} labeled anomaly points")
        return points
        
    def store_labeled_anomalies(self, points: List[PointStruct]):
        """Store labeled anomaly points in Qdrant"""
        logger.info(f"Storing {len(points)} labeled anomaly points in Qdrant...")
        
        # Store in batches of 100
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i+batch_size]
            self.client.upsert(
                collection_name=self.schemas.LABELED_ANOMALIES,
                points=batch
            )
            logger.info(f"Stored batch {i//batch_size + 1}/{(len(points)-1)//batch_size + 1}")
            
        logger.info("Successfully stored all labeled anomaly points")
        
    async def generate_and_store_labeled_anomalies(self, csv_path: str):
        """Main method to generate and store labeled anomalies"""
        logger.info("Starting labeled anomaly generation process...")
        
        # Load data
        df = self.load_anomalous_sensor_data(csv_path)
        
        # Assign labels
        df = self.assign_causes(df)
        df = self.assign_severities(df)
        df = self.assign_chemicals(df)
        df = self.generate_operator_notes(df)
        df = self.assign_training_weights(df)
        
        # Generate embeddings
        embeddings = self.generate_embeddings(df)
        
        # Create points
        points = self.create_labeled_anomaly_points(embeddings)
        
        # Store in Qdrant
        self.store_labeled_anomalies(points)
        
        # Print summary
        logger.info(f"\nLabeled anomaly generation complete! Stored {len(points)} total points")
        logger.info(f"\nCause distribution:")
        for cause in self.CAUSES:
            count = sum(1 for p in points if p.payload['ground_truth_cause'] == cause)
            logger.info(f"  - {cause}: {count}")
        logger.info(f"\nSeverity distribution:")
        for severity in self.SEVERITIES:
            count = sum(1 for p in points if p.payload['ground_truth_severity'] == severity)
            logger.info(f"  - {severity}: {count}")


async def main():
    """Main entry point"""
    try:
        # Connect to Qdrant (cloud or local based on env)
        logger.info("Connecting to Qdrant...")
        client = create_qdrant_client()
        
        # Initialize collections (if not already done)
        schemas = QdrantSchemas(client)
        schemas.initialize_all_collections()
        
        # Initialize sensor adapter
        logger.info("Initializing sensor adapter...")
        sensor_adapter = SensorEmbeddingAdapter()
        
        # Generate and store labeled anomalies
        generator = LabeledAnomalyGenerator(client, sensor_adapter)
        await generator.generate_and_store_labeled_anomalies('anomalous_sensor.csv')
        
        # Get collection info
        info = schemas.get_collection_info(schemas.LABELED_ANOMALIES)
        logger.info(f"\nLabeled anomalies collection info: {info}")
        
        logger.info("\nLabeled anomaly seeding completed successfully!")
        
    except Exception as e:
        logger.error(f"Failed to seed labeled anomalies: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
