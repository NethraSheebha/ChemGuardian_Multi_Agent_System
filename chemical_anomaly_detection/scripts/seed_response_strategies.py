"""Script to create sample response strategies"""

import asyncio
import logging
import sys
import os
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
from src.config.settings import SystemConfig
from qdrant_client.models import PointStruct


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ResponseStrategyGenerator:
    """Generates sample response strategies for different incident types"""
    
    # Incident causes
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
    
    # Plant zones
    PLANT_ZONES = ['Zone_A', 'Zone_B', 'Zone_C', 'Zone_D']
    
    # Response actions by severity
    MILD_ACTIONS = [
        'Log incident in system',
        'Notify shift supervisor',
        'Monitor sensor readings',
        'Increase inspection frequency',
        'Check equipment status',
        'Review recent maintenance logs'
    ]
    
    MEDIUM_ACTIONS = [
        'Activate local alarm',
        'Dispatch safety team',
        'Isolate affected area',
        'Increase ventilation',
        'Don PPE equipment',
        'Evacuate non-essential personnel',
        'Contact emergency coordinator',
        'Prepare containment equipment'
    ]
    
    HIGH_ACTIONS = [
        'Trigger emergency shutdown',
        'Activate plant-wide alarm',
        'Evacuate all personnel',
        'Contact fire department',
        'Contact hazmat team',
        'Notify regulatory authorities',
        'Activate emergency response plan',
        'Deploy emergency containment',
        'Establish incident command center',
        'Notify neighboring facilities'
    ]
    
    def __init__(self, qdrant_client):
        self.client = qdrant_client
        self.schemas = QdrantSchemas(qdrant_client)
        
    def generate_response_strategies(self) -> List[Dict]:
        """Generate response strategies for all combinations"""
        logger.info("Generating response strategies...")
        strategies = []
        
        incident_id = 1
        for cause in self.CAUSES:
            for severity in self.SEVERITIES:
                for zone in self.PLANT_ZONES:
                    # Generate 2-3 strategies per combination
                    num_strategies = np.random.randint(2, 4)
                    
                    for i in range(num_strategies):
                        strategy = self._create_strategy(
                            incident_id, cause, severity, zone
                        )
                        strategies.append(strategy)
                        incident_id += 1
                        
        logger.info(f"Generated {len(strategies)} response strategies")
        return strategies
        
    def _create_strategy(
        self,
        incident_id: int,
        cause: str,
        severity: str,
        plant_zone: str
    ) -> Dict:
        """Create a single response strategy"""
        
        # Select actions based on severity
        if severity == 'mild':
            action_pool = self.MILD_ACTIONS
            num_actions = np.random.randint(2, 4)
        elif severity == 'medium':
            action_pool = self.MEDIUM_ACTIONS
            num_actions = np.random.randint(3, 6)
        else:  # high
            action_pool = self.HIGH_ACTIONS
            num_actions = np.random.randint(5, 9)
            
        # Randomly select successful and failed actions
        all_actions = np.random.choice(action_pool, size=min(num_actions, len(action_pool)), replace=False).tolist()
        num_successful = max(1, int(len(all_actions) * np.random.uniform(0.7, 1.0)))
        
        successful_actions = all_actions[:num_successful]
        failed_actions = all_actions[num_successful:] if len(all_actions) > num_successful else []
        
        # Generate effectiveness score (higher for more successful actions)
        effectiveness_score = num_successful / len(all_actions) if all_actions else 0.5
        effectiveness_score += np.random.uniform(-0.1, 0.1)  # Add some noise
        effectiveness_score = max(0.0, min(1.0, effectiveness_score))
        
        # Generate response time (faster for mild, slower for high)
        if severity == 'mild':
            response_time = np.random.uniform(60, 300)  # 1-5 minutes
        elif severity == 'medium':
            response_time = np.random.uniform(120, 600)  # 2-10 minutes
        else:  # high
            response_time = np.random.uniform(180, 900)  # 3-15 minutes
            
        # Determine outcome based on effectiveness
        if effectiveness_score > 0.8:
            outcome = 'resolved'
        elif effectiveness_score > 0.5:
            outcome = 'escalated'
        else:
            outcome = 'ongoing'
            
        # Generate incident embedding (128-dim)
        # Create a deterministic embedding based on cause, severity, and zone
        np.random.seed(hash(f"{cause}_{severity}_{plant_zone}_{incident_id}") % (2**32))
        incident_embedding = np.random.randn(self.schemas.INCIDENT_DIM).astype(np.float32)
        
        # Add some structure based on severity
        if severity == 'high':
            incident_embedding *= 1.5
        elif severity == 'mild':
            incident_embedding *= 0.5
            
        return {
            'incident_id': f"INC_{incident_id:05d}",
            'cause': cause,
            'severity': severity,
            'plant_zone': plant_zone,
            'successful_actions': successful_actions,
            'failed_actions': failed_actions,
            'effectiveness_score': effectiveness_score,
            'response_time_seconds': response_time,
            'outcome': outcome,
            'incident_embedding': incident_embedding,
            'timestamp': datetime.now() - timedelta(days=np.random.randint(1, 365))
        }
        
    def create_response_strategy_points(self, strategies: List[Dict]) -> List[PointStruct]:
        """Create Qdrant points for response strategies"""
        logger.info("Creating response strategy points...")
        points = []
        
        for strategy in strategies:
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    'incident_embedding': strategy['incident_embedding'].tolist()
                },
                payload={
                    'incident_id': strategy['incident_id'],
                    'cause': strategy['cause'],
                    'severity': strategy['severity'],
                    'plant_zone': strategy['plant_zone'],
                    'successful_actions': strategy['successful_actions'],
                    'failed_actions': strategy['failed_actions'],
                    'effectiveness_score': float(strategy['effectiveness_score']),
                    'response_time_seconds': float(strategy['response_time_seconds']),
                    'outcome': strategy['outcome'],
                    'timestamp': strategy['timestamp'].isoformat()
                }
            )
            points.append(point)
            
        logger.info(f"Created {len(points)} response strategy points")
        return points
        
    def store_response_strategies(self, points: List[PointStruct]):
        """Store response strategy points in Qdrant"""
        logger.info(f"Storing {len(points)} response strategy points in Qdrant...")
        
        # Store in batches of 100
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i+batch_size]
            self.client.upsert(
                collection_name=self.schemas.RESPONSE_STRATEGIES,
                points=batch
            )
            logger.info(f"Stored batch {i//batch_size + 1}/{(len(points)-1)//batch_size + 1}")
            
        logger.info("Successfully stored all response strategy points")
        
    async def generate_and_store_response_strategies(self):
        """Main method to generate and store response strategies"""
        logger.info("Starting response strategy generation process...")
        
        # Generate strategies
        strategies = self.generate_response_strategies()
        
        # Create points
        points = self.create_response_strategy_points(strategies)
        
        # Store in Qdrant
        self.store_response_strategies(points)
        
        # Print summary
        logger.info(f"\nResponse strategy generation complete! Stored {len(points)} total points")
        logger.info(f"\nSeverity distribution:")
        for severity in self.SEVERITIES:
            count = sum(1 for p in points if p.payload['severity'] == severity)
            logger.info(f"  - {severity}: {count}")
        logger.info(f"\nOutcome distribution:")
        outcomes = set(p.payload['outcome'] for p in points)
        for outcome in outcomes:
            count = sum(1 for p in points if p.payload['outcome'] == outcome)
            logger.info(f"  - {outcome}: {count}")
        logger.info(f"\nAverage effectiveness score: {np.mean([p.payload['effectiveness_score'] for p in points]):.3f}")


async def main():
    """Main entry point"""
    try:
        # Connect to Qdrant (cloud or local based on env)
        logger.info("Connecting to Qdrant...")
        client = create_qdrant_client()
        
        # Initialize collections (if not already done)
        schemas = QdrantSchemas(client)
        schemas.initialize_all_collections()
        
        # Generate and store response strategies
        generator = ResponseStrategyGenerator(client)
        await generator.generate_and_store_response_strategies()
        
        # Get collection info
        info = schemas.get_collection_info(schemas.RESPONSE_STRATEGIES)
        logger.info(f"\nResponse strategies collection info: {info}")
        
        logger.info("\nResponse strategy seeding completed successfully!")
        
    except Exception as e:
        logger.error(f"Failed to seed response strategies: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
