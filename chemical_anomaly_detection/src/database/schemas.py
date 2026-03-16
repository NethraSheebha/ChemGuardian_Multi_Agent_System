"""Qdrant collection schemas and initialization"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    FieldCondition,
    Filter,
    MatchValue,
    Range,
    PayloadSchemaType,
)


logger = logging.getLogger(__name__)


class QdrantSchemas:
    """Manages Qdrant collection schemas and initialization"""
    
    # Collection names
    BASELINES = "baselines"
    DATA = "data"
    LABELED_ANOMALIES = "labeled_anomalies"
    RESPONSE_STRATEGIES = "response_strategies"
    
    # Vector dimensions
    VIDEO_DIM = 512
    AUDIO_DIM = 512
    SENSOR_DIM = 128
    INCIDENT_DIM = 128
    
    def __init__(self, client: QdrantClient):
        """
        Initialize schema manager
        
        Args:
            client: QdrantClient instance
        """
        self.client = client
        
    def create_baselines_collection(self) -> None:
        """
        Create baselines collection with multivector schema
        
        Collection stores baseline embeddings for normal operating conditions.
        Supports shift-specific and equipment-specific baselines.
        
        Vectors:
            - video: 1024-dim, Cosine distance
            - audio: 2048-dim, Cosine distance
            - sensor: 128-dim, Euclidean distance
            
        Payload:
            - timestamp: datetime
            - shift: keyword (morning/afternoon/night)
            - equipment_id: keyword
            - plant_zone: keyword
            - baseline_type: keyword (shift_baseline/equipment_baseline/global_baseline)
        """
        logger.info(f"Creating collection: {self.BASELINES}")
        
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            if any(c.name == self.BASELINES for c in collections):
                logger.info(f"Collection {self.BASELINES} already exists")
                return
                
            # Create collection with multivector configuration
            self.client.create_collection(
                collection_name=self.BASELINES,
                vectors_config={
                    "video": VectorParams(
                        size=self.VIDEO_DIM,
                        distance=Distance.COSINE
                    ),
                    "audio": VectorParams(
                        size=self.AUDIO_DIM,
                        distance=Distance.COSINE
                    ),
                    "sensor": VectorParams(
                        size=self.SENSOR_DIM,
                        distance=Distance.EUCLID
                    )
                }
            )
            
            # Create payload indexes for fast filtering
            self.client.create_payload_index(
                collection_name=self.BASELINES,
                field_name="shift",
                field_schema=PayloadSchemaType.KEYWORD
            )
            
            self.client.create_payload_index(
                collection_name=self.BASELINES,
                field_name="equipment_id",
                field_schema=PayloadSchemaType.KEYWORD
            )
            
            self.client.create_payload_index(
                collection_name=self.BASELINES,
                field_name="plant_zone",
                field_schema=PayloadSchemaType.KEYWORD
            )
            
            self.client.create_payload_index(
                collection_name=self.BASELINES,
                field_name="baseline_type",
                field_schema=PayloadSchemaType.KEYWORD
            )
            
            logger.info(f"Successfully created collection: {self.BASELINES}")
            
        except Exception as e:
            logger.error(f"Failed to create collection {self.BASELINES}: {e}")
            raise
            
    def create_data_collection(self) -> None:
        """
        Create data collection with multivector schema
        
        Collection stores all embeddings (normal and anomalous) with metadata.
        Supports time-window and location-based queries.
        
        Vectors:
            - video: 1024-dim, Cosine distance
            - audio: 2048-dim, Cosine distance
            - sensor: 128-dim, Euclidean distance
            
        Payload:
            - timestamp: datetime
            - location: geo (lat/lon)
            - camera_id: keyword
            - sensor_ids: keyword array
            - plant_zone: keyword
            - shift: keyword
            - is_anomaly: bool
            - anomaly_scores: dict (video/audio/sensor floats)
            - cause: keyword (if anomaly)
            - severity: keyword (if anomaly: mild/medium/high)
            - operator_feedback: keyword (confirmed/false_positive/false_negative)
        """
        logger.info(f"Creating collection: {self.DATA}")
        
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            if any(c.name == self.DATA for c in collections):
                logger.info(f"Collection {self.DATA} already exists")
                return
                
            # Create collection with multivector configuration
            self.client.create_collection(
                collection_name=self.DATA,
                vectors_config={
                    "video": VectorParams(
                        size=self.VIDEO_DIM,
                        distance=Distance.COSINE
                    ),
                    "audio": VectorParams(
                        size=self.AUDIO_DIM,
                        distance=Distance.COSINE
                    ),
                    "sensor": VectorParams(
                        size=self.SENSOR_DIM,
                        distance=Distance.EUCLID
                    )
                }
            )
            
            # Create payload indexes
            self.client.create_payload_index(
                collection_name=self.DATA,
                field_name="timestamp",
                field_schema=PayloadSchemaType.DATETIME
            )
            
            self.client.create_payload_index(
                collection_name=self.DATA,
                field_name="plant_zone",
                field_schema=PayloadSchemaType.KEYWORD
            )
            
            self.client.create_payload_index(
                collection_name=self.DATA,
                field_name="is_anomaly",
                field_schema=PayloadSchemaType.BOOL
            )
            
            self.client.create_payload_index(
                collection_name=self.DATA,
                field_name="severity",
                field_schema=PayloadSchemaType.KEYWORD
            )
            
            self.client.create_payload_index(
                collection_name=self.DATA,
                field_name="shift",
                field_schema=PayloadSchemaType.KEYWORD
            )
            
            logger.info(f"Successfully created collection: {self.DATA}")
            
        except Exception as e:
            logger.error(f"Failed to create collection {self.DATA}: {e}")
            raise
            
    def create_labeled_anomalies_collection(self) -> None:
        """
        Create labeled_anomalies collection
        
        Collection stores confirmed anomalies for continual learning.
        Used for cause inference through similarity search.
        
        Vectors:
            - video: 1024-dim, Cosine distance
            - audio: 2048-dim, Cosine distance
            - sensor: 128-dim, Euclidean distance
            
        Payload:
            - timestamp: datetime
            - ground_truth_cause: keyword
            - ground_truth_severity: keyword
            - chemical_detected: keyword
            - operator_notes: text
            - training_weight: float (importance for retraining)
        """
        logger.info(f"Creating collection: {self.LABELED_ANOMALIES}")
        
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            if any(c.name == self.LABELED_ANOMALIES for c in collections):
                logger.info(f"Collection {self.LABELED_ANOMALIES} already exists")
                return
                
            # Create collection with multivector configuration
            self.client.create_collection(
                collection_name=self.LABELED_ANOMALIES,
                vectors_config={
                    "video": VectorParams(
                        size=self.VIDEO_DIM,
                        distance=Distance.COSINE
                    ),
                    "audio": VectorParams(
                        size=self.AUDIO_DIM,
                        distance=Distance.COSINE
                    ),
                    "sensor": VectorParams(
                        size=self.SENSOR_DIM,
                        distance=Distance.EUCLID
                    )
                }
            )
            
            # Create payload indexes
            self.client.create_payload_index(
                collection_name=self.LABELED_ANOMALIES,
                field_name="ground_truth_cause",
                field_schema=PayloadSchemaType.KEYWORD
            )
            
            self.client.create_payload_index(
                collection_name=self.LABELED_ANOMALIES,
                field_name="ground_truth_severity",
                field_schema=PayloadSchemaType.KEYWORD
            )
            
            self.client.create_payload_index(
                collection_name=self.LABELED_ANOMALIES,
                field_name="chemical_detected",
                field_schema=PayloadSchemaType.KEYWORD
            )
            
            logger.info(f"Successfully created collection: {self.LABELED_ANOMALIES}")
            
        except Exception as e:
            logger.error(f"Failed to create collection {self.LABELED_ANOMALIES}: {e}")
            raise
            
    def create_response_strategies_collection(self) -> None:
        """
        Create response_strategies collection
        
        Collection stores historical response strategies with effectiveness scores.
        Used for similarity-based response selection.
        
        Vectors:
            - incident_embedding: 128-dim, Cosine distance
            
        Payload:
            - incident_id: keyword
            - cause: keyword
            - severity: keyword (mild/medium/high)
            - plant_zone: keyword
            - successful_actions: keyword array
            - failed_actions: keyword array
            - effectiveness_score: float (0-1)
            - response_time_seconds: float
            - outcome: keyword (resolved/escalated/ongoing)
        """
        logger.info(f"Creating collection: {self.RESPONSE_STRATEGIES}")
        
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            if any(c.name == self.RESPONSE_STRATEGIES for c in collections):
                logger.info(f"Collection {self.RESPONSE_STRATEGIES} already exists")
                return
                
            # Create collection with single vector configuration
            self.client.create_collection(
                collection_name=self.RESPONSE_STRATEGIES,
                vectors_config={
                    "incident_embedding": VectorParams(
                        size=self.INCIDENT_DIM,
                        distance=Distance.COSINE
                    )
                }
            )
            
            # Create payload indexes
            self.client.create_payload_index(
                collection_name=self.RESPONSE_STRATEGIES,
                field_name="severity",
                field_schema=PayloadSchemaType.KEYWORD
            )
            
            self.client.create_payload_index(
                collection_name=self.RESPONSE_STRATEGIES,
                field_name="plant_zone",
                field_schema=PayloadSchemaType.KEYWORD
            )
            
            self.client.create_payload_index(
                collection_name=self.RESPONSE_STRATEGIES,
                field_name="effectiveness_score",
                field_schema=PayloadSchemaType.FLOAT
            )
            
            self.client.create_payload_index(
                collection_name=self.RESPONSE_STRATEGIES,
                field_name="cause",
                field_schema=PayloadSchemaType.KEYWORD
            )
            
            logger.info(f"Successfully created collection: {self.RESPONSE_STRATEGIES}")
            
        except Exception as e:
            logger.error(f"Failed to create collection {self.RESPONSE_STRATEGIES}: {e}")
            raise
            
    def initialize_all_collections(self) -> None:
        """
        Initialize all collections
        
        Creates all required collections if they don't exist.
        Safe to call multiple times (idempotent).
        """
        logger.info("Initializing all Qdrant collections")
        
        self.create_baselines_collection()
        self.create_data_collection()
        self.create_labeled_anomalies_collection()
        self.create_response_strategies_collection()
        
        logger.info("All collections initialized successfully")
        
    def delete_all_collections(self) -> None:
        """
        Delete all collections (use with caution!)
        
        WARNING: This will delete all data in the collections.
        Only use for testing or complete system reset.
        """
        logger.warning("Deleting all Qdrant collections")
        
        collections = [
            self.BASELINES,
            self.DATA,
            self.LABELED_ANOMALIES,
            self.RESPONSE_STRATEGIES
        ]
        
        for collection_name in collections:
            try:
                self.client.delete_collection(collection_name=collection_name)
                logger.info(f"Deleted collection: {collection_name}")
            except Exception as e:
                logger.warning(f"Could not delete collection {collection_name}: {e}")
                
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """
        Get information about a collection
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Dictionary with collection information
        """
        try:
            info = self.client.get_collection(collection_name=collection_name)
            return {
                "name": collection_name,
                "points_count": info.points_count,
                "status": info.status,
                "config": info.config
            }
        except Exception as e:
            logger.error(f"Failed to get info for collection {collection_name}: {e}")
            raise
