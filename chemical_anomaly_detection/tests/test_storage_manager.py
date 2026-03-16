"""Tests for StorageManager"""

import pytest
import asyncio
import numpy as np
from datetime import datetime
from unittest.mock import Mock, MagicMock, AsyncMock, patch
from uuid import UUID

from src.agents.storage_manager import StorageManager, StorageResult
from src.agents.input_collection_agent import MultimodalEmbedding, ModalityStatus


@pytest.fixture
def mock_qdrant_client():
    """Create mock Qdrant client"""
    client = Mock()
    client.upsert = Mock(return_value=None)
    return client


@pytest.fixture
def storage_manager(mock_qdrant_client):
    """Create StorageManager instance"""
    return StorageManager(
        qdrant_client=mock_qdrant_client,
        collection_name="data",
        max_retries=3,
        retry_delay=0.1
    )


@pytest.fixture
def sample_embedding():
    """Create sample multimodal embedding"""
    return MultimodalEmbedding(
        timestamp=datetime.utcnow().isoformat(),
        video_embedding=np.random.randn(512).astype(np.float32),
        audio_embedding=np.random.randn(512).astype(np.float32),
        sensor_embedding=np.random.randn(128).astype(np.float32),
        metadata={
            "plant_zone": "Zone_A",
            "shift": "morning",
            "camera_id": "CAM001",
            "sensor_ids": ["SENS001", "SENS002"]
        }
    )


class TestStorageManager:
    """Test suite for StorageManager"""
    
    def test_initialization(self, storage_manager):
        """Test storage manager initialization"""
        assert storage_manager.collection_name == "data"
        assert storage_manager.max_retries == 3
        assert storage_manager.retry_delay == 0.1
        assert storage_manager.stats["total_stored"] == 0
    
    @pytest.mark.asyncio
    async def test_store_embedding_normal(
        self,
        storage_manager,
        sample_embedding
    ):
        """Test storing normal (non-anomaly) embedding"""
        # Store embedding
        result = await storage_manager.store_embedding(
            embedding=sample_embedding,
            is_anomaly=False,
            anomaly_scores={"video": 0.3, "audio": 0.2, "sensor": 1.0},
            confidence=0.5
        )
        
        # Verify result
        assert result.success is True
        assert result.is_anomaly is False
        assert UUID(result.point_id)  # Valid UUID
        
        # Verify statistics
        assert storage_manager.stats["total_stored"] == 1
        assert storage_manager.stats["normal_stored"] == 1
        assert storage_manager.stats["anomalies_stored"] == 0
    
    @pytest.mark.asyncio
    async def test_store_embedding_anomaly(
        self,
        storage_manager,
        sample_embedding
    ):
        """Test storing anomaly embedding"""
        # Store embedding
        result = await storage_manager.store_embedding(
            embedding=sample_embedding,
            is_anomaly=True,
            anomaly_scores={"video": 0.8, "audio": 0.7, "sensor": 3.0},
            confidence=0.9,
            cause="gas_leak",
            severity="high"
        )
        
        # Verify result
        assert result.success is True
        assert result.is_anomaly is True
        
        # Verify statistics
        assert storage_manager.stats["total_stored"] == 1
        assert storage_manager.stats["anomalies_stored"] == 1
        assert storage_manager.stats["normal_stored"] == 0
    
    @pytest.mark.asyncio
    async def test_store_embedding_with_callback(
        self,
        mock_qdrant_client,
        sample_embedding
    ):
        """Test storing anomaly with callback trigger"""
        # Create callback mock
        callback = AsyncMock()
        
        # Create storage manager with callback
        storage_manager = StorageManager(
            qdrant_client=mock_qdrant_client,
            anomaly_callback=callback
        )
        
        # Store anomaly
        result = await storage_manager.store_embedding(
            embedding=sample_embedding,
            is_anomaly=True,
            anomaly_scores={"video": 0.8},
            confidence=0.9
        )
        
        # Verify callback was triggered
        assert callback.called
        assert storage_manager.stats["anomaly_callbacks_triggered"] == 1
    
    @pytest.mark.asyncio
    async def test_store_embedding_partial_modalities(
        self,
        storage_manager
    ):
        """Test storing embedding with only some modalities"""
        # Create embedding with only video and sensor
        embedding = MultimodalEmbedding(
            timestamp=datetime.utcnow().isoformat(),
            video_embedding=np.random.randn(512).astype(np.float32),
            audio_embedding=None,
            sensor_embedding=np.random.randn(128).astype(np.float32),
            metadata={}
        )
        
        # Store embedding
        result = await storage_manager.store_embedding(
            embedding=embedding,
            is_anomaly=False,
            anomaly_scores={"video": 0.3, "sensor": 1.0},
            confidence=0.5
        )
        
        # Verify success
        assert result.success is True
        assert storage_manager.stats["total_stored"] == 1
    
    @pytest.mark.asyncio
    async def test_store_embedding_no_modalities(
        self,
        storage_manager
    ):
        """Test storing embedding with no modalities fails"""
        # Create embedding with no modalities
        embedding = MultimodalEmbedding(
            timestamp=datetime.utcnow().isoformat(),
            video_embedding=None,
            audio_embedding=None,
            sensor_embedding=None,
            metadata={}
        )
        
        # Store embedding
        result = await storage_manager.store_embedding(
            embedding=embedding,
            is_anomaly=False,
            anomaly_scores={},
            confidence=0.0
        )
        
        # Verify failure
        assert result.success is False
        assert result.error == "No vectors available"
        assert storage_manager.stats["storage_failures"] == 1
    
    @pytest.mark.asyncio
    async def test_store_embedding_retry_on_failure(
        self,
        mock_qdrant_client,
        sample_embedding
    ):
        """Test retry logic on storage failure"""
        # Mock upsert to fail twice then succeed
        mock_qdrant_client.upsert = Mock(
            side_effect=[
                Exception("Connection error"),
                Exception("Connection error"),
                None  # Success on third try
            ]
        )
        
        storage_manager = StorageManager(
            qdrant_client=mock_qdrant_client,
            max_retries=3,
            retry_delay=0.01  # Fast retry for testing
        )
        
        # Store embedding
        result = await storage_manager.store_embedding(
            embedding=sample_embedding,
            is_anomaly=False,
            anomaly_scores={"video": 0.3},
            confidence=0.5
        )
        
        # Verify success after retries
        assert result.success is True
        assert storage_manager.stats["retry_count"] == 2
        assert storage_manager.stats["total_stored"] == 1
    
    @pytest.mark.asyncio
    async def test_store_embedding_max_retries_exceeded(
        self,
        mock_qdrant_client,
        sample_embedding
    ):
        """Test failure after max retries"""
        # Mock upsert to always fail
        mock_qdrant_client.upsert = Mock(
            side_effect=Exception("Connection error")
        )
        
        storage_manager = StorageManager(
            qdrant_client=mock_qdrant_client,
            max_retries=3,
            retry_delay=0.01
        )
        
        # Store embedding
        result = await storage_manager.store_embedding(
            embedding=sample_embedding,
            is_anomaly=False,
            anomaly_scores={"video": 0.3},
            confidence=0.5
        )
        
        # Verify failure
        assert result.success is False
        assert storage_manager.stats["storage_failures"] == 1
        assert storage_manager.stats["retry_count"] == 3
    
    @pytest.mark.asyncio
    async def test_store_batch(
        self,
        storage_manager,
        sample_embedding
    ):
        """Test batch storage"""
        # Create batch of embeddings
        embeddings = [
            (sample_embedding, False, {"video": 0.3}, 0.5),
            (sample_embedding, True, {"video": 0.8}, 0.9),
            (sample_embedding, False, {"video": 0.2}, 0.3)
        ]
        
        # Store batch
        results = await storage_manager.store_batch(embeddings)
        
        # Verify results
        assert len(results) == 3
        assert all(r.success for r in results)
        assert storage_manager.stats["total_stored"] == 3
        assert storage_manager.stats["anomalies_stored"] == 1
        assert storage_manager.stats["normal_stored"] == 2
    
    @pytest.mark.asyncio
    async def test_callback_failure_handling(
        self,
        mock_qdrant_client,
        sample_embedding
    ):
        """Test handling of callback failures"""
        # Create callback that fails
        callback = AsyncMock(side_effect=Exception("Callback error"))
        
        storage_manager = StorageManager(
            qdrant_client=mock_qdrant_client,
            anomaly_callback=callback
        )
        
        # Store anomaly (should not fail even if callback fails)
        result = await storage_manager.store_embedding(
            embedding=sample_embedding,
            is_anomaly=True,
            anomaly_scores={"video": 0.8},
            confidence=0.9
        )
        
        # Verify storage succeeded despite callback failure
        assert result.success is True
        assert storage_manager.stats["anomaly_callbacks_failed"] == 1
    
    def test_get_stats(self, storage_manager):
        """Test getting statistics"""
        # Simulate some storage operations
        storage_manager.stats["total_stored"] = 10
        storage_manager.stats["anomalies_stored"] = 3
        storage_manager.stats["normal_stored"] = 7
        storage_manager.stats["storage_failures"] = 2
        
        # Get stats
        stats = storage_manager.get_stats()
        
        # Verify computed metrics
        assert stats["total_stored"] == 10
        assert stats["anomalies_stored"] == 3
        assert stats["success_rate"] == pytest.approx(10 / 12, abs=0.01)
        assert stats["anomaly_rate"] == 0.3
    
    def test_get_stats_no_operations(self, storage_manager):
        """Test getting statistics with no operations"""
        stats = storage_manager.get_stats()
        
        assert stats["total_stored"] == 0
        assert stats["success_rate"] == 0.0
        assert stats["anomaly_rate"] == 0.0
    
    def test_reset_stats(self, storage_manager):
        """Test resetting statistics"""
        # Set some stats
        storage_manager.stats["total_stored"] = 10
        storage_manager.stats["anomalies_stored"] = 3
        
        # Reset
        storage_manager.reset_stats()
        
        # Verify reset
        assert storage_manager.stats["total_stored"] == 0
        assert storage_manager.stats["anomalies_stored"] == 0
    
    @pytest.mark.asyncio
    async def test_payload_structure(
        self,
        storage_manager,
        sample_embedding
    ):
        """Test that payload contains all required fields"""
        # Capture the upsert call
        captured_points = []
        storage_manager.qdrant_client.upsert = Mock(
            side_effect=lambda collection_name, points: captured_points.extend(points)
        )
        
        # Store embedding
        await storage_manager.store_embedding(
            embedding=sample_embedding,
            is_anomaly=True,
            anomaly_scores={"video": 0.8, "audio": 0.7},
            confidence=0.9,
            cause="gas_leak",
            severity="high"
        )
        
        # Verify payload structure
        assert len(captured_points) == 1
        point = captured_points[0]
        
        assert "timestamp" in point.payload
        assert "is_anomaly" in point.payload
        assert point.payload["is_anomaly"] is True
        assert "anomaly_scores" in point.payload
        assert "confidence" in point.payload
        assert "cause" in point.payload
        assert point.payload["cause"] == "gas_leak"
        assert "severity" in point.payload
        assert point.payload["severity"] == "high"
        assert "plant_zone" in point.payload
        assert "modality_status" in point.payload
    
    @pytest.mark.asyncio
    async def test_vector_structure(
        self,
        storage_manager,
        sample_embedding
    ):
        """Test that vectors are properly structured"""
        # Capture the upsert call
        captured_points = []
        storage_manager.qdrant_client.upsert = Mock(
            side_effect=lambda collection_name, points: captured_points.extend(points)
        )
        
        # Store embedding
        await storage_manager.store_embedding(
            embedding=sample_embedding,
            is_anomaly=False,
            anomaly_scores={"video": 0.3},
            confidence=0.5
        )
        
        # Verify vector structure
        assert len(captured_points) == 1
        point = captured_points[0]
        
        assert "video" in point.vector
        assert "audio" in point.vector
        assert "sensor" in point.vector
        assert isinstance(point.vector["video"], list)
        assert len(point.vector["video"]) == 512
        assert len(point.vector["audio"]) == 512
        assert len(point.vector["sensor"]) == 128


class TestStorageResultDataclass:
    """Test StorageResult dataclass"""
    
    def test_storage_result_creation(self):
        """Test creating StorageResult"""
        result = StorageResult(
            success=True,
            point_id="test-id",
            is_anomaly=True,
            error=None
        )
        
        assert result.success is True
        assert result.point_id == "test-id"
        assert result.is_anomaly is True
        assert result.error is None
    
    def test_storage_result_with_error(self):
        """Test creating StorageResult with error"""
        result = StorageResult(
            success=False,
            point_id="test-id",
            is_anomaly=False,
            error="Storage failed"
        )
        
        assert result.success is False
        assert result.error == "Storage failed"
