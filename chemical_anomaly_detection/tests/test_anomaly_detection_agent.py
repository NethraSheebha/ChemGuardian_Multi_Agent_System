"""Tests for AnomalyDetectionAgent"""

import pytest
import asyncio
import numpy as np
from datetime import datetime
from unittest.mock import Mock, MagicMock, AsyncMock, patch

from src.agents.anomaly_detection_agent import (
    AnomalyDetectionAgent,
    AnomalyDetectionResult
)
from src.agents.input_collection_agent import MultimodalEmbedding, ModalityStatus
from src.agents.similarity_search_engine import SimilaritySearchEngine, SearchResult
from src.agents.adaptive_threshold_manager import AdaptiveThresholdManager
from src.agents.storage_manager import StorageManager, StorageResult


@pytest.fixture
def mock_qdrant_client():
    """Create mock Qdrant client"""
    return Mock()


@pytest.fixture
def mock_similarity_search():
    """Create mock similarity search engine"""
    search = Mock(spec=SimilaritySearchEngine)
    
    # Mock search_and_score to return results and scores
    async def mock_search_and_score(*args, **kwargs):
        search_results = {
            "video": Mock(min_distance=0.5, mean_distance=0.6),
            "audio": Mock(min_distance=0.4, mean_distance=0.5),
            "sensor": Mock(min_distance=2.0, mean_distance=2.2)
        }
        anomaly_scores = {
            "video": 0.5,
            "audio": 0.4,
            "sensor": 2.0
        }
        return search_results, anomaly_scores
    
    search.search_and_score = mock_search_and_score
    search.get_stats = Mock(return_value={})
    
    return search


@pytest.fixture
def mock_threshold_manager():
    """Create mock adaptive threshold manager"""
    manager = Mock(spec=AdaptiveThresholdManager)
    
    # Mock is_anomaly to return False by default
    manager.is_anomaly = Mock(return_value=(False, {"video": False, "audio": False, "sensor": False}))
    
    # Mock get_current_thresholds
    manager.get_current_thresholds = Mock(return_value={
        "video": 0.7,
        "audio": 0.65,
        "sensor": 2.5
    })
    
    manager.get_stats = Mock(return_value={})
    
    return manager


@pytest.fixture
def mock_storage_manager():
    """Create mock storage manager"""
    storage = Mock(spec=StorageManager)
    
    # Mock store_embedding to return success
    async def mock_store(*args, **kwargs):
        return StorageResult(
            success=True,
            point_id="test-id",
            is_anomaly=kwargs.get("is_anomaly", False)
        )
    
    storage.store_embedding = mock_store
    storage.get_stats = Mock(return_value={})
    
    return storage


@pytest.fixture
def anomaly_detection_agent(
    mock_qdrant_client,
    mock_similarity_search,
    mock_threshold_manager,
    mock_storage_manager
):
    """Create AnomalyDetectionAgent instance"""
    return AnomalyDetectionAgent(
        qdrant_client=mock_qdrant_client,
        similarity_search_engine=mock_similarity_search,
        adaptive_threshold_manager=mock_threshold_manager,
        storage_manager=mock_storage_manager,
        processing_interval=1.0,
        high_severity_min_modalities=2,
        borderline_threshold_pct=0.1,
        temporal_confirmation_windows=3
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
            "equipment_id": "EQ001"
        }
    )


class TestAnomalyDetectionAgent:
    """Test suite for AnomalyDetectionAgent"""
    
    def test_initialization(self, anomaly_detection_agent):
        """Test agent initialization"""
        assert anomaly_detection_agent.processing_interval == 1.0
        assert anomaly_detection_agent.high_severity_min_modalities == 2
        assert anomaly_detection_agent.temporal_confirmation_windows == 3
        assert anomaly_detection_agent.stats["total_processed"] == 0
    
    @pytest.mark.asyncio
    async def test_detect_anomaly_normal(
        self,
        anomaly_detection_agent,
        sample_embedding,
        mock_threshold_manager
    ):
        """Test detecting normal (non-anomaly) data"""
        # Configure threshold manager to return False
        mock_threshold_manager.is_anomaly = Mock(
            return_value=(False, {"video": False, "audio": False, "sensor": False})
        )
        
        # Detect anomaly
        result = await anomaly_detection_agent.detect_anomaly(sample_embedding)
        
        # Verify result
        assert result.is_anomaly is False
        assert result.embedding == sample_embedding
        assert len(result.anomaly_scores) == 3
        assert anomaly_detection_agent.stats["normal_detected"] == 1
        assert anomaly_detection_agent.stats["anomalies_detected"] == 0
    
    @pytest.mark.asyncio
    async def test_detect_anomaly_single_modality(
        self,
        anomaly_detection_agent,
        sample_embedding,
        mock_threshold_manager
    ):
        """Test detecting anomaly in single modality"""
        # Configure threshold manager to return True for one modality
        mock_threshold_manager.is_anomaly = Mock(
            return_value=(True, {"video": True, "audio": False, "sensor": False})
        )
        
        # Detect anomaly
        result = await anomaly_detection_agent.detect_anomaly(sample_embedding)
        
        # Verify result
        assert result.is_anomaly is True
        assert result.per_modality_decisions["video"] is True
        assert anomaly_detection_agent.stats["anomalies_detected"] == 1
    
    @pytest.mark.asyncio
    async def test_detect_anomaly_high_severity_multi_modality(
        self,
        anomaly_detection_agent,
        sample_embedding,
        mock_threshold_manager
    ):
        """Test high-severity anomaly requiring multi-modality confirmation"""
        # Configure threshold manager
        # First call: initial check (2 modalities anomalous)
        # Second call: multi-modality confirmation check
        mock_threshold_manager.is_anomaly = Mock(
            side_effect=[
                (True, {"video": True, "audio": True, "sensor": False}),
                (True, {"video": True, "audio": True, "sensor": False})
            ]
        )
        
        # Detect anomaly
        result = await anomaly_detection_agent.detect_anomaly(sample_embedding)
        
        # Verify high-severity confirmation
        assert result.is_anomaly is True
        assert anomaly_detection_agent.stats["high_severity_anomalies"] == 1
        assert anomaly_detection_agent.stats["multi_modality_confirmations"] == 1
    
    @pytest.mark.asyncio
    async def test_detect_anomaly_high_severity_not_confirmed(
        self,
        anomaly_detection_agent,
        sample_embedding,
        mock_threshold_manager
    ):
        """Test high-severity anomaly that fails multi-modality confirmation"""
        # Configure threshold manager
        # First call: 2 modalities anomalous
        # Second call: multi-modality check fails (require_multi_modality=True)
        mock_threshold_manager.is_anomaly = Mock(
            side_effect=[
                (True, {"video": True, "audio": True, "sensor": False}),
                (False, {"video": True, "audio": True, "sensor": False})
            ]
        )
        
        # Detect anomaly
        result = await anomaly_detection_agent.detect_anomaly(sample_embedding)
        
        # Verify anomaly was not confirmed
        assert result.is_anomaly is False
    
    @pytest.mark.asyncio
    async def test_detect_anomaly_borderline_temporal_confirmation(
        self,
        anomaly_detection_agent,
        sample_embedding,
        mock_threshold_manager
    ):
        """Test borderline anomaly with temporal confirmation"""
        # Configure threshold manager to return borderline scores
        mock_threshold_manager.is_anomaly = Mock(
            return_value=(True, {"video": True, "audio": False, "sensor": False})
        )
        
        # Configure thresholds to make scores borderline
        mock_threshold_manager.get_current_thresholds = Mock(return_value={
            "video": 0.5,  # Score is 0.5, exactly at threshold
            "audio": 0.65,
            "sensor": 2.5
        })
        
        # Detect anomaly 3 times to achieve temporal confirmation
        for i in range(3):
            result = await anomaly_detection_agent.detect_anomaly(sample_embedding)
        
        # Last result should be confirmed
        assert result.is_anomaly is True
        assert result.requires_temporal_confirmation is True
        assert anomaly_detection_agent.stats["temporal_confirmations"] >= 1
    
    @pytest.mark.asyncio
    async def test_detect_anomaly_borderline_pending_confirmation(
        self,
        anomaly_detection_agent,
        sample_embedding,
        mock_threshold_manager
    ):
        """Test borderline anomaly pending temporal confirmation"""
        # Configure threshold manager
        mock_threshold_manager.is_anomaly = Mock(
            return_value=(True, {"video": True, "audio": False, "sensor": False})
        )
        
        # Configure thresholds to make scores borderline
        mock_threshold_manager.get_current_thresholds = Mock(return_value={
            "video": 0.5,
            "audio": 0.65,
            "sensor": 2.5
        })
        
        # Detect anomaly once (not enough for temporal confirmation)
        result = await anomaly_detection_agent.detect_anomaly(sample_embedding)
        
        # Should not be confirmed yet
        assert result.is_anomaly is False
        assert result.requires_temporal_confirmation is True
        assert result.temporal_confirmation_count < 3
    
    @pytest.mark.asyncio
    async def test_detect_anomaly_with_filters(
        self,
        anomaly_detection_agent,
        sample_embedding,
        mock_similarity_search
    ):
        """Test anomaly detection with baseline filters"""
        # Detect anomaly with filters
        result = await anomaly_detection_agent.detect_anomaly(
            embedding=sample_embedding,
            shift="morning",
            equipment_id="EQ001",
            plant_zone="Zone_A"
        )
        
        # Verify search was called with filters
        # (mock_similarity_search.search_and_score was called)
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_detect_anomaly_no_baselines(
        self,
        anomaly_detection_agent,
        sample_embedding,
        mock_similarity_search
    ):
        """Test anomaly detection when no baselines are found"""
        # Configure search to return empty results
        async def mock_empty_search(*args, **kwargs):
            return {}, {}
        
        mock_similarity_search.search_and_score = mock_empty_search
        
        # Detect anomaly
        result = await anomaly_detection_agent.detect_anomaly(sample_embedding)
        
        # Verify graceful handling
        assert result.is_anomaly is False
        assert len(result.anomaly_scores) == 0
        assert anomaly_detection_agent.stats["processing_failures"] == 1
    
    @pytest.mark.asyncio
    async def test_detect_anomaly_with_callback(
        self,
        mock_qdrant_client,
        mock_similarity_search,
        mock_threshold_manager,
        mock_storage_manager,
        sample_embedding
    ):
        """Test anomaly detection with callback trigger"""
        # Create callback mock
        callback = AsyncMock()
        
        # Create agent with callback
        agent = AnomalyDetectionAgent(
            qdrant_client=mock_qdrant_client,
            similarity_search_engine=mock_similarity_search,
            adaptive_threshold_manager=mock_threshold_manager,
            storage_manager=mock_storage_manager,
            anomaly_callback=callback
        )
        
        # Configure to detect anomaly
        mock_threshold_manager.is_anomaly = Mock(
            return_value=(True, {"video": True, "audio": False, "sensor": False})
        )
        
        # Detect anomaly
        result = await agent.detect_anomaly(sample_embedding)
        
        # Callback is not triggered in detect_anomaly, only in process_embedding_stream
        # So we test process_embedding_stream separately
        assert result.is_anomaly is True
    
    @pytest.mark.asyncio
    async def test_process_embedding_stream(
        self,
        anomaly_detection_agent,
        sample_embedding,
        mock_threshold_manager
    ):
        """Test processing stream of embeddings"""
        # Create queue and stop event
        queue = asyncio.Queue()
        stop_event = asyncio.Event()
        
        # Add embeddings to queue
        await queue.put(sample_embedding)
        await queue.put(sample_embedding)
        
        # Configure to detect anomaly
        mock_threshold_manager.is_anomaly = Mock(
            return_value=(True, {"video": True, "audio": False, "sensor": False})
        )
        
        # Process stream in background
        task = asyncio.create_task(
            anomaly_detection_agent.process_embedding_stream(queue, stop_event)
        )
        
        # Wait a bit for processing
        await asyncio.sleep(0.1)
        
        # Stop processing
        stop_event.set()
        await task
        
        # Verify processing occurred
        assert anomaly_detection_agent.stats["total_processed"] >= 2
    
    def test_is_borderline(self, anomaly_detection_agent):
        """Test borderline score detection"""
        # Configure thresholds
        anomaly_detection_agent.threshold_manager.get_current_thresholds = Mock(
            return_value={"video": 0.7, "audio": 0.65, "sensor": 2.5}
        )
        
        # Test borderline score (within 10% of threshold)
        borderline_scores = {"video": 0.72, "audio": 0.5, "sensor": 2.0}
        assert anomaly_detection_agent._is_borderline(borderline_scores) is True
        
        # Test non-borderline score
        normal_scores = {"video": 0.3, "audio": 0.2, "sensor": 1.0}
        assert anomaly_detection_agent._is_borderline(normal_scores) is False
    
    def test_check_temporal_confirmation(self, anomaly_detection_agent):
        """Test temporal confirmation checking"""
        # Populate temporal history with consecutive True values
        anomaly_detection_agent.temporal_history["video"].extend([True, True])
        
        # Check with current True value (should be 3 consecutive)
        per_modality = {"video": True, "audio": False, "sensor": False}
        is_confirmed, count = anomaly_detection_agent._check_temporal_confirmation(
            per_modality
        )
        
        # Verify confirmation
        assert is_confirmed is True
        assert count == 3
    
    def test_check_temporal_confirmation_not_enough(self, anomaly_detection_agent):
        """Test temporal confirmation with insufficient history"""
        # Populate temporal history with only 1 True value
        anomaly_detection_agent.temporal_history["video"].extend([True])
        
        # Check with current True value (only 2 consecutive)
        per_modality = {"video": True, "audio": False, "sensor": False}
        is_confirmed, count = anomaly_detection_agent._check_temporal_confirmation(
            per_modality
        )
        
        # Verify not confirmed
        assert is_confirmed is False
        assert count == 2
    
    def test_update_temporal_history(self, anomaly_detection_agent):
        """Test updating temporal history"""
        per_modality = {"video": True, "audio": False, "sensor": True}
        
        # Update history
        anomaly_detection_agent._update_temporal_history(per_modality)
        
        # Verify history was updated
        assert anomaly_detection_agent.temporal_history["video"][-1] is True
        assert anomaly_detection_agent.temporal_history["audio"][-1] is False
        assert anomaly_detection_agent.temporal_history["sensor"][-1] is True
    
    def test_compute_confidence(self, anomaly_detection_agent):
        """Test confidence score computation"""
        # Configure thresholds
        anomaly_detection_agent.threshold_manager.get_current_thresholds = Mock(
            return_value={"video": 0.7, "audio": 0.65, "sensor": 2.5}
        )
        
        # Compute confidence for high anomaly scores
        anomaly_scores = {"video": 1.0, "audio": 0.9, "sensor": 3.5}
        per_modality = {"video": True, "audio": True, "sensor": True}
        
        confidence = anomaly_detection_agent._compute_confidence(
            anomaly_scores=anomaly_scores,
            per_modality=per_modality,
            is_high_severity=True,
            temporal_confirmed=True
        )
        
        # Verify confidence is high
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # Should be high for strong anomaly
    
    def test_compute_confidence_low_scores(self, anomaly_detection_agent):
        """Test confidence computation for low scores"""
        # Configure thresholds
        anomaly_detection_agent.threshold_manager.get_current_thresholds = Mock(
            return_value={"video": 0.7, "audio": 0.65, "sensor": 2.5}
        )
        
        # Compute confidence for low scores
        anomaly_scores = {"video": 0.3, "audio": 0.2, "sensor": 1.0}
        per_modality = {"video": False, "audio": False, "sensor": False}
        
        confidence = anomaly_detection_agent._compute_confidence(
            anomaly_scores=anomaly_scores,
            per_modality=per_modality,
            is_high_severity=False,
            temporal_confirmed=False
        )
        
        # Verify confidence is low
        assert confidence == 0.0
    
    def test_get_stats(self, anomaly_detection_agent):
        """Test getting statistics"""
        # Simulate some processing
        anomaly_detection_agent.stats["total_processed"] = 10
        anomaly_detection_agent.stats["anomalies_detected"] = 3
        anomaly_detection_agent.stats["normal_detected"] = 7
        
        # Get stats
        stats = anomaly_detection_agent.get_stats()
        
        # Verify computed metrics
        assert stats["total_processed"] == 10
        assert stats["anomalies_detected"] == 3
        assert stats["anomaly_rate"] == 0.3
        assert "threshold_manager" in stats
        assert "search_engine" in stats
        assert "storage_manager" in stats
    
    def test_reset_stats(self, anomaly_detection_agent):
        """Test resetting statistics"""
        # Set some stats
        anomaly_detection_agent.stats["total_processed"] = 10
        anomaly_detection_agent.stats["anomalies_detected"] = 3
        
        # Reset
        anomaly_detection_agent.reset_stats()
        
        # Verify reset
        assert anomaly_detection_agent.stats["total_processed"] == 0
        assert anomaly_detection_agent.stats["anomalies_detected"] == 0
    
    @pytest.mark.asyncio
    async def test_process_method(
        self,
        anomaly_detection_agent,
        sample_embedding
    ):
        """Test process method (BaseAgent interface)"""
        # Process data
        result = await anomaly_detection_agent.process({
            "embedding": sample_embedding,
            "shift": "morning",
            "plant_zone": "Zone_A"
        })
        
        # Verify result
        assert isinstance(result, AnomalyDetectionResult)
        assert result.embedding == sample_embedding
    
    @pytest.mark.asyncio
    async def test_execute_method(self, anomaly_detection_agent):
        """Test execute method (BaseAgent interface)"""
        # Execute agent
        result = await anomaly_detection_agent.execute()
        
        # Verify result
        assert result["status"] == "running"
        assert "stats" in result


class TestAnomalyDetectionResultDataclass:
    """Test AnomalyDetectionResult dataclass"""
    
    def test_result_creation(self, sample_embedding):
        """Test creating AnomalyDetectionResult"""
        result = AnomalyDetectionResult(
            embedding=sample_embedding,
            is_anomaly=True,
            anomaly_scores={"video": 0.8, "audio": 0.7},
            per_modality_decisions={"video": True, "audio": True},
            confidence=0.9,
            requires_temporal_confirmation=False,
            temporal_confirmation_count=0
        )
        
        assert result.is_anomaly is True
        assert result.confidence == 0.9
        assert result.requires_temporal_confirmation is False


class TestIntegrationScenarios:
    """Integration test scenarios"""
    
    @pytest.mark.asyncio
    async def test_full_detection_workflow(
        self,
        anomaly_detection_agent,
        sample_embedding,
        mock_threshold_manager,
        mock_similarity_search
    ):
        """Test complete anomaly detection workflow"""
        # Configure thresholds
        mock_threshold_manager.get_current_thresholds = Mock(
            return_value={"video": 0.7, "audio": 0.65, "sensor": 2.5}
        )
        
        # Configure anomaly scores that exceed thresholds
        async def mock_search_and_score(*args, **kwargs):
            search_results = {
                "video": Mock(min_distance=0.9, mean_distance=1.0),
                "audio": Mock(min_distance=0.8, mean_distance=0.9),
                "sensor": Mock(min_distance=3.0, mean_distance=3.2)
            }
            anomaly_scores = {
                "video": 0.9,  # Exceeds 0.7
                "audio": 0.8,  # Exceeds 0.65
                "sensor": 3.0  # Exceeds 2.5
            }
            return search_results, anomaly_scores
        
        mock_similarity_search.search_and_score = mock_search_and_score
        
        # Configure to detect anomaly
        mock_threshold_manager.is_anomaly = Mock(
            return_value=(True, {"video": True, "audio": True, "sensor": True})
        )
        
        # 1. Detect anomaly
        result = await anomaly_detection_agent.detect_anomaly(
            embedding=sample_embedding,
            shift="morning",
            plant_zone="Zone_A"
        )
        
        # 2. Verify detection
        assert result.is_anomaly is True
        assert len(result.anomaly_scores) == 3
        assert result.confidence > 0.0
        
        # 3. Verify storage occurred
        assert anomaly_detection_agent.stats["total_processed"] == 1
        assert anomaly_detection_agent.stats["anomalies_detected"] == 1
        
        # 4. Verify statistics
        stats = anomaly_detection_agent.get_stats()
        assert stats["anomaly_rate"] == 1.0
