"""Unit tests for LabeledAnomalyStore"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch
import numpy as np

from src.agents.labeled_anomaly_store import (
    LabeledAnomalyStore,
    OperatorFeedback,
    LabeledAnomaly
)
from src.agents.input_collection_agent import MultimodalEmbedding, ModalityStatus


@pytest.fixture
def mock_qdrant_client():
    """Create mock Qdrant client"""
    client = Mock()
    client.upsert = Mock()
    client.count = Mock()
    client.scroll = Mock()
    client.retrieve = Mock()
    return client


@pytest.fixture
def labeled_anomaly_store(mock_qdrant_client):
    """Create LabeledAnomalyStore instance"""
    return LabeledAnomalyStore(
        qdrant_client=mock_qdrant_client,
        collection_name="test_labeled_anomalies"
    )


@pytest.fixture
def sample_embedding():
    """Create sample multimodal embedding"""
    return MultimodalEmbedding(
        timestamp=datetime.utcnow().isoformat(),
        video_embedding=np.random.rand(512).astype(np.float32),
        audio_embedding=np.random.rand(512).astype(np.float32),
        sensor_embedding=np.random.rand(128).astype(np.float32),
        metadata={
            "plant_zone": "Zone_A",
            "shift": "morning",
            "equipment_id": "EQ001"
        },
        modality_status={
            "video": ModalityStatus.AVAILABLE,
            "audio": ModalityStatus.AVAILABLE,
            "sensor": ModalityStatus.AVAILABLE
        }
    )


@pytest.mark.asyncio
async def test_store_labeled_anomaly_success(labeled_anomaly_store, sample_embedding):
    """Test successful storage of labeled anomaly"""
    # Store labeled anomaly
    anomaly_id = await labeled_anomaly_store.store_labeled_anomaly(
        embedding=sample_embedding,
        ground_truth_cause="gas_plume",
        ground_truth_severity="high",
        operator_feedback=OperatorFeedback.CONFIRMED,
        chemical_detected="Chlorine",
        operator_notes="Confirmed gas leak in Zone A",
        training_weight=1.0
    )
    
    # Verify result
    assert anomaly_id is not None
    assert labeled_anomaly_store.stats["total_labeled"] == 1
    assert labeled_anomaly_store.stats["confirmed_count"] == 1
    assert labeled_anomaly_store.stats["by_cause"]["gas_plume"] == 1
    assert labeled_anomaly_store.stats["by_severity"]["high"] == 1


@pytest.mark.asyncio
async def test_store_labeled_anomaly_false_positive(labeled_anomaly_store, sample_embedding):
    """Test storage of false positive"""
    anomaly_id = await labeled_anomaly_store.store_labeled_anomaly(
        embedding=sample_embedding,
        ground_truth_cause="none",
        ground_truth_severity="mild",
        operator_feedback=OperatorFeedback.FALSE_POSITIVE,
        operator_notes="False alarm"
    )
    
    assert anomaly_id is not None
    assert labeled_anomaly_store.stats["false_positive_count"] == 1
    assert labeled_anomaly_store.stats["confirmed_count"] == 0


@pytest.mark.asyncio
async def test_store_labeled_anomaly_false_negative(labeled_anomaly_store, sample_embedding):
    """Test storage of false negative"""
    anomaly_id = await labeled_anomaly_store.store_labeled_anomaly(
        embedding=sample_embedding,
        ground_truth_cause="pressure_spike",
        ground_truth_severity="medium",
        operator_feedback=OperatorFeedback.FALSE_NEGATIVE,
        operator_notes="Missed detection"
    )
    
    assert anomaly_id is not None
    assert labeled_anomaly_store.stats["false_negative_count"] == 1


@pytest.mark.asyncio
async def test_store_labeled_anomaly_no_vectors(labeled_anomaly_store):
    """Test storage fails when no vectors available"""
    # Create embedding with no vectors
    empty_embedding = MultimodalEmbedding(
        timestamp=datetime.utcnow().isoformat(),
        video_embedding=None,
        audio_embedding=None,
        sensor_embedding=None,
        metadata={"plant_zone": "Zone_A"},
        modality_status={
            "video": ModalityStatus.FAILED,
            "audio": ModalityStatus.FAILED,
            "sensor": ModalityStatus.FAILED
        }
    )
    
    anomaly_id = await labeled_anomaly_store.store_labeled_anomaly(
        embedding=empty_embedding,
        ground_truth_cause="gas_plume",
        ground_truth_severity="high",
        operator_feedback=OperatorFeedback.CONFIRMED
    )
    
    assert anomaly_id is None
    assert labeled_anomaly_store.stats["storage_failures"] == 1


@pytest.mark.asyncio
async def test_get_labeled_count(labeled_anomaly_store, mock_qdrant_client):
    """Test getting count of labeled anomalies"""
    # Mock count response
    mock_count_result = Mock()
    mock_count_result.count = 42
    mock_qdrant_client.count.return_value = mock_count_result
    
    count = await labeled_anomaly_store.get_labeled_count()
    
    assert count == 42
    mock_qdrant_client.count.assert_called_once()


@pytest.mark.asyncio
async def test_get_labeled_count_with_filters(labeled_anomaly_store, mock_qdrant_client):
    """Test getting count with filters"""
    mock_count_result = Mock()
    mock_count_result.count = 10
    mock_qdrant_client.count.return_value = mock_count_result
    
    count = await labeled_anomaly_store.get_labeled_count(
        cause="gas_plume",
        severity="high",
        feedback=OperatorFeedback.CONFIRMED
    )
    
    assert count == 10


@pytest.mark.asyncio
async def test_get_labeled_anomalies(labeled_anomaly_store, mock_qdrant_client):
    """Test retrieving labeled anomalies"""
    # Mock scroll response
    mock_point = Mock()
    mock_point.id = "test_id"
    mock_point.vector = {"video": [0.1] * 512}
    mock_point.payload = {
        "ground_truth_cause": "gas_plume",
        "ground_truth_severity": "high"
    }
    
    mock_qdrant_client.scroll.return_value = ([mock_point], None)
    
    anomalies = await labeled_anomaly_store.get_labeled_anomalies(limit=10)
    
    assert len(anomalies) == 1
    assert anomalies[0]["anomaly_id"] == "test_id"
    assert anomalies[0]["payload"]["ground_truth_cause"] == "gas_plume"


@pytest.mark.asyncio
async def test_update_operator_feedback(labeled_anomaly_store, mock_qdrant_client):
    """Test updating operator feedback"""
    # Mock retrieve response
    mock_point = Mock()
    mock_point.id = "test_id"
    mock_point.vector = {"video": [0.1] * 512}
    mock_point.payload = {
        "operator_feedback": "confirmed",
        "operator_notes": "Original notes"
    }
    
    mock_qdrant_client.retrieve.return_value = [mock_point]
    
    # Update feedback
    success = await labeled_anomaly_store.update_operator_feedback(
        anomaly_id="test_id",
        new_feedback=OperatorFeedback.FALSE_POSITIVE,
        operator_notes="Actually a false positive"
    )
    
    assert success is True
    mock_qdrant_client.upsert.assert_called()


@pytest.mark.asyncio
async def test_update_operator_feedback_not_found(labeled_anomaly_store, mock_qdrant_client):
    """Test updating feedback for non-existent anomaly"""
    mock_qdrant_client.retrieve.return_value = []
    
    success = await labeled_anomaly_store.update_operator_feedback(
        anomaly_id="nonexistent",
        new_feedback=OperatorFeedback.FALSE_POSITIVE
    )
    
    assert success is False


def test_get_stats(labeled_anomaly_store):
    """Test getting statistics"""
    # Manually set some stats
    labeled_anomaly_store.stats["total_labeled"] = 100
    labeled_anomaly_store.stats["confirmed_count"] = 80
    labeled_anomaly_store.stats["false_positive_count"] = 15
    labeled_anomaly_store.stats["false_negative_count"] = 5
    
    stats = labeled_anomaly_store.get_stats()
    
    assert stats["total_labeled"] == 100
    assert stats["confirmed_rate"] == 0.8
    assert stats["false_positive_rate"] == 0.15
    assert stats["false_negative_rate"] == 0.05


def test_reset_stats(labeled_anomaly_store):
    """Test resetting statistics"""
    # Set some stats
    labeled_anomaly_store.stats["total_labeled"] = 100
    labeled_anomaly_store.stats["confirmed_count"] = 80
    
    # Reset
    labeled_anomaly_store.reset_stats()
    
    assert labeled_anomaly_store.stats["total_labeled"] == 0
    assert labeled_anomaly_store.stats["confirmed_count"] == 0


def test_labeled_anomaly_to_dict():
    """Test LabeledAnomaly to_dict conversion"""
    embedding = MultimodalEmbedding(
        timestamp=datetime.utcnow().isoformat(),
        video_embedding=np.random.rand(512).astype(np.float32),
        audio_embedding=None,
        sensor_embedding=np.random.rand(128).astype(np.float32),
        metadata={"plant_zone": "Zone_A"},
        modality_status={
            "video": ModalityStatus.AVAILABLE,
            "audio": ModalityStatus.FAILED,
            "sensor": ModalityStatus.AVAILABLE
        }
    )
    
    labeled_anomaly = LabeledAnomaly(
        anomaly_id="test_id",
        embedding=embedding,
        ground_truth_cause="gas_plume",
        ground_truth_severity="high",
        chemical_detected="Chlorine",
        operator_feedback=OperatorFeedback.CONFIRMED,
        operator_notes="Test notes",
        training_weight=1.0,
        labeled_at=datetime.utcnow().isoformat()
    )
    
    result = labeled_anomaly.to_dict()
    
    assert result["anomaly_id"] == "test_id"
    assert result["ground_truth_cause"] == "gas_plume"
    assert result["ground_truth_severity"] == "high"
    assert result["chemical_detected"] == "Chlorine"
    assert result["operator_feedback"] == "confirmed"
    assert result["training_weight"] == 1.0
    assert result["plant_zone"] == "Zone_A"
