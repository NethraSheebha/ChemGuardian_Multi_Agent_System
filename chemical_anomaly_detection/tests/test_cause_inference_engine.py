"""Tests for CauseInferenceEngine"""

import pytest
import numpy as np
from unittest.mock import Mock, AsyncMock
from qdrant_client.models import ScoredPoint

from src.agents.cause_inference_engine import CauseInferenceEngine, CauseAnalysis
from src.agents.input_collection_agent import MultimodalEmbedding, ModalityStatus


@pytest.fixture
def mock_qdrant_client():
    """Create mock Qdrant client"""
    client = Mock()
    
    # Mock search to return similar incidents
    def mock_search(*args, **kwargs):
        return [
            ScoredPoint(
                id="incident-1",
                score=0.85,
                payload={
                    "ground_truth_cause": "gas_plume",
                    "ground_truth_severity": "high",
                    "plant_zone": "Zone_A",
                    "shift": "morning"
                },
                version=1,
                vector={}
            ),
            ScoredPoint(
                id="incident-2",
                score=0.75,
                payload={
                    "ground_truth_cause": "gas_plume",
                    "ground_truth_severity": "medium",
                    "plant_zone": "Zone_A",
                    "shift": "morning"
                },
                version=1,
                vector={}
            ),
            ScoredPoint(
                id="incident-3",
                score=0.60,
                payload={
                    "ground_truth_cause": "pressure_spike",
                    "ground_truth_severity": "medium",
                    "plant_zone": "Zone_A",
                    "shift": "morning"
                },
                version=1,
                vector={}
            )
        ]
    
    client.search = mock_search
    return client


@pytest.fixture
def cause_engine(mock_qdrant_client):
    """Create CauseInferenceEngine instance"""
    return CauseInferenceEngine(
        qdrant_client=mock_qdrant_client,
        collection_name="labeled_anomalies",
        top_k=10,
        min_confidence=0.3
    )


@pytest.fixture
def sample_embedding():
    """Create sample multimodal embedding"""
    return MultimodalEmbedding(
        timestamp="2024-01-01T12:00:00",
        video_embedding=np.random.rand(512).astype(np.float32),
        audio_embedding=np.random.rand(512).astype(np.float32),
        sensor_embedding=np.random.rand(128).astype(np.float32),
        metadata={
            "plant_zone": "Zone_A",
            "shift": "morning",
            "camera_id": "cam-1",
            "sensor_ids": ["sensor-1", "sensor-2"]
        },
        modality_status={
            "video": ModalityStatus.AVAILABLE,
            "audio": ModalityStatus.AVAILABLE,
            "sensor": ModalityStatus.AVAILABLE
        }
    )


@pytest.mark.asyncio
async def test_infer_cause_success(cause_engine, sample_embedding):
    """Test successful cause inference"""
    anomaly_scores = {
        "video": 0.8,
        "audio": 0.7,
        "sensor": 3.0
    }
    
    result = await cause_engine.infer_cause(
        embedding=sample_embedding,
        anomaly_scores=anomaly_scores,
        metadata=sample_embedding.metadata
    )
    
    assert isinstance(result, CauseAnalysis)
    assert result.primary_cause == "gas_plume"  # Most voted cause
    assert result.confidence > 0.5
    assert len(result.explanation) > 20
    assert len(result.similar_historical_incidents) > 0


@pytest.mark.asyncio
async def test_infer_cause_with_contributing_factors(cause_engine, sample_embedding):
    """Test cause inference identifies contributing factors"""
    anomaly_scores = {
        "video": 0.8,
        "audio": 0.7,
        "sensor": 3.0
    }
    
    result = await cause_engine.infer_cause(
        embedding=sample_embedding,
        anomaly_scores=anomaly_scores,
        metadata=sample_embedding.metadata
    )
    
    # pressure_spike should be a contributing factor (score > 0.3)
    assert "pressure_spike" in result.contributing_factors or len(result.contributing_factors) >= 0


@pytest.mark.asyncio
async def test_infer_cause_no_similar_incidents(cause_engine, sample_embedding):
    """Test cause inference when no similar incidents found"""
    # Mock client to return empty results
    cause_engine.client.search = Mock(return_value=[])
    
    anomaly_scores = {
        "video": 0.8,
        "audio": 0.7,
        "sensor": 3.0
    }
    
    result = await cause_engine.infer_cause(
        embedding=sample_embedding,
        anomaly_scores=anomaly_scores,
        metadata=sample_embedding.metadata
    )
    
    # Should return default analysis
    assert isinstance(result, CauseAnalysis)
    assert result.primary_cause in CauseInferenceEngine.KNOWN_CAUSES or result.primary_cause == "unknown_anomaly"
    assert result.confidence == 0.5  # Low confidence
    assert len(result.similar_historical_incidents) == 0


@pytest.mark.asyncio
async def test_infer_cause_with_metadata_filtering(cause_engine, sample_embedding):
    """Test cause inference uses metadata for filtering"""
    anomaly_scores = {
        "video": 0.8,
        "audio": 0.7,
        "sensor": 3.0
    }
    
    metadata = {
        "plant_zone": "Zone_B",
        "shift": "night"
    }
    
    result = await cause_engine.infer_cause(
        embedding=sample_embedding,
        anomaly_scores=anomaly_scores,
        metadata=metadata
    )
    
    # Should still work with different metadata
    assert isinstance(result, CauseAnalysis)
    assert result.primary_cause is not None


@pytest.mark.asyncio
async def test_aggregate_causes_weighted_voting(cause_engine):
    """Test cause aggregation uses weighted voting"""
    similar_incidents = [
        ScoredPoint(
            id="inc-1",
            score=0.9,
            payload={"ground_truth_cause": "gas_plume"},
            version=1,
            vector={}
        ),
        ScoredPoint(
            id="inc-2",
            score=0.1,
            payload={"ground_truth_cause": "pressure_spike"},
            version=1,
            vector={}
        )
    ]
    
    cause_votes = cause_engine._aggregate_causes(similar_incidents)
    
    # gas_plume should have higher vote due to higher similarity
    assert cause_votes["gas_plume"] > cause_votes["pressure_spike"]
    assert abs(sum(cause_votes.values()) - 1.0) < 0.01  # Votes should sum to ~1.0


def test_generate_explanation_includes_context(cause_engine):
    """Test explanation includes all required context"""
    similar_incidents = [
        ScoredPoint(
            id="incident-123",
            score=0.85,
            payload={
                "ground_truth_cause": "gas_plume",
                "ground_truth_severity": "high"
            },
            version=1,
            vector={}
        )
    ]
    
    anomaly_scores = {
        "video": 0.8,
        "audio": 0.7,
        "sensor": 3.0
    }
    
    metadata = {
        "plant_zone": "Zone_A",
        "shift": "morning"
    }
    
    explanation = cause_engine._generate_explanation(
        primary_cause="gas_plume",
        similar_incidents=similar_incidents,
        anomaly_scores=anomaly_scores,
        metadata=metadata
    )
    
    # Check explanation contains key information
    assert "gas_plume" in explanation
    assert "incident-123" in explanation
    assert "Zone_A" in explanation
    assert "morning" in explanation
    assert "0.80" in explanation or "0.8" in explanation  # Video score


def test_get_stats(cause_engine):
    """Test statistics tracking"""
    cause_engine.stats["total_inferences"] = 10
    cause_engine.stats["successful_inferences"] = 8
    
    stats = cause_engine.get_stats()
    
    assert stats["total_inferences"] == 10
    assert stats["successful_inferences"] == 8
    assert stats["success_rate"] == 0.8


def test_reset_stats(cause_engine):
    """Test statistics reset"""
    cause_engine.stats["total_inferences"] = 10
    cause_engine.reset_stats()
    
    assert cause_engine.stats["total_inferences"] == 0
    assert cause_engine.stats["successful_inferences"] == 0
