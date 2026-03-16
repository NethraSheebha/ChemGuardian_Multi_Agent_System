"""Tests for CauseDetectionAgent"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, AsyncMock
from datetime import datetime

from src.agents.cause_detection_agent import CauseDetectionAgent, CauseDetectionResult
from src.agents.cause_inference_engine import CauseInferenceEngine, CauseAnalysis
from src.agents.severity_classifier import SeverityClassifier
from src.agents.anomaly_detection_agent import AnomalyDetectionResult
from src.agents.input_collection_agent import MultimodalEmbedding, ModalityStatus


@pytest.fixture
def mock_qdrant_client():
    """Create mock Qdrant client"""
    return Mock()


@pytest.fixture
def mock_cause_engine():
    """Create mock cause inference engine"""
    engine = Mock(spec=CauseInferenceEngine)
    
    # Mock infer_cause to return analysis
    async def mock_infer(*args, **kwargs):
        return CauseAnalysis(
            primary_cause="gas_plume",
            contributing_factors=["audio_anomaly"],
            confidence=0.85,
            explanation="Detected gas_plume with 85% similarity to incident-123",
            similar_historical_incidents=["incident-123", "incident-456"]
        )
    
    engine.infer_cause = mock_infer
    engine.get_stats = Mock(return_value={})
    engine.reset_stats = Mock()
    
    return engine


@pytest.fixture
def mock_severity_classifier():
    """Create mock severity classifier"""
    classifier = Mock(spec=SeverityClassifier)
    
    # Mock classify_severity to return medium by default
    classifier.classify_severity = Mock(return_value="medium")
    classifier.get_stats = Mock(return_value={})
    classifier.reset_stats = Mock()
    
    return classifier


@pytest.fixture
def cause_detection_agent(mock_qdrant_client, mock_cause_engine, mock_severity_classifier):
    """Create CauseDetectionAgent instance"""
    return CauseDetectionAgent(
        qdrant_client=mock_qdrant_client,
        cause_inference_engine=mock_cause_engine,
        severity_classifier=mock_severity_classifier,
        processing_interval=1.0
    )


@pytest.fixture
def sample_anomaly_result():
    """Create sample anomaly detection result"""
    embedding = MultimodalEmbedding(
        timestamp="2024-01-01T12:00:00",
        video_embedding=np.random.rand(512).astype(np.float32),
        audio_embedding=np.random.rand(512).astype(np.float32),
        sensor_embedding=np.random.rand(128).astype(np.float32),
        metadata={
            "plant_zone": "Zone_A",
            "shift": "morning",
            "camera_id": "cam-1",
            "sensor_ids": ["sensor-1", "sensor-2"],
            "gas_concentration_ppm": 800
        },
        modality_status={
            "video": ModalityStatus.AVAILABLE,
            "audio": ModalityStatus.AVAILABLE,
            "sensor": ModalityStatus.AVAILABLE
        }
    )
    
    return AnomalyDetectionResult(
        embedding=embedding,
        is_anomaly=True,
        anomaly_scores={"video": 0.8, "audio": 0.7, "sensor": 3.0},
        per_modality_decisions={"video": True, "audio": True, "sensor": True},
        confidence=0.85,
        requires_temporal_confirmation=False
    )


@pytest.mark.asyncio
async def test_analyze_anomaly_success(cause_detection_agent, sample_anomaly_result):
    """Test successful anomaly analysis"""
    result = await cause_detection_agent.analyze_anomaly(sample_anomaly_result)
    
    assert isinstance(result, CauseDetectionResult)
    assert result.anomaly_result == sample_anomaly_result
    assert isinstance(result.cause_analysis, CauseAnalysis)
    assert result.severity in ["mild", "medium", "high"]
    assert result.timestamp is not None


@pytest.mark.asyncio
async def test_analyze_anomaly_calls_components(
    cause_detection_agent,
    sample_anomaly_result,
    mock_cause_engine,
    mock_severity_classifier
):
    """Test that analyze_anomaly calls both components"""
    result = await cause_detection_agent.analyze_anomaly(sample_anomaly_result)
    
    # Verify severity classifier was called
    assert mock_severity_classifier.classify_severity.called


@pytest.mark.asyncio
async def test_analyze_anomaly_updates_stats(cause_detection_agent, sample_anomaly_result):
    """Test that analyze_anomaly updates statistics"""
    initial_count = cause_detection_agent.stats["total_processed"]
    
    await cause_detection_agent.analyze_anomaly(sample_anomaly_result)
    
    assert cause_detection_agent.stats["total_processed"] == initial_count + 1
    assert cause_detection_agent.stats["successful_analyses"] > 0


@pytest.mark.asyncio
async def test_route_to_response_agent_mild(cause_detection_agent, sample_anomaly_result):
    """Test routing to mild response agent"""
    mild_callback = AsyncMock()
    cause_detection_agent.callbacks["mild"] = mild_callback
    
    # Create result with mild severity
    result = CauseDetectionResult(
        anomaly_result=sample_anomaly_result,
        cause_analysis=CauseAnalysis(
            primary_cause="gas_plume",
            contributing_factors=[],
            confidence=0.8,
            explanation="Test",
            similar_historical_incidents=[]
        ),
        severity="mild",
        timestamp=datetime.utcnow().isoformat()
    )
    
    await cause_detection_agent.route_to_response_agent(result)
    
    # Verify mild callback was called
    mild_callback.assert_called_once()
    assert cause_detection_agent.stats["mild_routed"] == 1


@pytest.mark.asyncio
async def test_route_to_response_agent_medium(cause_detection_agent, sample_anomaly_result):
    """Test routing to medium response agent"""
    medium_callback = AsyncMock()
    cause_detection_agent.callbacks["medium"] = medium_callback
    
    # Create result with medium severity
    result = CauseDetectionResult(
        anomaly_result=sample_anomaly_result,
        cause_analysis=CauseAnalysis(
            primary_cause="gas_plume",
            contributing_factors=[],
            confidence=0.8,
            explanation="Test",
            similar_historical_incidents=[]
        ),
        severity="medium",
        timestamp=datetime.utcnow().isoformat()
    )
    
    await cause_detection_agent.route_to_response_agent(result)
    
    # Verify medium callback was called
    medium_callback.assert_called_once()
    assert cause_detection_agent.stats["medium_routed"] == 1


@pytest.mark.asyncio
async def test_route_to_response_agent_high(cause_detection_agent, sample_anomaly_result):
    """Test routing to high response agent"""
    high_callback = AsyncMock()
    cause_detection_agent.callbacks["high"] = high_callback
    
    # Create result with high severity
    result = CauseDetectionResult(
        anomaly_result=sample_anomaly_result,
        cause_analysis=CauseAnalysis(
            primary_cause="gas_leak_with_hissing",
            contributing_factors=[],
            confidence=0.9,
            explanation="Test",
            similar_historical_incidents=[]
        ),
        severity="high",
        timestamp=datetime.utcnow().isoformat()
    )
    
    await cause_detection_agent.route_to_response_agent(result)
    
    # Verify high callback was called
    high_callback.assert_called_once()
    assert cause_detection_agent.stats["high_routed"] == 1


@pytest.mark.asyncio
async def test_route_to_response_agent_no_callback(cause_detection_agent, sample_anomaly_result):
    """Test routing when no callback is registered"""
    # Create result with severity that has no callback
    result = CauseDetectionResult(
        anomaly_result=sample_anomaly_result,
        cause_analysis=CauseAnalysis(
            primary_cause="gas_plume",
            contributing_factors=[],
            confidence=0.8,
            explanation="Test",
            similar_historical_incidents=[]
        ),
        severity="mild",
        timestamp=datetime.utcnow().isoformat()
    )
    
    # Should not raise exception
    await cause_detection_agent.route_to_response_agent(result)


@pytest.mark.asyncio
async def test_process_anomaly_stream(cause_detection_agent, sample_anomaly_result):
    """Test processing stream of anomalies"""
    # Create queue and stop event
    anomaly_queue = asyncio.Queue()
    stop_event = asyncio.Event()
    
    # Add anomaly to queue
    await anomaly_queue.put(sample_anomaly_result)
    
    # Start processing in background
    task = asyncio.create_task(
        cause_detection_agent.process_anomaly_stream(anomaly_queue, stop_event)
    )
    
    # Wait a bit for processing
    await asyncio.sleep(0.1)
    
    # Stop processing
    stop_event.set()
    await task
    
    # Verify anomaly was processed
    assert cause_detection_agent.stats["total_processed"] > 0


def test_get_stats(cause_detection_agent):
    """Test statistics retrieval"""
    cause_detection_agent.stats["total_processed"] = 10
    cause_detection_agent.stats["successful_analyses"] = 8
    
    stats = cause_detection_agent.get_stats()
    
    assert stats["total_processed"] == 10
    assert stats["successful_analyses"] == 8
    assert stats["success_rate"] == 0.8
    assert "cause_engine" in stats
    assert "severity_classifier" in stats


def test_reset_stats(cause_detection_agent):
    """Test statistics reset"""
    cause_detection_agent.stats["total_processed"] = 10
    cause_detection_agent.reset_stats()
    
    assert cause_detection_agent.stats["total_processed"] == 0
    assert cause_detection_agent.stats["successful_analyses"] == 0


@pytest.mark.asyncio
async def test_process_method(cause_detection_agent, sample_anomaly_result):
    """Test process method (BaseAgent interface)"""
    data = {"anomaly_result": sample_anomaly_result}
    
    result = await cause_detection_agent.process(data)
    
    assert isinstance(result, CauseDetectionResult)


@pytest.mark.asyncio
async def test_process_method_invalid_data(cause_detection_agent):
    """Test process method with invalid data"""
    data = {"invalid": "data"}
    
    with pytest.raises(ValueError):
        await cause_detection_agent.process(data)


@pytest.mark.asyncio
async def test_execute_method(cause_detection_agent):
    """Test execute method"""
    result = await cause_detection_agent.execute()
    
    assert result["status"] == "running"
    assert "stats" in result
