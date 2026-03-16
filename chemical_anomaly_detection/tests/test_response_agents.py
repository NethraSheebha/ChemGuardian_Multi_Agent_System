"""Tests for Risk Response Agents"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from src.agents.response_strategy_engine import ResponseStrategyEngine, ResponseStrategy
from src.agents.mild_response_agent import MildResponseAgent
from src.agents.medium_response_agent import MediumResponseAgent
from src.agents.high_response_agent import HighResponseAgent
from src.agents.cause_detection_agent import CauseDetectionResult
from src.agents.cause_inference_engine import CauseAnalysis
from src.agents.anomaly_detection_agent import AnomalyDetectionResult
from src.agents.input_collection_agent import MultimodalEmbedding, ModalityStatus
from src.integrations.msds_integration import MSDSIntegration, ChemicalInfo
from src.integrations.sop_integration import SOPIntegration


@pytest.fixture
def mock_qdrant_client():
    """Create mock Qdrant client"""
    client = Mock()
    
    # Mock scroll method for response strategy search
    mock_point = Mock()
    mock_point.id = "response-1"
    mock_point.score = 0.9
    mock_point.payload = {
        "severity": "medium",
        "plant_zone": "Zone_A",
        "cause": "gas_plume",
        "successful_actions": ["isolate_area", "increase_ventilation"],
        "effectiveness_score": 0.85
    }
    
    client.scroll = Mock(return_value=([mock_point], None))
    
    return client


@pytest.fixture
def mock_msds_integration():
    """Create mock MSDS integration"""
    msds = Mock(spec=MSDSIntegration)
    
    # Mock get_chemical_info
    msds.get_chemical_info = Mock(return_value=ChemicalInfo(
        name="Chlorine",
        cas_number="7782-50-5",
        exposure_limits={"TWA": 0.5, "STEL": 1.0, "IDLH": 10.0},
        emergency_procedures=[
            "Evacuate area immediately",
            "Activate emergency ventilation",
            "Notify hazmat team"
        ],
        ppe_requirements=[
            "Full-face respirator with chlorine cartridge",
            "Chemical-resistant suit",
            "Rubber gloves and boots"
        ]
    ))
    
    return msds


@pytest.fixture
def mock_sop_integration():
    """Create mock SOP integration"""
    sop = Mock(spec=SOPIntegration)
    
    # Mock get_procedures
    sop.get_procedures = Mock(return_value=[
        "Activate Zone A emergency shutdown",
        "Evacuate all personnel from Zone A",
        "Seal Zone A ventilation system",
        "Deploy emergency response team"
    ])
    
    return sop


@pytest.fixture
def response_strategy_engine(mock_qdrant_client, mock_msds_integration, mock_sop_integration):
    """Create ResponseStrategyEngine instance"""
    return ResponseStrategyEngine(
        qdrant_client=mock_qdrant_client,
        msds_integration=mock_msds_integration,
        sop_integration=mock_sop_integration,
        collection_name="response_strategies",
        top_k=5
    )


@pytest.fixture
def mild_response_agent(mock_qdrant_client, response_strategy_engine):
    """Create MildResponseAgent instance"""
    return MildResponseAgent(
        qdrant_client=mock_qdrant_client,
        response_strategy_engine=response_strategy_engine,
        processing_interval=1.0
    )


@pytest.fixture
def medium_response_agent(mock_qdrant_client, response_strategy_engine):
    """Create MediumResponseAgent instance"""
    return MediumResponseAgent(
        qdrant_client=mock_qdrant_client,
        response_strategy_engine=response_strategy_engine,
        processing_interval=1.0
    )


@pytest.fixture
def high_response_agent(mock_qdrant_client, response_strategy_engine):
    """Create HighResponseAgent instance"""
    return HighResponseAgent(
        qdrant_client=mock_qdrant_client,
        response_strategy_engine=response_strategy_engine,
        processing_interval=1.0
    )


@pytest.fixture
def sample_cause_detection_result():
    """Create sample cause detection result"""
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
    
    anomaly_result = AnomalyDetectionResult(
        embedding=embedding,
        is_anomaly=True,
        anomaly_scores={"video": 0.8, "audio": 0.7, "sensor": 3.0},
        per_modality_decisions={"video": True, "audio": True, "sensor": True},
        confidence=0.85,
        requires_temporal_confirmation=False
    )
    
    cause_analysis = CauseAnalysis(
        primary_cause="gas_plume",
        contributing_factors=["audio_anomaly"],
        confidence=0.85,
        explanation="Detected gas_plume with 85% similarity to incident-123",
        similar_historical_incidents=["incident-123", "incident-456"]
    )
    
    return CauseDetectionResult(
        anomaly_result=anomaly_result,
        cause_analysis=cause_analysis,
        severity="medium",
        timestamp=datetime.utcnow().isoformat()
    )


# ResponseStrategyEngine Tests

@pytest.mark.asyncio
async def test_response_strategy_engine_get_strategy(
    response_strategy_engine,
    sample_cause_detection_result
):
    """Test getting response strategy"""
    cause_analysis = sample_cause_detection_result.cause_analysis
    metadata = sample_cause_detection_result.anomaly_result.embedding.metadata
    
    strategy = await response_strategy_engine.get_response_strategy(
        cause=cause_analysis,
        severity="medium",
        metadata=metadata
    )
    
    assert isinstance(strategy, ResponseStrategy)
    assert isinstance(strategy.actions, list)
    assert isinstance(strategy.sop_procedures, list)
    assert isinstance(strategy.similar_incidents, list)
    assert 0.0 <= strategy.confidence <= 1.0


@pytest.mark.asyncio
async def test_response_strategy_engine_msds_integration(
    response_strategy_engine,
    mock_msds_integration
):
    """Test MSDS integration in response strategy"""
    cause_analysis = CauseAnalysis(
        primary_cause="gas_leak_chlorine",
        contributing_factors=[],
        confidence=0.9,
        explanation="Test",
        similar_historical_incidents=[]
    )
    
    strategy = await response_strategy_engine.get_response_strategy(
        cause=cause_analysis,
        severity="high",
        metadata={"plant_zone": "Zone_A"}
    )
    
    assert strategy.msds_info is not None
    assert strategy.msds_info.name == "Chlorine"
    assert len(strategy.msds_info.emergency_procedures) > 0


@pytest.mark.asyncio
async def test_response_strategy_engine_sop_integration(
    response_strategy_engine,
    mock_sop_integration
):
    """Test SOP integration in response strategy"""
    cause_analysis = CauseAnalysis(
        primary_cause="gas_plume",
        contributing_factors=[],
        confidence=0.85,
        explanation="Test",
        similar_historical_incidents=[]
    )
    
    strategy = await response_strategy_engine.get_response_strategy(
        cause=cause_analysis,
        severity="medium",
        metadata={"plant_zone": "Zone_A"}
    )
    
    assert len(strategy.sop_procedures) > 0
    mock_sop_integration.get_procedures.assert_called_once()


def test_response_strategy_engine_stats(response_strategy_engine):
    """Test statistics tracking"""
    response_strategy_engine.stats["total_queries"] = 10
    response_strategy_engine.stats["successful_queries"] = 8
    
    stats = response_strategy_engine.get_stats()
    
    assert stats["total_queries"] == 10
    assert stats["successful_queries"] == 8
    assert stats["success_rate"] == 0.8


# MildResponseAgent Tests

@pytest.mark.asyncio
async def test_mild_response_agent_execute_response(
    mild_response_agent,
    sample_cause_detection_result
):
    """Test mild response execution"""
    # Set severity to mild
    sample_cause_detection_result.severity = "mild"
    
    result = await mild_response_agent.execute_response(sample_cause_detection_result)
    
    assert result["severity"] == "mild"
    assert "cause" in result
    assert "actions_executed" in result
    assert "incident_log" in result
    assert "timestamp" in result


@pytest.mark.asyncio
async def test_mild_response_agent_updates_stats(
    mild_response_agent,
    sample_cause_detection_result
):
    """Test that mild response updates statistics"""
    sample_cause_detection_result.severity = "mild"
    
    initial_count = mild_response_agent.stats["total_incidents"]
    
    await mild_response_agent.execute_response(sample_cause_detection_result)
    
    assert mild_response_agent.stats["total_incidents"] == initial_count + 1
    assert mild_response_agent.stats["successful_responses"] > 0


@pytest.mark.asyncio
async def test_mild_response_agent_process_method(
    mild_response_agent,
    sample_cause_detection_result
):
    """Test process method (BaseAgent interface)"""
    sample_cause_detection_result.severity = "mild"
    data = {"cause_detection_result": sample_cause_detection_result}
    
    result = await mild_response_agent.process(data)
    
    assert result["severity"] == "mild"


def test_mild_response_agent_get_stats(mild_response_agent):
    """Test statistics retrieval"""
    mild_response_agent.stats["total_incidents"] = 10
    mild_response_agent.stats["successful_responses"] = 9
    
    stats = mild_response_agent.get_stats()
    
    assert stats["total_incidents"] == 10
    assert stats["successful_responses"] == 9
    assert stats["success_rate"] == 0.9


# MediumResponseAgent Tests

@pytest.mark.asyncio
async def test_medium_response_agent_execute_response(
    medium_response_agent,
    sample_cause_detection_result
):
    """Test medium response execution"""
    result = await medium_response_agent.execute_response(sample_cause_detection_result)
    
    assert result["severity"] == "medium"
    assert "cause" in result
    assert "actions_executed" in result
    assert "msds_actions" in result
    assert "alerts_sent" in result
    assert "incident_log" in result


@pytest.mark.asyncio
async def test_medium_response_agent_msds_integration(
    medium_response_agent,
    sample_cause_detection_result
):
    """Test MSDS integration in medium response"""
    # Set cause to trigger MSDS lookup
    sample_cause_detection_result.cause_analysis.primary_cause = "gas_leak_chlorine"
    
    result = await medium_response_agent.execute_response(sample_cause_detection_result)
    
    assert "msds_actions" in result
    assert medium_response_agent.stats["msds_integrations"] > 0


@pytest.mark.asyncio
async def test_medium_response_agent_alerts(
    medium_response_agent,
    sample_cause_detection_result
):
    """Test alert triggering in medium response"""
    result = await medium_response_agent.execute_response(sample_cause_detection_result)
    
    assert result["alerts_sent"] > 0
    assert medium_response_agent.stats["alerts_sent"] > 0


def test_medium_response_agent_get_stats(medium_response_agent):
    """Test statistics retrieval"""
    medium_response_agent.stats["total_incidents"] = 10
    medium_response_agent.stats["successful_responses"] = 8
    medium_response_agent.stats["msds_integrations"] = 5
    
    stats = medium_response_agent.get_stats()
    
    assert stats["total_incidents"] == 10
    assert stats["successful_responses"] == 8
    assert stats["msds_integrations"] == 5
    assert stats["success_rate"] == 0.8


# HighResponseAgent Tests

@pytest.mark.asyncio
async def test_high_response_agent_execute_response(
    high_response_agent,
    sample_cause_detection_result
):
    """Test high response execution"""
    # Set severity to high
    sample_cause_detection_result.severity = "high"
    
    result = await high_response_agent.execute_response(sample_cause_detection_result)
    
    assert result["severity"] == "high"
    assert result["alarm_triggered"] is True
    assert "actions_executed" in result
    assert "msds_actions" in result
    assert "sop_actions" in result
    assert "authorities_notified" in result


@pytest.mark.asyncio
async def test_high_response_agent_emergency_alarm(
    high_response_agent,
    sample_cause_detection_result
):
    """Test emergency alarm triggering"""
    sample_cause_detection_result.severity = "high"
    
    result = await high_response_agent.execute_response(sample_cause_detection_result)
    
    assert result["alarm_triggered"] is True
    assert high_response_agent.stats["alarms_triggered"] > 0


@pytest.mark.asyncio
async def test_high_response_agent_authorities_notification(
    high_response_agent,
    sample_cause_detection_result
):
    """Test authorities notification"""
    sample_cause_detection_result.severity = "high"
    
    result = await high_response_agent.execute_response(sample_cause_detection_result)
    
    assert result["authorities_notified"] > 0
    assert high_response_agent.stats["authorities_notified"] > 0


@pytest.mark.asyncio
async def test_high_response_agent_sop_execution(
    high_response_agent,
    sample_cause_detection_result
):
    """Test SOP execution in high response"""
    sample_cause_detection_result.severity = "high"
    
    result = await high_response_agent.execute_response(sample_cause_detection_result)
    
    assert "sop_actions" in result
    assert high_response_agent.stats["sop_executions"] > 0


def test_high_response_agent_get_stats(high_response_agent):
    """Test statistics retrieval"""
    high_response_agent.stats["total_incidents"] = 5
    high_response_agent.stats["successful_responses"] = 5
    high_response_agent.stats["alarms_triggered"] = 5
    high_response_agent.stats["authorities_notified"] = 20
    
    stats = high_response_agent.get_stats()
    
    assert stats["total_incidents"] == 5
    assert stats["successful_responses"] == 5
    assert stats["alarms_triggered"] == 5
    assert stats["authorities_notified"] == 20
    assert stats["success_rate"] == 1.0


# Integration Tests

@pytest.mark.asyncio
async def test_severity_based_routing(
    mild_response_agent,
    medium_response_agent,
    high_response_agent,
    sample_cause_detection_result
):
    """Test that different severities route to correct agents"""
    # Test mild
    sample_cause_detection_result.severity = "mild"
    mild_result = await mild_response_agent.execute_response(sample_cause_detection_result)
    assert mild_result["severity"] == "mild"
    
    # Test medium
    sample_cause_detection_result.severity = "medium"
    medium_result = await medium_response_agent.execute_response(sample_cause_detection_result)
    assert medium_result["severity"] == "medium"
    
    # Test high
    sample_cause_detection_result.severity = "high"
    high_result = await high_response_agent.execute_response(sample_cause_detection_result)
    assert high_result["severity"] == "high"
    assert high_result["alarm_triggered"] is True


@pytest.mark.asyncio
async def test_response_agent_error_handling(
    mild_response_agent,
    sample_cause_detection_result
):
    """Test error handling in response agents"""
    # Test with invalid data
    with pytest.raises(ValueError):
        await mild_response_agent.process({"invalid": "data"})


def test_response_agent_stats_reset(mild_response_agent):
    """Test statistics reset"""
    mild_response_agent.stats["total_incidents"] = 10
    mild_response_agent.reset_stats()
    
    assert mild_response_agent.stats["total_incidents"] == 0
    assert mild_response_agent.stats["successful_responses"] == 0
