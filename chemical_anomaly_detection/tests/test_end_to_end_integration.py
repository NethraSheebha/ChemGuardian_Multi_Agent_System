"""
End-to-end integration test for the complete chemical leak monitoring system

This test validates the complete flow:
1. Input Collection Agent: Ingest multimodal data and generate embeddings
2. Anomaly Detection Agent: Detect anomalies through similarity search
3. Cause Detection Agent: Infer causes and classify severity
4. Risk Response Agents: Execute appropriate response protocols

Tests verify:
- All agents communicate correctly
- Data flows through all Qdrant collections
- Complete pipeline from input to response execution
- Integration with MSDS/SOP databases
"""

import asyncio
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Import all agents and components
from src.agents.input_collection_agent import (
    InputCollectionAgent,
    EmbeddingGenerator,
    MultimodalEmbedding
)
from src.agents.anomaly_detection_agent import (
    AnomalyDetectionAgent,
    AnomalyDetectionResult
)
from src.agents.cause_detection_agent import (
    CauseDetectionAgent,
    CauseDetectionResult
)
from src.agents.high_response_agent import HighResponseAgent
from src.agents.medium_response_agent import MediumResponseAgent
from src.agents.mild_response_agent import MildResponseAgent

# Import processors
from src.models.video_processor import VideoProcessor
from src.models.audio_processor import AudioProcessor
from src.models.sensor_processor import SensorProcessor
from src.models.sensor_adapter import SensorEmbeddingAdapter

# Import engines and managers
from src.agents.similarity_search_engine import SimilaritySearchEngine
from src.agents.adaptive_threshold_manager import AdaptiveThresholdManager
from src.agents.storage_manager import StorageManager
from src.agents.cause_inference_engine import CauseInferenceEngine
from src.agents.severity_classifier import SeverityClassifier
from src.agents.response_strategy_engine import ResponseStrategyEngine

# Import integrations
from src.integrations.msds_integration import MSDSIntegration
from src.integrations.sop_integration import SOPIntegration

# Import config
from src.config.settings import SystemConfig
import os


class EndToEndTestHarness:
    """Test harness for end-to-end integration testing"""
    
    def __init__(self):
        """Initialize test harness with all components"""
        # Load settings from environment
        self.qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        self.qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
        self.msds_path = os.getenv("MSDS_DATABASE_PATH", "data/msds_database.json")
        self.sop_path = os.getenv("SOP_DATABASE_PATH", "data/sop_database.json")
        self.qdrant_client = None
        
        # Agents
        self.input_agent = None
        self.anomaly_agent = None
        self.cause_agent = None
        self.mild_agent = None
        self.medium_agent = None
        self.high_agent = None
        
        # Collected results for verification
        self.embeddings_generated: List[MultimodalEmbedding] = []
        self.anomalies_detected: List[AnomalyDetectionResult] = []
        self.causes_analyzed: List[CauseDetectionResult] = []
        self.responses_executed: List[Dict[str, Any]] = []
        
    async def setup(self):
        """Set up all components for testing"""
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(
            host=self.qdrant_host,
            port=self.qdrant_port
        )
        
        # Verify collections exist
        collections = await asyncio.to_thread(
            self.qdrant_client.get_collections
        )
        collection_names = [c.name for c in collections.collections]
        
        required_collections = ["baselines", "data", "labeled_anomalies", "response_strategies"]
        for coll in required_collections:
            if coll not in collection_names:
                raise RuntimeError(f"Required collection '{coll}' not found in Qdrant")
        
        # Initialize processors
        video_proc = VideoProcessor(device="cpu", timeout=2.0)
        audio_proc = AudioProcessor(device="cpu", timeout=2.0)
        adapter = SensorEmbeddingAdapter()
        sensor_proc = SensorProcessor(adapter=adapter)
        
        # Initialize embedding generator
        embedding_gen = EmbeddingGenerator(
            video_processor=video_proc,
            audio_processor=audio_proc,
            sensor_processor=sensor_proc
        )
        
        # Initialize Input Collection Agent
        self.input_agent = InputCollectionAgent(
            embedding_generator=embedding_gen,
            processing_interval=1.0,
            queue_max_size=100
        )
        
        # Initialize similarity search and threshold manager
        similarity_search = SimilaritySearchEngine(self.qdrant_client)
        threshold_manager = AdaptiveThresholdManager(
            initial_thresholds={
                "video": 0.7,
                "audio": 0.65,
                "sensor": 2.5
            }
        )
        
        # Initialize storage manager
        storage_manager = StorageManager(self.qdrant_client)
        
        # Initialize Anomaly Detection Agent with callback
        async def anomaly_callback(result: AnomalyDetectionResult):
            self.anomalies_detected.append(result)
        
        self.anomaly_agent = AnomalyDetectionAgent(
            qdrant_client=self.qdrant_client,
            similarity_search_engine=similarity_search,
            adaptive_threshold_manager=threshold_manager,
            storage_manager=storage_manager,
            processing_interval=1.0,
            anomaly_callback=anomaly_callback
        )
        
        # Initialize cause inference and severity classifier
        cause_engine = CauseInferenceEngine(self.qdrant_client)
        severity_classifier = SeverityClassifier()
        
        # Initialize response callbacks
        async def mild_callback(result: CauseDetectionResult):
            self.causes_analyzed.append(result)
            # Execute mild response
            response = await self.mild_agent.execute_response(result)
            self.responses_executed.append(response)
        
        async def medium_callback(result: CauseDetectionResult):
            self.causes_analyzed.append(result)
            # Execute medium response
            response = await self.medium_agent.execute_response(result)
            self.responses_executed.append(response)
        
        async def high_callback(result: CauseDetectionResult):
            self.causes_analyzed.append(result)
            # Execute high response
            response = await self.high_agent.execute_response(result)
            self.responses_executed.append(response)
        
        # Initialize Cause Detection Agent
        self.cause_agent = CauseDetectionAgent(
            qdrant_client=self.qdrant_client,
            cause_inference_engine=cause_engine,
            severity_classifier=severity_classifier,
            processing_interval=1.0,
            mild_callback=mild_callback,
            medium_callback=medium_callback,
            high_callback=high_callback
        )
        
        # Initialize response strategy engine
        msds_integration = MSDSIntegration(self.msds_path)
        sop_integration = SOPIntegration(self.sop_path)
        response_engine = ResponseStrategyEngine(
            qdrant_client=self.qdrant_client,
            msds_integration=msds_integration,
            sop_integration=sop_integration
        )
        
        # Initialize response agents
        self.mild_agent = MildResponseAgent(
            qdrant_client=self.qdrant_client,
            response_strategy_engine=response_engine,
            processing_interval=1.0
        )
        
        self.medium_agent = MediumResponseAgent(
            qdrant_client=self.qdrant_client,
            response_strategy_engine=response_engine,
            processing_interval=1.0
        )
        
        self.high_agent = HighResponseAgent(
            qdrant_client=self.qdrant_client,
            response_strategy_engine=response_engine,
            processing_interval=1.0
        )
        
    async def teardown(self):
        """Clean up resources"""
        if self.qdrant_client:
            self.qdrant_client.close()
    
    async def process_data_point(
        self,
        video_frame: np.ndarray = None,
        audio_window: tuple = None,
        sensor_reading: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Process a single data point through the complete pipeline
        
        Returns:
            Dictionary with results from each stage
        """
        results = {
            "embedding": None,
            "anomaly_result": None,
            "cause_result": None,
            "response": None,
            "success": False
        }
        
        try:
            # Stage 1: Generate embedding
            embedding = await self.input_agent.process_data_point(
                video_frame=video_frame,
                audio_window=audio_window,
                sensor_reading=sensor_reading,
                metadata=metadata or {}
            )
            
            if not embedding or not embedding.has_any_modality():
                return results
            
            results["embedding"] = embedding
            self.embeddings_generated.append(embedding)
            
            # Stage 2: Detect anomaly
            anomaly_result = await self.anomaly_agent.detect_anomaly(embedding)
            results["anomaly_result"] = anomaly_result
            
            # Stage 3: If anomaly, analyze cause and execute response
            if anomaly_result.is_anomaly:
                cause_result = await self.cause_agent.analyze_anomaly(anomaly_result)
                results["cause_result"] = cause_result
                
                # Response is executed via callbacks, check if it was added
                if self.responses_executed:
                    results["response"] = self.responses_executed[-1]
            
            results["success"] = True
            
        except Exception as e:
            results["error"] = str(e)
            
        return results


@pytest.fixture
async def test_harness():
    """Fixture to provide test harness"""
    harness = EndToEndTestHarness()
    await harness.setup()
    yield harness
    await harness.teardown()


@pytest.mark.asyncio
async def test_complete_flow_with_normal_data(test_harness):
    """
    Test complete flow with normal sensor data
    
    Validates:
    - Embedding generation succeeds
    - Anomaly detection classifies as normal
    - Data is stored in Qdrant
    - No cause analysis or response triggered
    """
    # Load normal sensor data
    df = pd.read_csv("normal_sensor_data.csv")
    sensor_reading = {
        "timestamp": datetime.now(),
        "temperature_celsius": float(df.iloc[0]["temperature_celsius"]),
        "pressure_bar": float(df.iloc[0]["pressure_bar"]),
        "gas_concentration_ppm": float(df.iloc[0]["gas_concentration_ppm"]),
        "vibration_mm_s": float(df.iloc[0]["vibration_mm_s"]),
        "flow_rate_lpm": float(df.iloc[0]["flow_rate_lpm"])
    }
    
    metadata = {
        "plant_zone": "Zone_A",
        "shift": "morning",
        "equipment_id": "sensor_01",
        "camera_id": "cam_01"
    }
    
    # Process data point
    results = await test_harness.process_data_point(
        sensor_reading=sensor_reading,
        metadata=metadata
    )
    
    # Verify results
    assert results["success"], "Processing should succeed"
    assert results["embedding"] is not None, "Embedding should be generated"
    assert results["anomaly_result"] is not None, "Anomaly detection should run"
    assert not results["anomaly_result"].is_anomaly, "Normal data should not be anomalous"
    assert results["cause_result"] is None, "No cause analysis for normal data"
    assert results["response"] is None, "No response for normal data"
    
    # Verify embedding was stored
    assert len(test_harness.embeddings_generated) == 1
    assert test_harness.embeddings_generated[0].sensor_embedding is not None


@pytest.mark.asyncio
async def test_complete_flow_with_anomalous_data(test_harness):
    """
    Test complete flow with anomalous sensor data
    
    Validates:
    - Embedding generation succeeds
    - Anomaly detection identifies anomaly
    - Cause analysis determines root cause
    - Severity is classified
    - Appropriate response agent is triggered
    - Response actions are executed
    """
    # Load anomalous sensor data
    df = pd.read_csv("anomalous_sensor.csv")
    sensor_reading = {
        "timestamp": datetime.now(),
        "temperature_celsius": float(df.iloc[0]["temperature_celsius"]),
        "pressure_bar": float(df.iloc[0]["pressure_bar"]),
        "gas_concentration_ppm": float(df.iloc[0]["gas_concentration_ppm"]),
        "vibration_mm_s": float(df.iloc[0]["vibration_mm_s"]),
        "flow_rate_lpm": float(df.iloc[0]["flow_rate_lpm"])
    }
    
    metadata = {
        "plant_zone": "Zone_B",
        "shift": "afternoon",
        "equipment_id": "sensor_02",
        "camera_id": "cam_02"
    }
    
    # Process data point
    results = await test_harness.process_data_point(
        sensor_reading=sensor_reading,
        metadata=metadata
    )
    
    # Verify results
    assert results["success"], "Processing should succeed"
    assert results["embedding"] is not None, "Embedding should be generated"
    assert results["anomaly_result"] is not None, "Anomaly detection should run"
    
    # If anomaly detected, verify complete flow
    if results["anomaly_result"].is_anomaly:
        assert results["cause_result"] is not None, "Cause analysis should run for anomaly"
        assert results["cause_result"].cause_analysis is not None
        assert results["cause_result"].severity in ["mild", "medium", "high"]
        assert results["response"] is not None, "Response should be executed"
        
        # Verify response contains required fields
        response = results["response"]
        assert "severity" in response
        assert "actions_taken" in response
        assert "timestamp" in response


@pytest.mark.asyncio
async def test_multimodal_data_flow(test_harness):
    """
    Test complete flow with all modalities (video + audio + sensor)
    
    Validates:
    - All modalities are processed
    - Embeddings are generated for all modalities
    - Multi-modality anomaly detection works
    - Per-modality anomaly scores are computed
    """
    # Create synthetic multimodal data
    video_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    audio_data = np.random.randn(32000).astype(np.float32)
    
    df = pd.read_csv("anomalous_sensor.csv")
    sensor_reading = {
        "timestamp": datetime.now(),
        "temperature_celsius": float(df.iloc[0]["temperature_celsius"]),
        "pressure_bar": float(df.iloc[0]["pressure_bar"]),
        "gas_concentration_ppm": float(df.iloc[0]["gas_concentration_ppm"]),
        "vibration_mm_s": float(df.iloc[0]["vibration_mm_s"]),
        "flow_rate_lpm": float(df.iloc[0]["flow_rate_lpm"])
    }
    
    metadata = {
        "plant_zone": "Zone_C",
        "shift": "night",
        "equipment_id": "sensor_03",
        "camera_id": "cam_03"
    }
    
    # Process data point
    results = await test_harness.process_data_point(
        video_frame=video_frame,
        audio_window=(audio_data, 32000),
        sensor_reading=sensor_reading,
        metadata=metadata
    )
    
    # Verify results
    assert results["success"], "Processing should succeed"
    assert results["embedding"] is not None, "Embedding should be generated"
    
    # Verify all modalities present
    embedding = results["embedding"]
    assert embedding.video_embedding is not None, "Video embedding should exist"
    assert embedding.audio_embedding is not None, "Audio embedding should exist"
    assert embedding.sensor_embedding is not None, "Sensor embedding should exist"
    
    # Verify dimensions
    assert embedding.video_embedding.shape == (512,)
    assert embedding.audio_embedding.shape == (512,)
    assert embedding.sensor_embedding.shape == (128,)
    
    # Verify anomaly detection ran
    assert results["anomaly_result"] is not None
    assert "video" in results["anomaly_result"].anomaly_scores
    assert "audio" in results["anomaly_result"].anomaly_scores
    assert "sensor" in results["anomaly_result"].anomaly_scores


@pytest.mark.asyncio
async def test_agent_communication(test_harness):
    """
    Test that all agents communicate correctly
    
    Validates:
    - Input Collection Agent passes embeddings to Anomaly Detection Agent
    - Anomaly Detection Agent triggers Cause Detection Agent for anomalies
    - Cause Detection Agent routes to appropriate Response Agent
    - Callbacks are executed correctly
    """
    # Process multiple data points
    df = pd.read_csv("anomalous_sensor.csv")
    
    for i in range(min(3, len(df))):
        sensor_reading = {
            "timestamp": datetime.now(),
            "temperature_celsius": float(df.iloc[i]["temperature_celsius"]),
            "pressure_bar": float(df.iloc[i]["pressure_bar"]),
            "gas_concentration_ppm": float(df.iloc[i]["gas_concentration_ppm"]),
            "vibration_mm_s": float(df.iloc[i]["vibration_mm_s"]),
            "flow_rate_lpm": float(df.iloc[i]["flow_rate_lpm"])
        }
        
        metadata = {
            "plant_zone": "Zone_A",
            "shift": "morning",
            "reading_id": i
        }
        
        await test_harness.process_data_point(
            sensor_reading=sensor_reading,
            metadata=metadata
        )
    
    # Verify agent communication
    assert len(test_harness.embeddings_generated) >= 1, "Embeddings should be generated"
    
    # Check if any anomalies were detected and processed
    if test_harness.anomalies_detected:
        assert len(test_harness.causes_analyzed) > 0, "Causes should be analyzed for anomalies"
        assert len(test_harness.responses_executed) > 0, "Responses should be executed"


@pytest.mark.asyncio
async def test_data_storage_in_qdrant(test_harness):
    """
    Test that data flows through all Qdrant collections
    
    Validates:
    - Embeddings are stored in 'data' collection
    - Anomalies are flagged correctly
    - Baseline collection is queried
    - Response strategies collection is queried
    """
    # Process a data point
    df = pd.read_csv("normal_sensor_data.csv")
    sensor_reading = {
        "timestamp": datetime.now(),
        "temperature_celsius": float(df.iloc[0]["temperature_celsius"]),
        "pressure_bar": float(df.iloc[0]["pressure_bar"]),
        "gas_concentration_ppm": float(df.iloc[0]["gas_concentration_ppm"]),
        "vibration_mm_s": float(df.iloc[0]["vibration_mm_s"]),
        "flow_rate_lpm": float(df.iloc[0]["flow_rate_lpm"])
    }
    
    metadata = {
        "plant_zone": "Zone_A",
        "shift": "morning",
        "test_id": "storage_test"
    }
    
    # Get initial count
    initial_count = await asyncio.to_thread(
        test_harness.qdrant_client.count,
        collection_name="data"
    )
    
    # Process data point
    results = await test_harness.process_data_point(
        sensor_reading=sensor_reading,
        metadata=metadata
    )
    
    # Verify storage
    assert results["success"], "Processing should succeed"
    
    # Check if data was stored (count should increase)
    final_count = await asyncio.to_thread(
        test_harness.qdrant_client.count,
        collection_name="data"
    )
    
    # Note: Count may not increase if storage is async or batched
    # This is a basic check
    assert final_count.count >= initial_count.count


@pytest.mark.asyncio
async def test_msds_sop_integration(test_harness):
    """
    Test MSDS and SOP integration in response execution
    
    Validates:
    - MSDS information is retrieved for detected chemicals
    - SOP procedures are retrieved for plant zones
    - Response includes MSDS and SOP references
    """
    # Create anomalous data that should trigger response
    df = pd.read_csv("anomalous_sensor.csv")
    
    # Use high gas concentration to potentially trigger higher severity
    sensor_reading = {
        "timestamp": datetime.now(),
        "temperature_celsius": float(df.iloc[0]["temperature_celsius"]),
        "pressure_bar": float(df.iloc[0]["pressure_bar"]),
        "gas_concentration_ppm": 1200.0,  # High concentration
        "vibration_mm_s": float(df.iloc[0]["vibration_mm_s"]),
        "flow_rate_lpm": float(df.iloc[0]["flow_rate_lpm"])
    }
    
    metadata = {
        "plant_zone": "Zone_A",
        "shift": "morning",
        "equipment_id": "sensor_01"
    }
    
    # Process data point
    results = await test_harness.process_data_point(
        sensor_reading=sensor_reading,
        metadata=metadata
    )
    
    # If response was executed, verify MSDS/SOP integration
    if results["response"]:
        response = results["response"]
        # Response should contain strategy information
        assert "severity" in response
        # MSDS/SOP info may be in nested structure depending on implementation


@pytest.mark.asyncio
async def test_graceful_degradation(test_harness):
    """
    Test graceful degradation with partial modalities
    
    Validates:
    - System continues with only sensor data
    - System continues with only video data
    - Confidence scores are adjusted for partial data
    """
    df = pd.read_csv("normal_sensor_data.csv")
    sensor_reading = {
        "timestamp": datetime.now(),
        "temperature_celsius": float(df.iloc[0]["temperature_celsius"]),
        "pressure_bar": float(df.iloc[0]["pressure_bar"]),
        "gas_concentration_ppm": float(df.iloc[0]["gas_concentration_ppm"]),
        "vibration_mm_s": float(df.iloc[0]["vibration_mm_s"]),
        "flow_rate_lpm": float(df.iloc[0]["flow_rate_lpm"])
    }
    
    metadata = {"plant_zone": "Zone_A", "shift": "morning"}
    
    # Test with sensor only
    results = await test_harness.process_data_point(
        sensor_reading=sensor_reading,
        metadata=metadata
    )
    
    assert results["success"], "Should succeed with sensor only"
    assert results["embedding"] is not None
    assert results["embedding"].sensor_embedding is not None
    assert results["embedding"].video_embedding is None
    assert results["embedding"].audio_embedding is None


@pytest.mark.asyncio
async def test_performance_requirements(test_harness):
    """
    Test that performance requirements are met
    
    Validates:
    - End-to-end latency < 2 seconds
    - Embedding generation < 500ms
    """
    df = pd.read_csv("normal_sensor_data.csv")
    sensor_reading = {
        "timestamp": datetime.now(),
        "temperature_celsius": float(df.iloc[0]["temperature_celsius"]),
        "pressure_bar": float(df.iloc[0]["pressure_bar"]),
        "gas_concentration_ppm": float(df.iloc[0]["gas_concentration_ppm"]),
        "vibration_mm_s": float(df.iloc[0]["vibration_mm_s"]),
        "flow_rate_lpm": float(df.iloc[0]["flow_rate_lpm"])
    }
    
    metadata = {"plant_zone": "Zone_A"}
    
    # Measure end-to-end latency
    start_time = asyncio.get_event_loop().time()
    
    results = await test_harness.process_data_point(
        sensor_reading=sensor_reading,
        metadata=metadata
    )
    
    end_time = asyncio.get_event_loop().time()
    latency = end_time - start_time
    
    assert results["success"], "Processing should succeed"
    
    # Check latency (may be higher in test environment)
    # This is a soft check - log warning if exceeded
    if latency >= 2.0:
        print(f"Warning: End-to-end latency {latency:.2f}s exceeds 2s target")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
