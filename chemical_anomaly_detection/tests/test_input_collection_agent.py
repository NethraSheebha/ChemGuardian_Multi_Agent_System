"""Tests for Input Collection Agent"""

import asyncio
import pytest
import numpy as np
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

from src.agents.input_collection_agent import (
    EmbeddingGenerator,
    InputCollectionAgent,
    MultimodalEmbedding,
    ModalityStatus
)
from src.models.video_processor import VideoProcessor
from src.models.audio_processor import AudioProcessor
from src.models.sensor_processor import SensorProcessor


class TestMultimodalEmbedding:
    """Test MultimodalEmbedding dataclass"""
    
    def test_embedding_initialization(self):
        """Test basic initialization"""
        timestamp = datetime.utcnow().isoformat()
        video_emb = np.random.randn(512)
        audio_emb = np.random.randn(512)
        sensor_emb = np.random.randn(128)
        
        embedding = MultimodalEmbedding(
            timestamp=timestamp,
            video_embedding=video_emb,
            audio_embedding=audio_emb,
            sensor_embedding=sensor_emb,
            metadata={"location": "zone_a"}
        )
        
        assert embedding.timestamp == timestamp
        assert np.array_equal(embedding.video_embedding, video_emb)
        assert np.array_equal(embedding.audio_embedding, audio_emb)
        assert np.array_equal(embedding.sensor_embedding, sensor_emb)
        assert embedding.metadata["location"] == "zone_a"
    
    def test_modality_status_auto_set(self):
        """Test automatic modality status setting"""
        embedding = MultimodalEmbedding(
            timestamp=datetime.utcnow().isoformat(),
            video_embedding=np.random.randn(512),
            audio_embedding=None,
            sensor_embedding=np.random.randn(128)
        )
        
        assert embedding.modality_status["video"] == ModalityStatus.AVAILABLE
        assert embedding.modality_status["audio"] == ModalityStatus.MISSING
        assert embedding.modality_status["sensor"] == ModalityStatus.AVAILABLE
    
    def test_get_available_modalities(self):
        """Test getting list of available modalities"""
        embedding = MultimodalEmbedding(
            timestamp=datetime.utcnow().isoformat(),
            video_embedding=np.random.randn(512),
            audio_embedding=None,
            sensor_embedding=np.random.randn(128)
        )
        
        available = embedding.get_available_modalities()
        assert "video" in available
        assert "sensor" in available
        assert "audio" not in available
    
    def test_has_any_modality_true(self):
        """Test has_any_modality returns True when modalities exist"""
        embedding = MultimodalEmbedding(
            timestamp=datetime.utcnow().isoformat(),
            video_embedding=np.random.randn(512)
        )
        
        assert embedding.has_any_modality() is True
    
    def test_has_any_modality_false(self):
        """Test has_any_modality returns False when no modalities"""
        embedding = MultimodalEmbedding(
            timestamp=datetime.utcnow().isoformat()
        )
        
        assert embedding.has_any_modality() is False
    
    def test_to_dict(self):
        """Test conversion to dictionary"""
        timestamp = datetime.utcnow().isoformat()
        video_emb = np.array([1.0, 2.0, 3.0])
        
        embedding = MultimodalEmbedding(
            timestamp=timestamp,
            video_embedding=video_emb,
            metadata={"location": "zone_a"}
        )
        
        result = embedding.to_dict()
        
        assert result["timestamp"] == timestamp
        assert result["video_embedding"] == [1.0, 2.0, 3.0]
        assert result["audio_embedding"] is None
        assert result["metadata"]["location"] == "zone_a"
        assert "video" in result["modality_status"]


class TestEmbeddingGenerator:
    """Test EmbeddingGenerator class"""
    
    @pytest.fixture
    def mock_processors(self):
        """Create mock processors"""
        video_proc = Mock(spec=VideoProcessor)
        audio_proc = Mock(spec=AudioProcessor)
        sensor_proc = Mock(spec=SensorProcessor)
        
        # Setup async methods
        video_proc.process_frame = AsyncMock(return_value=np.random.randn(512))
        audio_proc.process_audio = AsyncMock(return_value=np.random.randn(512))
        sensor_proc.process = AsyncMock(return_value=np.random.randn(128))
        
        return video_proc, audio_proc, sensor_proc
    
    def test_generator_initialization(self, mock_processors):
        """Test generator initialization"""
        video_proc, audio_proc, sensor_proc = mock_processors
        
        generator = EmbeddingGenerator(
            video_processor=video_proc,
            audio_processor=audio_proc,
            sensor_processor=sensor_proc
        )
        
        assert generator.video_processor == video_proc
        assert generator.audio_processor == audio_proc
        assert generator.sensor_processor == sensor_proc
        assert generator.available_processors["video"] is True
        assert generator.available_processors["audio"] is True
        assert generator.available_processors["sensor"] is True
    
    def test_generator_partial_processors(self):
        """Test generator with only some processors"""
        video_proc = Mock(spec=VideoProcessor)
        
        generator = EmbeddingGenerator(video_processor=video_proc)
        
        assert generator.available_processors["video"] is True
        assert generator.available_processors["audio"] is False
        assert generator.available_processors["sensor"] is False
    
    @pytest.mark.asyncio
    async def test_generate_all_modalities(self, mock_processors):
        """Test generating embeddings with all modalities"""
        video_proc, audio_proc, sensor_proc = mock_processors
        
        generator = EmbeddingGenerator(
            video_processor=video_proc,
            audio_processor=audio_proc,
            sensor_processor=sensor_proc
        )
        
        video_frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        audio_data = np.random.randn(32000)
        sensor_reading = {
            "temperature": 25.0,
            "pressure": 101.3,
            "gas_concentration": 0.1,
            "vibration": 0.5,
            "flow_rate": 10.0
        }
        metadata = {"location": "zone_a", "camera_id": "cam_01"}
        
        embedding = await generator.generate(
            video_frame=video_frame,
            audio_window=(audio_data, 32000),
            sensor_reading=sensor_reading,
            metadata=metadata
        )
        
        assert embedding.video_embedding is not None
        assert embedding.audio_embedding is not None
        assert embedding.sensor_embedding is not None
        assert embedding.metadata["location"] == "zone_a"
        assert len(embedding.get_available_modalities()) == 3
        
        # Verify processors were called
        video_proc.process_frame.assert_called_once()
        audio_proc.process_audio.assert_called_once()
        sensor_proc.process.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_partial_modalities(self, mock_processors):
        """Test generating embeddings with only some modalities"""
        video_proc, audio_proc, sensor_proc = mock_processors
        
        generator = EmbeddingGenerator(
            video_processor=video_proc,
            audio_processor=audio_proc,
            sensor_processor=sensor_proc
        )
        
        video_frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        embedding = await generator.generate(
            video_frame=video_frame,
            audio_window=None,
            sensor_reading=None
        )
        
        assert embedding.video_embedding is not None
        assert embedding.audio_embedding is None
        assert embedding.sensor_embedding is None
        assert len(embedding.get_available_modalities()) == 1
    
    @pytest.mark.asyncio
    async def test_generate_with_processor_failure(self, mock_processors):
        """Test handling processor failures"""
        video_proc, audio_proc, sensor_proc = mock_processors
        
        # Make video processor fail
        video_proc.process_frame = AsyncMock(side_effect=Exception("Processing failed"))
        
        generator = EmbeddingGenerator(
            video_processor=video_proc,
            audio_processor=audio_proc,
            sensor_processor=sensor_proc
        )
        
        video_frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        audio_data = np.random.randn(32000)
        
        embedding = await generator.generate(
            video_frame=video_frame,
            audio_window=(audio_data, 32000),
            sensor_reading=None
        )
        
        assert embedding.video_embedding is None
        assert embedding.audio_embedding is not None
        assert embedding.modality_status["video"] == ModalityStatus.FAILED
        assert embedding.modality_status["audio"] == ModalityStatus.AVAILABLE
    
    @pytest.mark.asyncio
    async def test_generate_parallel_processing(self, mock_processors):
        """Test that modalities are processed in parallel"""
        video_proc, audio_proc, sensor_proc = mock_processors
        
        # Add delays to simulate processing time
        async def delayed_video(*args, **kwargs):
            await asyncio.sleep(0.1)
            return np.random.randn(512)
        
        async def delayed_audio(*args, **kwargs):
            await asyncio.sleep(0.1)
            return np.random.randn(512)
        
        async def delayed_sensor(*args, **kwargs):
            await asyncio.sleep(0.1)
            return np.random.randn(128)
        
        video_proc.process_frame = delayed_video
        audio_proc.process_audio = delayed_audio
        sensor_proc.process_reading = delayed_sensor
        
        generator = EmbeddingGenerator(
            video_processor=video_proc,
            audio_processor=audio_proc,
            sensor_processor=sensor_proc
        )
        
        video_frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        audio_data = np.random.randn(32000)
        sensor_reading = {"temperature": 25.0, "pressure": 101.3, "gas_concentration": 0.1, "vibration": 0.5, "flow_rate": 10.0}
        
        start_time = asyncio.get_event_loop().time()
        embedding = await generator.generate(
            video_frame=video_frame,
            audio_window=(audio_data, 32000),
            sensor_reading=sensor_reading
        )
        elapsed = asyncio.get_event_loop().time() - start_time
        
        # If parallel, should take ~0.1s, if sequential would take ~0.3s
        assert elapsed < 0.2, "Processing should be parallel"
        assert embedding.has_any_modality()


class TestInputCollectionAgent:
    """Test InputCollectionAgent class"""
    
    @pytest.fixture
    def mock_generator(self):
        """Create mock embedding generator"""
        generator = Mock(spec=EmbeddingGenerator)
        
        async def mock_generate(*args, **kwargs):
            return MultimodalEmbedding(
                timestamp=datetime.utcnow().isoformat(),
                video_embedding=np.random.randn(512),
                audio_embedding=np.random.randn(512),
                sensor_embedding=np.random.randn(128)
            )
        
        generator.generate = mock_generate
        return generator
    
    def test_agent_initialization(self, mock_generator):
        """Test agent initialization"""
        agent = InputCollectionAgent(
            embedding_generator=mock_generator,
            processing_interval=1.0,
            queue_max_size=100
        )
        
        assert agent.embedding_generator == mock_generator
        assert agent.processing_interval == 1.0
        assert agent.queue_max_size == 100
        assert agent.get_queue_size() == 0
    
    @pytest.mark.asyncio
    async def test_process_data_point_success(self, mock_generator):
        """Test successful data point processing"""
        agent = InputCollectionAgent(embedding_generator=mock_generator)
        
        video_frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        audio_data = np.random.randn(32000)
        sensor_reading = {"temperature": 25.0, "pressure": 101.3, "gas_concentration": 0.1, "vibration": 0.5, "flow_rate": 10.0}
        
        embedding = await agent.process_data_point(
            video_frame=video_frame,
            audio_window=(audio_data, 32000),
            sensor_reading=sensor_reading,
            metadata={"location": "zone_a"}
        )
        
        assert embedding is not None
        assert embedding.has_any_modality()
        assert agent.stats["successful_embeddings"] == 1
        assert agent.stats["total_processed"] == 1
    
    @pytest.mark.asyncio
    async def test_process_data_point_no_modalities(self):
        """Test processing with no available modalities"""
        generator = Mock(spec=EmbeddingGenerator)
        
        async def mock_generate(*args, **kwargs):
            return MultimodalEmbedding(
                timestamp=datetime.utcnow().isoformat()
            )
        
        generator.generate = mock_generate
        
        agent = InputCollectionAgent(embedding_generator=generator)
        
        embedding = await agent.process_data_point()
        
        assert embedding is None
        assert agent.stats["failed_embeddings"] == 1
    
    @pytest.mark.asyncio
    async def test_process_data_point_exception(self):
        """Test handling exceptions during processing"""
        generator = Mock(spec=EmbeddingGenerator)
        generator.generate = AsyncMock(side_effect=Exception("Processing error"))
        
        agent = InputCollectionAgent(embedding_generator=generator)
        
        embedding = await agent.process_data_point(
            video_frame=np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        )
        
        assert embedding is None
        assert agent.stats["failed_embeddings"] == 1
    
    @pytest.mark.asyncio
    async def test_enqueue_embedding(self, mock_generator):
        """Test enqueuing embeddings"""
        agent = InputCollectionAgent(
            embedding_generator=mock_generator,
            queue_max_size=10
        )
        
        embedding = MultimodalEmbedding(
            timestamp=datetime.utcnow().isoformat(),
            video_embedding=np.random.randn(512)
        )
        
        success = await agent.enqueue_embedding(embedding)
        
        assert success is True
        assert agent.get_queue_size() == 1
    
    @pytest.mark.asyncio
    async def test_enqueue_embedding_queue_full(self, mock_generator):
        """Test enqueuing when queue is full"""
        agent = InputCollectionAgent(
            embedding_generator=mock_generator,
            queue_max_size=2
        )
        
        # Fill the queue
        for _ in range(2):
            embedding = MultimodalEmbedding(
                timestamp=datetime.utcnow().isoformat(),
                video_embedding=np.random.randn(512)
            )
            await agent.enqueue_embedding(embedding)
        
        # Try to add one more
        embedding = MultimodalEmbedding(
            timestamp=datetime.utcnow().isoformat(),
            video_embedding=np.random.randn(512)
        )
        success = await agent.enqueue_embedding(embedding)
        
        assert success is False
        assert agent.stats["queue_full_count"] == 1
        assert agent.get_queue_size() == 2
    
    @pytest.mark.asyncio
    async def test_get_next_embedding(self, mock_generator):
        """Test getting embeddings from queue"""
        agent = InputCollectionAgent(embedding_generator=mock_generator)
        
        # Add embedding to queue
        embedding_in = MultimodalEmbedding(
            timestamp=datetime.utcnow().isoformat(),
            video_embedding=np.random.randn(512)
        )
        await agent.enqueue_embedding(embedding_in)
        
        # Get embedding from queue
        embedding_out = await agent.get_next_embedding(timeout=1.0)
        
        assert embedding_out is not None
        assert embedding_out.timestamp == embedding_in.timestamp
        assert agent.get_queue_size() == 0
    
    @pytest.mark.asyncio
    async def test_get_next_embedding_timeout(self, mock_generator):
        """Test timeout when getting from empty queue"""
        agent = InputCollectionAgent(embedding_generator=mock_generator)
        
        embedding = await agent.get_next_embedding(timeout=0.1)
        
        assert embedding is None
    
    def test_get_stats(self, mock_generator):
        """Test getting agent statistics"""
        agent = InputCollectionAgent(
            embedding_generator=mock_generator,
            queue_max_size=100
        )
        
        agent.stats["total_processed"] = 10
        agent.stats["successful_embeddings"] = 8
        agent.stats["failed_embeddings"] = 2
        
        stats = agent.get_stats()
        
        assert stats["total_processed"] == 10
        assert stats["successful_embeddings"] == 8
        assert stats["failed_embeddings"] == 2
        assert "queue_size" in stats
        assert "queue_utilization" in stats
    
    @pytest.mark.asyncio
    async def test_execute(self, mock_generator):
        """Test execute method"""
        agent = InputCollectionAgent(embedding_generator=mock_generator)
        
        result = await agent.execute()
        
        assert result["status"] == "running"
        assert "stats" in result


@pytest.mark.asyncio
async def test_end_to_end_latency():
    """Test end-to-end processing latency meets requirements (<500ms)"""
    from src.models.sensor_adapter import SensorEmbeddingAdapter
    
    # Create real processors (will use mock models)
    video_proc = VideoProcessor(device="cpu", timeout=0.5)
    audio_proc = AudioProcessor(device="cpu", timeout=0.5)
    
    # Create sensor adapter and processor
    adapter = SensorEmbeddingAdapter()
    sensor_proc = SensorProcessor(adapter=adapter)
    
    generator = EmbeddingGenerator(
        video_processor=video_proc,
        audio_processor=audio_proc,
        sensor_processor=sensor_proc
    )
    
    agent = InputCollectionAgent(embedding_generator=generator)
    
    # Prepare test data
    video_frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    audio_data = np.random.randn(32000).astype(np.float32)
    sensor_reading = {
        "temperature": 25.0,
        "pressure": 101.3,
        "gas_concentration": 0.1,
        "vibration": 0.5,
        "flow_rate": 10.0
    }
    
    # Measure latency
    start_time = asyncio.get_event_loop().time()
    embedding = await agent.process_data_point(
        video_frame=video_frame,
        audio_window=(audio_data, 32000),
        sensor_reading=sensor_reading
    )
    elapsed = asyncio.get_event_loop().time() - start_time
    
    assert embedding is not None
    assert elapsed < 0.5, f"Processing took {elapsed*1000:.1f}ms, should be <500ms"
    assert embedding.has_any_modality()
