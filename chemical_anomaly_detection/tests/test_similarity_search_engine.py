"""Tests for SimilaritySearchEngine"""

import pytest
import asyncio
import numpy as np
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch
from qdrant_client.models import ScoredPoint, Record

from src.agents.similarity_search_engine import SimilaritySearchEngine, SearchResult
from src.agents.input_collection_agent import MultimodalEmbedding, ModalityStatus


@pytest.fixture
def mock_qdrant_client():
    """Create mock Qdrant client"""
    client = Mock()
    return client


@pytest.fixture
def search_engine(mock_qdrant_client):
    """Create SimilaritySearchEngine instance"""
    return SimilaritySearchEngine(
        qdrant_client=mock_qdrant_client,
        collection_name="baselines",
        top_k=10,
        search_timeout=1.0
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


@pytest.fixture
def mock_scored_points():
    """Create mock scored points"""
    points = []
    for i in range(5):
        point = Mock(spec=ScoredPoint)
        point.score = 0.1 + i * 0.05  # Distances: 0.1, 0.15, 0.2, 0.25, 0.3
        point.id = i
        point.payload = {
            "baseline_id": f"baseline_{i}",
            "shift": "morning",
            "plant_zone": "Zone_A"
        }
        points.append(point)
    return points


class TestSimilaritySearchEngine:
    """Test suite for SimilaritySearchEngine"""
    
    def test_initialization(self, search_engine):
        """Test search engine initialization"""
        assert search_engine.collection_name == "baselines"
        assert search_engine.top_k == 10
        assert search_engine.search_timeout == 1.0
        assert search_engine.stats["total_searches"] == 0
    
    @pytest.mark.asyncio
    async def test_search_baselines_all_modalities(
        self,
        search_engine,
        sample_embedding,
        mock_scored_points
    ):
        """Test searching baselines with all modalities available"""
        # Mock search to return scored points
        search_engine.qdrant_client.search = Mock(return_value=mock_scored_points)
        
        # Execute search
        results = await search_engine.search_baselines(sample_embedding)
        
        # Verify results
        assert len(results) == 3  # video, audio, sensor
        assert "video" in results
        assert "audio" in results
        assert "sensor" in results
        
        # Verify each result
        for modality, result in results.items():
            assert isinstance(result, SearchResult)
            assert result.modality == modality
            assert len(result.scored_points) == 5
            assert result.min_distance == 0.1
            assert result.mean_distance == pytest.approx(0.2, abs=0.01)
        
        # Verify statistics
        assert search_engine.stats["total_searches"] == 1
        assert search_engine.stats["successful_searches"] == 1
    
    @pytest.mark.asyncio
    async def test_search_baselines_partial_modalities(
        self,
        search_engine,
        mock_scored_points
    ):
        """Test searching with only some modalities available"""
        # Create embedding with only video and sensor
        embedding = MultimodalEmbedding(
            timestamp=datetime.utcnow().isoformat(),
            video_embedding=np.random.randn(512).astype(np.float32),
            audio_embedding=None,  # Missing audio
            sensor_embedding=np.random.randn(128).astype(np.float32),
            metadata={}
        )
        
        # Mock search
        search_engine.qdrant_client.search = Mock(return_value=mock_scored_points)
        
        # Execute search
        results = await search_engine.search_baselines(embedding)
        
        # Verify only video and sensor results
        assert len(results) == 2
        assert "video" in results
        assert "sensor" in results
        assert "audio" not in results
    
    @pytest.mark.asyncio
    async def test_search_baselines_with_filters(
        self,
        search_engine,
        sample_embedding,
        mock_scored_points
    ):
        """Test searching with shift, equipment_id, and plant_zone filters"""
        # Mock search
        search_engine.qdrant_client.search = Mock(return_value=mock_scored_points)
        
        # Execute search with filters
        results = await search_engine.search_baselines(
            embedding=sample_embedding,
            shift="morning",
            equipment_id="EQ001",
            plant_zone="Zone_A"
        )
        
        # Verify search was called with filters
        assert search_engine.qdrant_client.search.called
        call_args = search_engine.qdrant_client.search.call_args
        
        # Check that filter was passed
        assert call_args[1]["query_filter"] is not None
        
        # Verify results
        assert len(results) == 3
    
    @pytest.mark.asyncio
    async def test_search_baselines_no_results(
        self,
        search_engine,
        sample_embedding
    ):
        """Test searching when no baselines are found"""
        # Mock search to return empty list
        search_engine.qdrant_client.search = Mock(return_value=[])
        
        # Execute search
        results = await search_engine.search_baselines(sample_embedding)
        
        # Verify empty results
        assert len(results) == 0
        assert search_engine.stats["successful_searches"] == 1
    
    @pytest.mark.asyncio
    async def test_search_baselines_timeout(
        self,
        search_engine,
        sample_embedding
    ):
        """Test search timeout handling"""
        # Mock search to hang
        async def slow_search(*args, **kwargs):
            await asyncio.sleep(2.0)  # Longer than timeout
            return []
        
        search_engine._search_modality = slow_search
        
        # Execute search (should timeout)
        results = await search_engine.search_baselines(sample_embedding)
        
        # Verify timeout was handled
        assert len(results) == 0
        assert search_engine.stats["timeout_count"] == 1
        assert search_engine.stats["failed_searches"] == 1
    
    @pytest.mark.asyncio
    async def test_search_baselines_error_handling(
        self,
        search_engine,
        sample_embedding
    ):
        """Test error handling during search"""
        # Mock search to raise exception
        search_engine.qdrant_client.search = Mock(
            side_effect=Exception("Connection error")
        )
        
        # Execute search
        results = await search_engine.search_baselines(sample_embedding)
        
        # Verify error was handled gracefully
        # Individual modality failures don't fail the whole search,
        # they just result in no results for those modalities
        assert len(results) == 0
        assert search_engine.stats["successful_searches"] == 1  # Search completed, just with no results
    
    def test_compute_anomaly_scores(self, search_engine, mock_scored_points):
        """Test computing anomaly scores from search results"""
        # Create search results
        search_results = {
            "video": SearchResult(
                modality="video",
                scored_points=mock_scored_points,
                min_distance=0.1,
                mean_distance=0.2
            ),
            "audio": SearchResult(
                modality="audio",
                scored_points=mock_scored_points,
                min_distance=0.15,
                mean_distance=0.25
            ),
            "sensor": SearchResult(
                modality="sensor",
                scored_points=mock_scored_points,
                min_distance=0.3,
                mean_distance=0.4
            )
        }
        
        # Compute anomaly scores
        scores = search_engine.compute_anomaly_scores(search_results)
        
        # Verify scores
        assert len(scores) == 3
        assert scores["video"] == 0.1
        assert scores["audio"] == 0.15
        assert scores["sensor"] == 0.3
    
    def test_compute_anomaly_scores_empty(self, search_engine):
        """Test computing anomaly scores with empty results"""
        scores = search_engine.compute_anomaly_scores({})
        assert len(scores) == 0
    
    @pytest.mark.asyncio
    async def test_search_and_score(
        self,
        search_engine,
        sample_embedding,
        mock_scored_points
    ):
        """Test combined search and score operation"""
        # Mock search
        search_engine.qdrant_client.search = Mock(return_value=mock_scored_points)
        
        # Execute search and score
        search_results, anomaly_scores = await search_engine.search_and_score(
            embedding=sample_embedding,
            shift="morning",
            plant_zone="Zone_A"
        )
        
        # Verify search results
        assert len(search_results) == 3
        assert "video" in search_results
        assert "audio" in search_results
        assert "sensor" in search_results
        
        # Verify anomaly scores
        assert len(anomaly_scores) == 3
        assert all(isinstance(score, float) for score in anomaly_scores.values())
        assert all(score >= 0 for score in anomaly_scores.values())
    
    def test_build_filter_all_criteria(self, search_engine):
        """Test building filter with all criteria"""
        filter_obj = search_engine._build_filter(
            shift="morning",
            equipment_id="EQ001",
            plant_zone="Zone_A"
        )
        
        assert filter_obj is not None
        assert len(filter_obj.must) == 3
    
    def test_build_filter_partial_criteria(self, search_engine):
        """Test building filter with partial criteria"""
        filter_obj = search_engine._build_filter(
            shift="morning",
            equipment_id=None,
            plant_zone="Zone_A"
        )
        
        assert filter_obj is not None
        assert len(filter_obj.must) == 2
    
    def test_build_filter_no_criteria(self, search_engine):
        """Test building filter with no criteria"""
        filter_obj = search_engine._build_filter(
            shift=None,
            equipment_id=None,
            plant_zone=None
        )
        
        assert filter_obj is None
    
    def test_get_stats(self, search_engine):
        """Test getting statistics"""
        # Simulate some searches
        search_engine.stats["total_searches"] = 10
        search_engine.stats["successful_searches"] = 8
        search_engine.stats["failed_searches"] = 2
        search_engine.stats["total_search_time_ms"] = 800.0
        
        # Get stats
        stats = search_engine.get_stats()
        
        # Verify computed metrics
        assert stats["total_searches"] == 10
        assert stats["successful_searches"] == 8
        assert stats["failed_searches"] == 2
        assert stats["avg_search_time_ms"] == 100.0
        assert stats["success_rate"] == 0.8
    
    def test_get_stats_no_searches(self, search_engine):
        """Test getting statistics with no searches"""
        stats = search_engine.get_stats()
        
        assert stats["total_searches"] == 0
        assert stats["avg_search_time_ms"] == 0.0
        assert stats["success_rate"] == 0.0
    
    def test_reset_stats(self, search_engine):
        """Test resetting statistics"""
        # Set some stats
        search_engine.stats["total_searches"] = 10
        search_engine.stats["successful_searches"] = 8
        
        # Reset
        search_engine.reset_stats()
        
        # Verify reset
        assert search_engine.stats["total_searches"] == 0
        assert search_engine.stats["successful_searches"] == 0
    
    def test_search_result_to_dict(self, mock_scored_points):
        """Test SearchResult to_dict method"""
        result = SearchResult(
            modality="video",
            scored_points=mock_scored_points,
            min_distance=0.1,
            mean_distance=0.2
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["modality"] == "video"
        assert result_dict["num_results"] == 5
        assert result_dict["min_distance"] == 0.1
        assert result_dict["mean_distance"] == 0.2
    
    @pytest.mark.asyncio
    async def test_search_performance(
        self,
        search_engine,
        sample_embedding,
        mock_scored_points
    ):
        """Test that search completes within 1 second"""
        # Mock fast search
        search_engine.qdrant_client.search = Mock(return_value=mock_scored_points)
        
        # Measure search time
        start = asyncio.get_event_loop().time()
        results = await search_engine.search_baselines(sample_embedding)
        elapsed = asyncio.get_event_loop().time() - start
        
        # Verify performance
        assert elapsed < 1.0  # Should complete within 1 second
        assert len(results) == 3
    
    @pytest.mark.asyncio
    async def test_parallel_modality_search(
        self,
        search_engine,
        sample_embedding,
        mock_scored_points
    ):
        """Test that modalities are searched in parallel"""
        # Track call order
        call_times = []
        
        def mock_search(*args, **kwargs):
            call_times.append(asyncio.get_event_loop().time())
            return mock_scored_points
        
        search_engine.qdrant_client.search = mock_search
        
        # Execute search
        start = asyncio.get_event_loop().time()
        results = await search_engine.search_baselines(sample_embedding)
        
        # Verify all searches happened (roughly) at the same time
        # (parallel execution means they should all start within a few ms)
        if len(call_times) > 1:
            time_spread = max(call_times) - min(call_times)
            assert time_spread < 0.1  # All started within 100ms
    
    @pytest.mark.asyncio
    async def test_search_with_missing_modality_embedding(self, search_engine):
        """Test search with completely missing embeddings"""
        # Create embedding with no modalities
        embedding = MultimodalEmbedding(
            timestamp=datetime.utcnow().isoformat(),
            video_embedding=None,
            audio_embedding=None,
            sensor_embedding=None,
            metadata={}
        )
        
        # Execute search
        results = await search_engine.search_baselines(embedding)
        
        # Verify empty results
        assert len(results) == 0
    
    @pytest.mark.asyncio
    async def test_search_modality_individual(
        self,
        search_engine,
        mock_scored_points
    ):
        """Test individual modality search"""
        # Mock search
        search_engine.qdrant_client.search = Mock(return_value=mock_scored_points)
        
        # Search single modality
        embedding = np.random.randn(512).astype(np.float32)
        result = await search_engine._search_modality("video", embedding, None)
        
        # Verify result
        assert result is not None
        assert result.modality == "video"
        assert result.min_distance == 0.1
        assert len(result.scored_points) == 5
    
    @pytest.mark.asyncio
    async def test_search_modality_error(self, search_engine):
        """Test individual modality search error handling"""
        # Mock search to raise exception
        search_engine.qdrant_client.search = Mock(
            side_effect=Exception("Search failed")
        )
        
        # Search single modality
        embedding = np.random.randn(512).astype(np.float32)
        result = await search_engine._search_modality("video", embedding, None)
        
        # Verify error was handled
        assert result is None


class TestSearchResultDataclass:
    """Test SearchResult dataclass"""
    
    def test_search_result_creation(self):
        """Test creating SearchResult"""
        points = [Mock(score=0.1), Mock(score=0.2)]
        result = SearchResult(
            modality="video",
            scored_points=points,
            min_distance=0.1,
            mean_distance=0.15
        )
        
        assert result.modality == "video"
        assert len(result.scored_points) == 2
        assert result.min_distance == 0.1
        assert result.mean_distance == 0.15


class TestIntegrationScenarios:
    """Integration test scenarios"""
    
    @pytest.mark.asyncio
    async def test_full_search_workflow(
        self,
        search_engine,
        sample_embedding,
        mock_scored_points
    ):
        """Test complete search workflow"""
        # Mock search
        search_engine.qdrant_client.search = Mock(return_value=mock_scored_points)
        
        # 1. Search baselines
        search_results = await search_engine.search_baselines(
            embedding=sample_embedding,
            shift="morning",
            plant_zone="Zone_A"
        )
        
        # 2. Compute anomaly scores
        anomaly_scores = search_engine.compute_anomaly_scores(search_results)
        
        # 3. Verify workflow
        assert len(search_results) == 3
        assert len(anomaly_scores) == 3
        assert all(modality in anomaly_scores for modality in search_results.keys())
        
        # 4. Check statistics
        stats = search_engine.get_stats()
        assert stats["total_searches"] == 1
        assert stats["successful_searches"] == 1
        assert stats["success_rate"] == 1.0
