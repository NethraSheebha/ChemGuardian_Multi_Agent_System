"""Tests for Baseline Manager"""

import asyncio
import pytest
import numpy as np
from datetime import datetime
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from qdrant_client.models import PointStruct, ScoredPoint

from src.agents.baseline_manager import (
    BaselineManager,
    ShiftType,
    BaselineType,
    BaselineMetadata
)
from src.agents.input_collection_agent import MultimodalEmbedding, ModalityStatus


class TestShiftType:
    """Test ShiftType enum"""
    
    def test_shift_types(self):
        """Test shift type values"""
        assert ShiftType.MORNING.value == "morning"
        assert ShiftType.AFTERNOON.value == "afternoon"
        assert ShiftType.NIGHT.value == "night"


class TestBaselineType:
    """Test BaselineType enum"""
    
    def test_baseline_types(self):
        """Test baseline type values"""
        assert BaselineType.SHIFT_SPECIFIC.value == "shift_specific"
        assert BaselineType.EQUIPMENT_SPECIFIC.value == "equipment_specific"
        assert BaselineType.ZONE_SPECIFIC.value == "zone_specific"
        assert BaselineType.GLOBAL.value == "global"


class TestBaselineMetadata:
    """Test BaselineMetadata dataclass"""
    
    def test_metadata_initialization(self):
        """Test basic initialization"""
        metadata = BaselineMetadata(
            baseline_id="test_baseline",
            baseline_type=BaselineType.SHIFT_SPECIFIC,
            shift=ShiftType.MORNING,
            sample_count=10
        )
        
        assert metadata.baseline_id == "test_baseline"
        assert metadata.baseline_type == BaselineType.SHIFT_SPECIFIC
        assert metadata.shift == ShiftType.MORNING
        assert metadata.sample_count == 10
    
    def test_metadata_to_dict(self):
        """Test conversion to dictionary"""
        metadata = BaselineMetadata(
            baseline_id="test_baseline",
            baseline_type=BaselineType.EQUIPMENT_SPECIFIC,
            equipment_id="pump_01",
            plant_zone="zone_a",
            sample_count=20
        )
        
        result = metadata.to_dict()
        
        assert result["baseline_id"] == "test_baseline"
        assert result["baseline_type"] == "equipment_specific"
        assert result["equipment_id"] == "pump_01"
        assert result["plant_zone"] == "zone_a"
        assert result["sample_count"] == 20


class TestBaselineManager:
    """Test BaselineManager class"""
    
    @pytest.fixture
    def mock_qdrant_client(self):
        """Create mock Qdrant client"""
        client = Mock()
        client.upsert = Mock()
        client.retrieve = Mock()
        client.scroll = Mock()
        return client
    
    @pytest.fixture
    def baseline_manager(self, mock_qdrant_client):
        """Create baseline manager instance"""
        return BaselineManager(
            qdrant_client=mock_qdrant_client,
            collection_name="baselines",
            drift_threshold=0.15,
            min_samples_for_baseline=10
        )
    
    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embeddings"""
        embeddings = []
        for i in range(15):
            embedding = MultimodalEmbedding(
                timestamp=datetime.utcnow().isoformat(),
                video_embedding=np.random.randn(512),
                audio_embedding=np.random.randn(512),
                sensor_embedding=np.random.randn(128)
            )
            embeddings.append(embedding)
        return embeddings
    
    def test_manager_initialization(self, mock_qdrant_client):
        """Test manager initialization"""
        manager = BaselineManager(
            qdrant_client=mock_qdrant_client,
            drift_threshold=0.2,
            min_samples_for_baseline=5
        )
        
        assert manager.qdrant_client == mock_qdrant_client
        assert manager.drift_threshold == 0.2
        assert manager.min_samples_for_baseline == 5
        assert manager.stats["baselines_created"] == 0
    
    def test_get_shift_from_timestamp(self, baseline_manager):
        """Test shift determination from timestamp"""
        # Morning shift (6am-2pm)
        morning_time = "2024-01-01T10:00:00"
        assert baseline_manager._get_shift_from_timestamp(morning_time) == ShiftType.MORNING
        
        # Afternoon shift (2pm-10pm)
        afternoon_time = "2024-01-01T16:00:00"
        assert baseline_manager._get_shift_from_timestamp(afternoon_time) == ShiftType.AFTERNOON
        
        # Night shift (10pm-6am)
        night_time = "2024-01-01T23:00:00"
        assert baseline_manager._get_shift_from_timestamp(night_time) == ShiftType.NIGHT
    
    def test_compute_centroid(self, baseline_manager):
        """Test centroid computation"""
        embeddings = [
            np.array([1.0, 2.0, 3.0]),
            np.array([2.0, 3.0, 4.0]),
            np.array([3.0, 4.0, 5.0])
        ]
        
        centroid = baseline_manager._compute_centroid(embeddings)
        
        expected = np.array([2.0, 3.0, 4.0])
        np.testing.assert_array_almost_equal(centroid, expected)
    
    def test_compute_centroid_empty(self, baseline_manager):
        """Test centroid computation with empty list"""
        with pytest.raises(ValueError, match="Cannot compute centroid of empty list"):
            baseline_manager._compute_centroid([])
    
    def test_compute_drift_score_identical(self, baseline_manager):
        """Test drift score for identical embeddings"""
        embedding = np.random.randn(512)
        
        drift = baseline_manager._compute_drift_score(embedding, embedding)
        
        assert drift < 0.01  # Should be very close to 0
    
    def test_compute_drift_score_different(self, baseline_manager):
        """Test drift score for different embeddings"""
        emb1 = np.random.randn(512)
        emb2 = np.random.randn(512)
        
        drift = baseline_manager._compute_drift_score(emb1, emb2)
        
        assert 0.0 <= drift <= 1.0
    
    def test_create_baseline_id(self, baseline_manager):
        """Test baseline ID creation"""
        baseline_id = baseline_manager._create_baseline_id(
            baseline_type=BaselineType.SHIFT_SPECIFIC,
            shift=ShiftType.MORNING,
            equipment_id=None,
            plant_zone="zone_a"
        )
        
        assert "shift_specific" in baseline_id
        assert "morning" in baseline_id
        assert "zone_a" in baseline_id
    
    @pytest.mark.asyncio
    async def test_generate_baseline_insufficient_samples(
        self, baseline_manager, sample_embeddings
    ):
        """Test baseline generation with insufficient samples"""
        # Only 5 samples (need 10)
        result = await baseline_manager.generate_baseline(
            embeddings=sample_embeddings[:5],
            baseline_type=BaselineType.GLOBAL
        )
        
        assert result is None
        assert baseline_manager.stats["baselines_created"] == 0
    
    @pytest.mark.asyncio
    async def test_generate_baseline_success(
        self, baseline_manager, sample_embeddings
    ):
        """Test successful baseline generation"""
        result = await baseline_manager.generate_baseline(
            embeddings=sample_embeddings,
            baseline_type=BaselineType.SHIFT_SPECIFIC,
            shift=ShiftType.MORNING,
            plant_zone="zone_a"
        )
        
        assert result is not None
        assert "shift_specific" in result
        assert "morning" in result
        assert baseline_manager.stats["baselines_created"] == 1
    
    @pytest.mark.asyncio
    async def test_generate_baseline_partial_modalities(
        self, baseline_manager
    ):
        """Test baseline generation with partial modalities"""
        # Create embeddings with only video and sensor
        embeddings = []
        for i in range(15):
            embedding = MultimodalEmbedding(
                timestamp=datetime.utcnow().isoformat(),
                video_embedding=np.random.randn(512),
                sensor_embedding=np.random.randn(128)
            )
            embeddings.append(embedding)
        
        result = await baseline_manager.generate_baseline(
            embeddings=embeddings,
            baseline_type=BaselineType.GLOBAL
        )
        
        assert result is not None
        assert baseline_manager.stats["baselines_created"] == 1
    
    @pytest.mark.asyncio
    async def test_store_baseline(self, baseline_manager, mock_qdrant_client):
        """Test baseline storage in Qdrant"""
        metadata = BaselineMetadata(
            baseline_id="test_baseline",
            baseline_type=BaselineType.GLOBAL,
            sample_count=10
        )
        
        video_baseline = np.random.randn(512)
        audio_baseline = np.random.randn(512)
        sensor_baseline = np.random.randn(128)
        
        await baseline_manager._store_baseline(
            baseline_id="test_baseline",
            video_baseline=video_baseline,
            audio_baseline=audio_baseline,
            sensor_baseline=sensor_baseline,
            metadata=metadata
        )
        
        # Verify upsert was called
        mock_qdrant_client.upsert.assert_called_once()
        call_args = mock_qdrant_client.upsert.call_args
        assert call_args.kwargs["collection_name"] == "baselines"
        assert len(call_args.kwargs["points"]) == 1
    
    @pytest.mark.asyncio
    async def test_retrieve_baseline(self, baseline_manager, mock_qdrant_client):
        """Test baseline retrieval from Qdrant"""
        # Mock Qdrant response
        mock_point = Mock()
        mock_point.vector = {
            "video": np.random.randn(512).tolist(),
            "audio": np.random.randn(512).tolist(),
            "sensor": np.random.randn(128).tolist()
        }
        mock_point.payload = {
            "baseline_id": "test_baseline",
            "baseline_type": "global",
            "sample_count": 10
        }
        mock_qdrant_client.retrieve.return_value = [mock_point]
        
        result = await baseline_manager._retrieve_baseline("test_baseline")
        
        assert result is not None
        assert "video" in result
        assert "audio" in result
        assert "sensor" in result
        assert "metadata" in result
        assert isinstance(result["video"], np.ndarray)
    
    @pytest.mark.asyncio
    async def test_retrieve_baseline_not_found(
        self, baseline_manager, mock_qdrant_client
    ):
        """Test baseline retrieval when not found"""
        mock_qdrant_client.retrieve.return_value = []
        
        result = await baseline_manager._retrieve_baseline("nonexistent")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_update_rolling_baseline_no_drift(
        self, baseline_manager, mock_qdrant_client, sample_embeddings
    ):
        """Test rolling baseline update without drift"""
        # Mock existing baseline
        existing_video = np.random.randn(512)
        existing_audio = np.random.randn(512)
        existing_sensor = np.random.randn(128)
        
        mock_point = Mock()
        mock_point.vector = {
            "video": existing_video.tolist(),
            "audio": existing_audio.tolist(),
            "sensor": existing_sensor.tolist()
        }
        mock_point.payload = {
            "baseline_id": "test_baseline",
            "baseline_type": "global",
            "sample_count": 10,
            "last_updated": datetime.utcnow().isoformat(),
            "drift_score": 0.0
        }
        mock_qdrant_client.retrieve.return_value = [mock_point]
        
        # Update with new embeddings (similar to existing)
        new_embeddings = []
        for i in range(5):
            embedding = MultimodalEmbedding(
                timestamp=datetime.utcnow().isoformat(),
                video_embedding=existing_video + np.random.randn(512) * 0.01,  # Small noise
                audio_embedding=existing_audio + np.random.randn(512) * 0.01,
                sensor_embedding=existing_sensor + np.random.randn(128) * 0.01
            )
            new_embeddings.append(embedding)
        
        result = await baseline_manager.update_rolling_baseline(
            baseline_id="test_baseline",
            new_embeddings=new_embeddings
        )
        
        assert result is True
        assert baseline_manager.stats["baselines_updated"] == 1
        assert baseline_manager.stats["drift_detected_count"] == 0
    
    @pytest.mark.asyncio
    async def test_update_rolling_baseline_with_drift(
        self, baseline_manager, mock_qdrant_client, sample_embeddings
    ):
        """Test rolling baseline update with drift detection"""
        # Mock existing baseline
        existing_video = np.random.randn(512)
        
        mock_point = Mock()
        mock_point.vector = {
            "video": existing_video.tolist(),
            "audio": np.random.randn(512).tolist(),
            "sensor": np.random.randn(128).tolist()
        }
        mock_point.payload = {
            "baseline_id": "test_baseline",
            "baseline_type": "global",
            "sample_count": 10,
            "last_updated": datetime.utcnow().isoformat(),
            "drift_score": 0.0
        }
        mock_qdrant_client.retrieve.return_value = [mock_point]
        
        # Create new embeddings very different from existing (high drift)
        new_embeddings = []
        for i in range(5):
            embedding = MultimodalEmbedding(
                timestamp=datetime.utcnow().isoformat(),
                video_embedding=-existing_video,  # Opposite direction = high drift
                audio_embedding=np.random.randn(512),
                sensor_embedding=np.random.randn(128)
            )
            new_embeddings.append(embedding)
        
        result = await baseline_manager.update_rolling_baseline(
            baseline_id="test_baseline",
            new_embeddings=new_embeddings
        )
        
        assert result is False  # Drift detected, update rejected
        assert baseline_manager.stats["drift_detected_count"] == 1
        assert baseline_manager.stats["recalibrations_triggered"] == 1
    
    @pytest.mark.asyncio
    async def test_update_rolling_baseline_not_found(
        self, baseline_manager, mock_qdrant_client, sample_embeddings
    ):
        """Test rolling baseline update when baseline not found"""
        mock_qdrant_client.retrieve.return_value = []
        
        result = await baseline_manager.update_rolling_baseline(
            baseline_id="nonexistent",
            new_embeddings=sample_embeddings[:5]
        )
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_get_baseline_for_context_specific(
        self, baseline_manager, mock_qdrant_client
    ):
        """Test getting baseline for specific context"""
        # Mock Qdrant scroll response
        mock_point = Mock()
        mock_point.vector = {
            "video": np.random.randn(512).tolist(),
            "audio": np.random.randn(512).tolist(),
            "sensor": np.random.randn(128).tolist()
        }
        mock_point.payload = {
            "baseline_id": "shift_morning_zone_a",
            "baseline_type": "shift_specific",
            "shift": "morning",
            "plant_zone": "zone_a"
        }
        mock_qdrant_client.scroll.return_value = ([mock_point], None)
        
        result = await baseline_manager.get_baseline_for_context(
            shift=ShiftType.MORNING,
            plant_zone="zone_a"
        )
        
        assert result is not None
        assert "video" in result
        assert "metadata" in result
    
    @pytest.mark.asyncio
    async def test_get_baseline_for_context_fallback(
        self, baseline_manager, mock_qdrant_client
    ):
        """Test baseline fallback to less specific"""
        # First call returns empty (no specific baseline)
        # Second call returns global baseline
        mock_point = Mock()
        mock_point.vector = {
            "video": np.random.randn(512).tolist()
        }
        mock_point.payload = {
            "baseline_id": "global_baseline",
            "baseline_type": "global"
        }
        
        mock_qdrant_client.scroll.side_effect = [
            ([], None),  # No shift+zone specific
            ([], None),  # No zone specific
            ([], None),  # No shift specific
            ([mock_point], None)  # Global baseline found
        ]
        
        result = await baseline_manager.get_baseline_for_context(
            shift=ShiftType.MORNING,
            plant_zone="zone_a"
        )
        
        assert result is not None
        assert result["metadata"]["baseline_type"] == "global"
    
    def test_get_stats(self, baseline_manager):
        """Test getting statistics"""
        baseline_manager.stats["baselines_created"] = 5
        baseline_manager.stats["baselines_updated"] = 10
        
        stats = baseline_manager.get_stats()
        
        assert stats["baselines_created"] == 5
        assert stats["baselines_updated"] == 10
        assert "drift_detected_count" in stats
        assert "recalibrations_triggered" in stats
