"""Unit tests for RetrainingManager"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch, AsyncMock
import numpy as np
import torch

from src.agents.retraining_manager import (
    RetrainingManager,
    RetrainingConfig,
    ReplayBuffer,
    AnomalyDataset
)
from src.agents.labeled_anomaly_store import LabeledAnomalyStore, OperatorFeedback
from src.agents.adaptive_threshold_manager import AdaptiveThresholdManager
from src.models.sensor_adapter import SensorEmbeddingAdapter


@pytest.fixture
def mock_labeled_store():
    """Create mock labeled anomaly store"""
    store = Mock(spec=LabeledAnomalyStore)
    store.get_labeled_count = AsyncMock(return_value=1000)
    store.get_labeled_anomalies = AsyncMock(return_value=[])
    return store


@pytest.fixture
def mock_threshold_manager():
    """Create mock threshold manager"""
    return Mock(spec=AdaptiveThresholdManager)


@pytest.fixture
def mock_sensor_adapter():
    """Create mock sensor adapter"""
    adapter = Mock(spec=SensorEmbeddingAdapter)
    adapter.to = Mock(return_value=adapter)
    adapter.train = Mock()
    adapter.parameters = Mock(return_value=[])
    adapter.state_dict = Mock(return_value={})
    return adapter


@pytest.fixture
def retraining_config():
    """Create retraining configuration"""
    return RetrainingConfig(
        labeled_data_threshold=100,
        replay_buffer_size=50,
        batch_size=16,
        num_epochs=2,
        validation_split=0.2
    )


@pytest.fixture
def retraining_manager(mock_labeled_store, mock_threshold_manager, mock_sensor_adapter, retraining_config):
    """Create RetrainingManager instance"""
    return RetrainingManager(
        labeled_anomaly_store=mock_labeled_store,
        threshold_manager=mock_threshold_manager,
        sensor_adapter=mock_sensor_adapter,
        config=retraining_config
    )


def test_replay_buffer_add_samples():
    """Test adding samples to replay buffer"""
    buffer = ReplayBuffer(max_size=10)
    
    embeddings = [np.random.rand(128) for _ in range(5)]
    causes = ["gas_plume"] * 5
    severities = ["high"] * 5
    
    buffer.add_samples(embeddings, causes, severities)
    
    assert len(buffer.samples) == 5


def test_replay_buffer_max_size():
    """Test replay buffer respects max size"""
    buffer = ReplayBuffer(max_size=5)
    
    embeddings = [np.random.rand(128) for _ in range(10)]
    causes = ["gas_plume"] * 10
    severities = ["high"] * 10
    
    buffer.add_samples(embeddings, causes, severities)
    
    # Should only keep last 5 samples
    assert len(buffer.samples) == 5


def test_replay_buffer_get_samples():
    """Test getting samples from replay buffer"""
    buffer = ReplayBuffer(max_size=10)
    
    embeddings = [np.random.rand(128) for _ in range(3)]
    causes = ["gas_plume", "pressure_spike", "valve_malfunction"]
    severities = ["high", "medium", "mild"]
    
    buffer.add_samples(embeddings, causes, severities)
    
    retrieved_embs, retrieved_causes, retrieved_sevs = buffer.get_samples()
    
    assert len(retrieved_embs) == 3
    assert retrieved_causes == causes
    assert retrieved_sevs == severities


def test_replay_buffer_clear():
    """Test clearing replay buffer"""
    buffer = ReplayBuffer(max_size=10)
    
    embeddings = [np.random.rand(128) for _ in range(5)]
    causes = ["gas_plume"] * 5
    severities = ["high"] * 5
    
    buffer.add_samples(embeddings, causes, severities)
    assert len(buffer.samples) == 5
    
    buffer.clear()
    assert len(buffer.samples) == 0


def test_anomaly_dataset():
    """Test AnomalyDataset"""
    embeddings = [np.random.rand(128) for _ in range(10)]
    causes = ["gas_plume"] * 5 + ["pressure_spike"] * 5
    severities = ["high"] * 10
    
    cause_to_idx = {"gas_plume": 0, "pressure_spike": 1}
    severity_to_idx = {"mild": 0, "medium": 1, "high": 2}
    
    dataset = AnomalyDataset(embeddings, causes, severities, cause_to_idx, severity_to_idx)
    
    assert len(dataset) == 10
    
    emb, cause_label, severity_label = dataset[0]
    assert isinstance(emb, torch.Tensor)
    assert emb.shape == (128,)
    assert cause_label == 0
    assert severity_label == 2


@pytest.mark.asyncio
async def test_check_and_trigger_retraining_below_threshold(retraining_manager, mock_labeled_store):
    """Test retraining not triggered when below threshold"""
    mock_labeled_store.get_labeled_count.return_value = 50
    
    triggered = await retraining_manager.check_and_trigger_retraining()
    
    assert triggered is False


@pytest.mark.asyncio
async def test_check_and_trigger_retraining_above_threshold(retraining_manager, mock_labeled_store):
    """Test retraining triggered when above threshold"""
    mock_labeled_store.get_labeled_count.return_value = 150
    
    # Mock the retrain method to return success
    retraining_manager.retrain_embedding_adapters = AsyncMock(return_value=True)
    
    triggered = await retraining_manager.check_and_trigger_retraining()
    
    assert triggered is True
    retraining_manager.retrain_embedding_adapters.assert_called_once()


@pytest.mark.asyncio
async def test_retrain_embedding_adapters_insufficient_data(retraining_manager, mock_labeled_store):
    """Test retraining fails with insufficient data"""
    mock_labeled_store.get_labeled_anomalies.return_value = []
    
    success = await retraining_manager.retrain_embedding_adapters()
    
    assert success is False


@pytest.mark.asyncio
async def test_retrain_embedding_adapters_success(retraining_manager, mock_labeled_store):
    """Test successful retraining"""
    # Create mock labeled anomalies
    labeled_anomalies = []
    for i in range(50):
        anomaly = {
            "anomaly_id": f"test_{i}",
            "vectors": {
                "sensor": np.random.rand(128).tolist()
            },
            "payload": {
                "ground_truth_cause": "gas_plume" if i < 25 else "pressure_spike",
                "ground_truth_severity": "high" if i < 30 else "medium"
            }
        }
        labeled_anomalies.append(anomaly)
    
    mock_labeled_store.get_labeled_anomalies.return_value = labeled_anomalies
    
    # Mock training and evaluation
    retraining_manager._train_model = AsyncMock(return_value=0.85)
    retraining_manager._evaluate_on_replay_buffer = AsyncMock(return_value=0.90)
    retraining_manager._update_thresholds_from_validation = AsyncMock()
    retraining_manager._save_model = AsyncMock()
    
    success = await retraining_manager.retrain_embedding_adapters()
    
    assert success is True
    assert retraining_manager.stats["retraining_count"] == 1
    assert retraining_manager.stats["last_validation_accuracy"] == 0.85


@pytest.mark.asyncio
async def test_retrain_with_replay_buffer(retraining_manager, mock_labeled_store):
    """Test retraining with replay buffer samples"""
    # Add samples to replay buffer
    replay_embeddings = [np.random.rand(128) for _ in range(10)]
    replay_causes = ["gas_plume"] * 10
    replay_severities = ["high"] * 10
    
    retraining_manager.replay_buffer.add_samples(
        replay_embeddings, replay_causes, replay_severities
    )
    
    # Create mock labeled anomalies
    labeled_anomalies = []
    for i in range(30):
        anomaly = {
            "anomaly_id": f"test_{i}",
            "vectors": {
                "sensor": np.random.rand(128).tolist()
            },
            "payload": {
                "ground_truth_cause": "pressure_spike",
                "ground_truth_severity": "medium"
            }
        }
        labeled_anomalies.append(anomaly)
    
    mock_labeled_store.get_labeled_anomalies.return_value = labeled_anomalies
    
    # Mock methods
    retraining_manager._train_model = AsyncMock(return_value=0.85)
    retraining_manager._evaluate_on_replay_buffer = AsyncMock(side_effect=[0.90, 0.88])
    retraining_manager._update_thresholds_from_validation = AsyncMock()
    retraining_manager._save_model = AsyncMock()
    
    success = await retraining_manager.retrain_embedding_adapters()
    
    assert success is True
    # Should evaluate replay buffer twice (before and after training)
    assert retraining_manager._evaluate_on_replay_buffer.call_count == 2


@pytest.mark.asyncio
async def test_catastrophic_forgetting_detection(retraining_manager, mock_labeled_store):
    """Test detection of catastrophic forgetting"""
    # Add samples to replay buffer
    replay_embeddings = [np.random.rand(128) for _ in range(10)]
    replay_causes = ["gas_plume"] * 10
    replay_severities = ["high"] * 10
    
    retraining_manager.replay_buffer.add_samples(
        replay_embeddings, replay_causes, replay_severities
    )
    
    # Create mock labeled anomalies
    labeled_anomalies = []
    for i in range(30):
        anomaly = {
            "anomaly_id": f"test_{i}",
            "vectors": {
                "sensor": np.random.rand(128).tolist()
            },
            "payload": {
                "ground_truth_cause": "pressure_spike",
                "ground_truth_severity": "medium"
            }
        }
        labeled_anomalies.append(anomaly)
    
    mock_labeled_store.get_labeled_anomalies.return_value = labeled_anomalies
    
    # Mock methods - simulate catastrophic forgetting
    # Old accuracy: 0.90, New accuracy: 0.50 (significant drop)
    retraining_manager._train_model = AsyncMock(return_value=0.85)
    retraining_manager._evaluate_on_replay_buffer = AsyncMock(side_effect=[0.90, 0.50])
    retraining_manager._update_thresholds_from_validation = AsyncMock()
    retraining_manager._save_model = AsyncMock()
    
    success = await retraining_manager.retrain_embedding_adapters()
    
    # Should fail due to catastrophic forgetting
    assert success is False
    assert retraining_manager.stats["catastrophic_forgetting_detected"] == 1


def test_get_stats(retraining_manager):
    """Test getting statistics"""
    retraining_manager.stats["retraining_count"] = 5
    retraining_manager.stats["last_validation_accuracy"] = 0.87
    
    stats = retraining_manager.get_stats()
    
    assert stats["retraining_count"] == 5
    assert stats["last_validation_accuracy"] == 0.87


def test_reset_stats(retraining_manager):
    """Test resetting statistics"""
    retraining_manager.stats["retraining_count"] = 5
    retraining_manager.stats["last_validation_accuracy"] = 0.87
    
    retraining_manager.reset_stats()
    
    assert retraining_manager.stats["retraining_count"] == 0
    assert retraining_manager.stats["last_validation_accuracy"] == 0.0


def test_retraining_config_defaults():
    """Test default retraining configuration"""
    config = RetrainingConfig()
    
    assert config.labeled_data_threshold == 1000
    assert config.replay_buffer_size == 200
    assert config.batch_size == 32
    assert config.learning_rate == 0.001
    assert config.num_epochs == 10
    assert config.validation_split == 0.2
    assert config.catastrophic_forgetting_threshold == 0.9
