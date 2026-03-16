"""Unit tests for ModelVersionManager"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from pathlib import Path
import tempfile
import json

from src.agents.model_version_manager import (
    ModelVersionManager,
    ModelVersion
)


@pytest.fixture
def mock_qdrant_client():
    """Create mock Qdrant client"""
    client = Mock()
    client.upsert = Mock()
    client.scroll = Mock()
    client.retrieve = Mock()
    return client


@pytest.fixture
def temp_version_path():
    """Create temporary directory for version metadata"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def version_manager(mock_qdrant_client, temp_version_path):
    """Create ModelVersionManager instance"""
    return ModelVersionManager(
        qdrant_client=mock_qdrant_client,
        version_metadata_path=temp_version_path
    )


def test_model_version_to_dict():
    """Test ModelVersion to_dict conversion"""
    version = ModelVersion(
        version_id="sensor_adapter_v1",
        version_number=1,
        model_type="sensor_adapter",
        model_path="/models/sensor_v1.pt",
        training_metrics={"accuracy": 0.85, "loss": 0.15},
        backward_compatible=True
    )
    
    result = version.to_dict()
    
    assert result["version_id"] == "sensor_adapter_v1"
    assert result["version_number"] == 1
    assert result["model_type"] == "sensor_adapter"
    assert result["training_metrics"]["accuracy"] == 0.85
    assert result["backward_compatible"] is True


def test_model_version_from_dict():
    """Test ModelVersion from_dict conversion"""
    data = {
        "version_id": "sensor_adapter_v1",
        "version_number": 1,
        "model_type": "sensor_adapter",
        "model_path": "/models/sensor_v1.pt",
        "baseline_collection_version": None,
        "created_at": datetime.utcnow().isoformat(),
        "deployed_at": None,
        "is_active": False,
        "training_metrics": {"accuracy": 0.85},
        "backward_compatible": True,
        "compatibility_notes": ""
    }
    
    version = ModelVersion.from_dict(data)
    
    assert version.version_id == "sensor_adapter_v1"
    assert version.version_number == 1
    assert version.model_type == "sensor_adapter"
    assert version.training_metrics["accuracy"] == 0.85


@pytest.mark.asyncio
async def test_create_version(version_manager, mock_qdrant_client):
    """Test creating a new model version"""
    # Mock scroll to return no existing versions
    mock_qdrant_client.scroll.return_value = ([], None)
    
    version = await version_manager.create_version(
        model_type="sensor_adapter",
        model_path="/models/sensor_v1.pt",
        training_metrics={"accuracy": 0.85, "loss": 0.15},
        backward_compatible=True
    )
    
    assert version.version_id == "sensor_adapter_v1"
    assert version.version_number == 1
    assert version.model_type == "sensor_adapter"
    assert version.training_metrics["accuracy"] == 0.85
    assert version_manager.stats["total_versions"] == 1


@pytest.mark.asyncio
async def test_create_version_increments_number(version_manager, mock_qdrant_client):
    """Test version number increments correctly"""
    # Mock scroll to return existing version
    mock_point = Mock()
    mock_point.payload = {"version_number": 2}
    mock_qdrant_client.scroll.return_value = ([mock_point], None)
    
    version = await version_manager.create_version(
        model_type="sensor_adapter",
        model_path="/models/sensor_v3.pt"
    )
    
    assert version.version_number == 3
    assert version.version_id == "sensor_adapter_v3"


@pytest.mark.asyncio
async def test_deploy_new_version_success(version_manager, mock_qdrant_client):
    """Test successful deployment of new version"""
    # Create a version first
    mock_qdrant_client.scroll.return_value = ([], None)
    version = await version_manager.create_version(
        model_type="sensor_adapter",
        model_path="/models/sensor_v1.pt",
        backward_compatible=True
    )
    
    # Mock retrieve to return the version
    mock_point = Mock()
    mock_point.payload = version.to_dict()
    mock_qdrant_client.retrieve.return_value = [mock_point]
    
    # Deploy
    success = await version_manager.deploy_new_version(version.version_id)
    
    assert success is True
    assert version_manager.stats["deployments"] == 1
    assert version_manager.active_versions["sensor_adapter"].version_id == version.version_id


@pytest.mark.asyncio
async def test_deploy_version_not_found(version_manager, mock_qdrant_client):
    """Test deployment fails when version not found"""
    mock_qdrant_client.retrieve.return_value = []
    
    success = await version_manager.deploy_new_version("nonexistent_v1")
    
    assert success is False


@pytest.mark.asyncio
async def test_deploy_replaces_active_version(version_manager, mock_qdrant_client):
    """Test deploying new version replaces active version"""
    # Create and deploy first version
    mock_qdrant_client.scroll.return_value = ([], None)
    version1 = await version_manager.create_version(
        model_type="sensor_adapter",
        model_path="/models/sensor_v1.pt"
    )
    
    mock_point1 = Mock()
    mock_point1.payload = version1.to_dict()
    mock_qdrant_client.retrieve.return_value = [mock_point1]
    
    await version_manager.deploy_new_version(version1.version_id)
    
    # Create and deploy second version
    mock_point2_scroll = Mock()
    mock_point2_scroll.payload = {"version_number": 1}
    mock_qdrant_client.scroll.return_value = ([mock_point2_scroll], None)
    
    version2 = await version_manager.create_version(
        model_type="sensor_adapter",
        model_path="/models/sensor_v2.pt"
    )
    
    mock_point2 = Mock()
    mock_point2.payload = version2.to_dict()
    mock_qdrant_client.retrieve.return_value = [mock_point2]
    
    await version_manager.deploy_new_version(version2.version_id)
    
    # Check that version2 is now active
    assert version_manager.active_versions["sensor_adapter"].version_id == version2.version_id
    assert version_manager.stats["deployments"] == 2


@pytest.mark.asyncio
async def test_deploy_incompatible_version_fails(version_manager, mock_qdrant_client):
    """Test deployment fails for incompatible version when previous version exists"""
    # Create and deploy first version
    mock_qdrant_client.scroll.return_value = ([], None)
    version1 = await version_manager.create_version(
        model_type="sensor_adapter",
        model_path="/models/sensor_v1.pt",
        backward_compatible=True
    )
    
    mock_point1 = Mock()
    mock_point1.payload = version1.to_dict()
    mock_qdrant_client.retrieve.return_value = [mock_point1]
    
    success = await version_manager.deploy_new_version(version1.version_id)
    assert success is True
    
    # Create second incompatible version
    mock_point2_scroll = Mock()
    mock_point2_scroll.payload = {"version_number": 1}
    mock_qdrant_client.scroll.return_value = ([mock_point2_scroll], None)
    
    version2 = await version_manager.create_version(
        model_type="sensor_adapter",
        model_path="/models/sensor_v2.pt",
        backward_compatible=False  # Marked as incompatible
    )
    
    mock_point2 = Mock()
    mock_point2.payload = version2.to_dict()
    mock_qdrant_client.retrieve.return_value = [mock_point2]
    
    # Deploy should fail due to incompatibility (previous version exists)
    success = await version_manager.deploy_new_version(version2.version_id)
    assert success is False
    assert version_manager.stats["compatibility_failures"] == 1


@pytest.mark.asyncio
async def test_rollback_to_version(version_manager, mock_qdrant_client):
    """Test rolling back to previous version"""
    # Create two versions
    mock_qdrant_client.scroll.return_value = ([], None)
    version1 = await version_manager.create_version(
        model_type="sensor_adapter",
        model_path="/models/sensor_v1.pt"
    )
    
    mock_point1_scroll = Mock()
    mock_point1_scroll.payload = {"version_number": 1}
    mock_qdrant_client.scroll.return_value = ([mock_point1_scroll], None)
    
    version2 = await version_manager.create_version(
        model_type="sensor_adapter",
        model_path="/models/sensor_v2.pt"
    )
    
    # Deploy version2
    mock_point2 = Mock()
    mock_point2.payload = version2.to_dict()
    mock_qdrant_client.retrieve.return_value = [mock_point2]
    
    await version_manager.deploy_new_version(version2.version_id)
    
    # Rollback to version1
    mock_point1 = Mock()
    mock_point1.payload = version1.to_dict()
    mock_qdrant_client.retrieve.return_value = [mock_point1]
    
    success = await version_manager.rollback_to_version(version1.version_id)
    
    assert success is True
    assert version_manager.stats["rollbacks"] == 1
    assert version_manager.active_versions["sensor_adapter"].version_id == version1.version_id


@pytest.mark.asyncio
async def test_get_version_history(version_manager, mock_qdrant_client):
    """Test getting version history"""
    # Mock scroll to return multiple versions
    mock_points = []
    for i in range(3):
        mock_point = Mock()
        mock_point.payload = {
            "version_id": f"sensor_adapter_v{i+1}",
            "version_number": i + 1,
            "model_type": "sensor_adapter",
            "model_path": f"/models/sensor_v{i+1}.pt",
            "baseline_collection_version": None,
            "created_at": datetime.utcnow().isoformat(),
            "deployed_at": None,
            "is_active": False,
            "training_metrics": {},
            "backward_compatible": True,
            "compatibility_notes": ""
        }
        mock_points.append(mock_point)
    
    mock_qdrant_client.scroll.return_value = (mock_points, None)
    
    history = await version_manager.get_version_history(model_type="sensor_adapter")
    
    assert len(history) == 3
    # Should be sorted by version number descending
    assert history[0].version_number == 3
    assert history[1].version_number == 2
    assert history[2].version_number == 1


@pytest.mark.asyncio
async def test_get_active_version(version_manager, mock_qdrant_client):
    """Test getting active version"""
    # Mock scroll to return active version
    mock_point = Mock()
    mock_point.payload = {
        "version_id": "sensor_adapter_v2",
        "version_number": 2,
        "model_type": "sensor_adapter",
        "model_path": "/models/sensor_v2.pt",
        "baseline_collection_version": None,
        "created_at": datetime.utcnow().isoformat(),
        "deployed_at": datetime.utcnow().isoformat(),
        "is_active": True,
        "training_metrics": {},
        "backward_compatible": True,
        "compatibility_notes": ""
    }
    
    mock_qdrant_client.scroll.return_value = ([mock_point], None)
    
    active_version = await version_manager.get_active_version("sensor_adapter")
    
    assert active_version is not None
    assert active_version.version_id == "sensor_adapter_v2"
    assert active_version.is_active is True


@pytest.mark.asyncio
async def test_get_active_version_none(version_manager, mock_qdrant_client):
    """Test getting active version when none exists"""
    mock_qdrant_client.scroll.return_value = ([], None)
    
    active_version = await version_manager.get_active_version("sensor_adapter")
    
    assert active_version is None


def test_get_stats(version_manager):
    """Test getting statistics"""
    version_manager.stats["total_versions"] = 5
    version_manager.stats["deployments"] = 3
    version_manager.stats["rollbacks"] = 1
    
    stats = version_manager.get_stats()
    
    assert stats["total_versions"] == 5
    assert stats["deployments"] == 3
    assert stats["rollbacks"] == 1


def test_reset_stats(version_manager):
    """Test resetting statistics"""
    version_manager.stats["total_versions"] = 5
    version_manager.stats["deployments"] = 3
    
    version_manager.reset_stats()
    
    assert version_manager.stats["total_versions"] == 0
    assert version_manager.stats["deployments"] == 0


def test_version_metadata_file_creation(version_manager, temp_version_path):
    """Test that version metadata files are created"""
    # Check that directory was created
    assert Path(temp_version_path).exists()
