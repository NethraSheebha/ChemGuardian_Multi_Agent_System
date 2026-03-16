"""Unit tests for Qdrant schema creation"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from qdrant_client.models import Distance, VectorParams, CollectionInfo, CollectionsResponse
from src.database.schemas import QdrantSchemas


class TestQdrantSchemas:
    """Test Qdrant schema creation and management"""
    
    @pytest.fixture
    def mock_client(self):
        """Create a mock Qdrant client"""
        client = Mock()
        client.get_collections.return_value = CollectionsResponse(collections=[])
        client.create_collection = Mock()
        client.create_payload_index = Mock()
        client.delete_collection = Mock()
        return client
        
    @pytest.fixture
    def schemas(self, mock_client):
        """Create QdrantSchemas instance with mock client"""
        return QdrantSchemas(mock_client)
        
    def test_create_baselines_collection(self, schemas, mock_client):
        """Test baselines collection creation"""
        schemas.create_baselines_collection()
        
        # Verify collection was created
        mock_client.create_collection.assert_called_once()
        call_args = mock_client.create_collection.call_args
        
        assert call_args.kwargs['collection_name'] == 'baselines'
        assert 'video' in call_args.kwargs['vectors_config']
        assert 'audio' in call_args.kwargs['vectors_config']
        assert 'sensor' in call_args.kwargs['vectors_config']
        
        # Verify vector dimensions (updated to 512 for video and audio)
        assert call_args.kwargs['vectors_config']['video'].size == 512
        assert call_args.kwargs['vectors_config']['audio'].size == 512
        assert call_args.kwargs['vectors_config']['sensor'].size == 128
        
        # Verify distance metrics
        assert call_args.kwargs['vectors_config']['video'].distance == Distance.COSINE
        assert call_args.kwargs['vectors_config']['audio'].distance == Distance.COSINE
        assert call_args.kwargs['vectors_config']['sensor'].distance == Distance.EUCLID
        
        # Verify payload indexes were created
        assert mock_client.create_payload_index.call_count == 4
        
    def test_create_data_collection(self, schemas, mock_client):
        """Test data collection creation"""
        schemas.create_data_collection()
        
        # Verify collection was created
        mock_client.create_collection.assert_called_once()
        call_args = mock_client.create_collection.call_args
        
        assert call_args.kwargs['collection_name'] == 'data'
        assert 'video' in call_args.kwargs['vectors_config']
        assert 'audio' in call_args.kwargs['vectors_config']
        assert 'sensor' in call_args.kwargs['vectors_config']
        
        # Verify payload indexes for time-window and location queries
        assert mock_client.create_payload_index.call_count == 5
        
    def test_create_labeled_anomalies_collection(self, schemas, mock_client):
        """Test labeled_anomalies collection creation"""
        schemas.create_labeled_anomalies_collection()
        
        # Verify collection was created
        mock_client.create_collection.assert_called_once()
        call_args = mock_client.create_collection.call_args
        
        assert call_args.kwargs['collection_name'] == 'labeled_anomalies'
        
        # Verify payload indexes
        assert mock_client.create_payload_index.call_count == 3
        
    def test_create_response_strategies_collection(self, schemas, mock_client):
        """Test response_strategies collection creation"""
        schemas.create_response_strategies_collection()
        
        # Verify collection was created
        mock_client.create_collection.assert_called_once()
        call_args = mock_client.create_collection.call_args
        
        assert call_args.kwargs['collection_name'] == 'response_strategies'
        assert 'incident_embedding' in call_args.kwargs['vectors_config']
        
        # Verify incident embedding dimension
        assert call_args.kwargs['vectors_config']['incident_embedding'].size == 128
        
        # Verify payload indexes
        assert mock_client.create_payload_index.call_count == 4
        
    def test_initialize_all_collections(self, schemas, mock_client):
        """Test initialization of all collections"""
        schemas.initialize_all_collections()
        
        # Verify all 4 collections were created
        assert mock_client.create_collection.call_count == 4
        
    def test_collection_already_exists(self, mock_client):
        """Test that existing collections are not recreated"""
        # Mock existing collection with proper structure
        from qdrant_client.models import CollectionDescription
        
        # Create a mock collection with the name attribute
        existing_collection = Mock(spec=CollectionDescription)
        existing_collection.name = 'baselines'
        
        mock_response = Mock()
        mock_response.collections = [existing_collection]
        mock_client.get_collections.return_value = mock_response
        
        schemas = QdrantSchemas(mock_client)
        schemas.create_baselines_collection()
        
        # Verify collection was not created again
        mock_client.create_collection.assert_not_called()
        
    def test_delete_all_collections(self, schemas, mock_client):
        """Test deletion of all collections"""
        schemas.delete_all_collections()
        
        # Verify all 4 collections were deleted
        assert mock_client.delete_collection.call_count == 4
        
    def test_get_collection_info(self, schemas, mock_client):
        """Test getting collection information"""
        # Mock collection info
        mock_info = Mock()
        mock_info.vectors_count = 100
        mock_info.points_count = 100
        mock_info.status = "green"
        mock_info.config = {}
        mock_client.get_collection.return_value = mock_info
        
        info = schemas.get_collection_info('baselines')
        
        assert info['name'] == 'baselines'
        assert info['vectors_count'] == 100
        assert info['points_count'] == 100
        assert info['status'] == 'green'
        
    def test_vector_dimensions_constants(self, schemas):
        """Test that vector dimension constants are correct (updated to 512 for video/audio)"""
        assert schemas.VIDEO_DIM == 512
        assert schemas.AUDIO_DIM == 512
        assert schemas.SENSOR_DIM == 128
        assert schemas.INCIDENT_DIM == 128
        
    def test_collection_name_constants(self, schemas):
        """Test that collection name constants are correct"""
        assert schemas.BASELINES == "baselines"
        assert schemas.DATA == "data"
        assert schemas.LABELED_ANOMALIES == "labeled_anomalies"
        assert schemas.RESPONSE_STRATEGIES == "response_strategies"
