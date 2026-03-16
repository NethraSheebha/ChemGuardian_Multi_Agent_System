"""Unit tests for MSDS integration."""

import json
import pytest
import tempfile
from pathlib import Path
from src.integrations.msds_integration import MSDSIntegration, ChemicalInfo


@pytest.fixture
def sample_msds_data():
    """Sample MSDS data for testing."""
    return {
        "chlorine": {
            "name": "Chlorine",
            "cas_number": "7782-50-5",
            "exposure_limits": {
                "TWA": 0.5,
                "STEL": 1.0,
                "IDLH": 10.0
            },
            "emergency_procedures": [
                "Evacuate area immediately",
                "Activate emergency ventilation systems"
            ],
            "ppe_requirements": [
                "Self-contained breathing apparatus (SCBA)",
                "Chemical-resistant suit"
            ]
        },
        "ammonia": {
            "name": "Ammonia",
            "cas_number": "7664-41-7",
            "exposure_limits": {
                "TWA": 25.0,
                "STEL": 35.0,
                "IDLH": 300.0
            },
            "emergency_procedures": [
                "Evacuate personnel from affected area",
                "Activate emergency ventilation"
            ],
            "ppe_requirements": [
                "Self-contained breathing apparatus (SCBA)",
                "Chemical-resistant suit"
            ]
        }
    }


@pytest.fixture
def msds_json_file(sample_msds_data):
    """Create temporary JSON file with MSDS data."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_msds_data, f)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    Path(temp_path).unlink()


def test_msds_integration_initialization(msds_json_file):
    """Test MSDS integration initialization."""
    msds = MSDSIntegration(msds_json_file)
    
    assert len(msds.msds_db) == 2
    assert 'chlorine' in msds.msds_db
    assert 'ammonia' in msds.msds_db


def test_get_chemical_info_found(msds_json_file):
    """Test retrieving chemical info that exists."""
    msds = MSDSIntegration(msds_json_file)
    
    chlorine_info = msds.get_chemical_info('chlorine')
    
    assert chlorine_info is not None
    assert chlorine_info.name == 'Chlorine'
    assert chlorine_info.cas_number == '7782-50-5'
    assert chlorine_info.exposure_limits['TWA'] == 0.5
    assert chlorine_info.exposure_limits['STEL'] == 1.0
    assert chlorine_info.exposure_limits['IDLH'] == 10.0
    assert len(chlorine_info.emergency_procedures) == 2
    assert len(chlorine_info.ppe_requirements) == 2


def test_get_chemical_info_case_insensitive(msds_json_file):
    """Test case-insensitive chemical lookup."""
    msds = MSDSIntegration(msds_json_file)
    
    # Test various cases
    assert msds.get_chemical_info('CHLORINE') is not None
    assert msds.get_chemical_info('Chlorine') is not None
    assert msds.get_chemical_info('chlorine') is not None
    assert msds.get_chemical_info('  chlorine  ') is not None


def test_get_chemical_info_not_found(msds_json_file):
    """Test retrieving chemical info that doesn't exist."""
    msds = MSDSIntegration(msds_json_file)
    
    result = msds.get_chemical_info('nonexistent_chemical')
    
    assert result is None


def test_get_all_chemicals(msds_json_file):
    """Test getting list of all chemicals."""
    msds = MSDSIntegration(msds_json_file)
    
    chemicals = msds.get_all_chemicals()
    
    assert len(chemicals) == 2
    assert 'Chlorine' in chemicals
    assert 'Ammonia' in chemicals


def test_chemical_info_dataclass():
    """Test ChemicalInfo dataclass."""
    info = ChemicalInfo(
        name='Test Chemical',
        cas_number='123-45-6',
        exposure_limits={'TWA': 10.0},
        emergency_procedures=['Evacuate'],
        ppe_requirements=['Respirator']
    )
    
    assert info.name == 'Test Chemical'
    assert info.cas_number == '123-45-6'
    assert info.exposure_limits['TWA'] == 10.0
    assert len(info.emergency_procedures) == 1
    assert len(info.ppe_requirements) == 1


def test_msds_integration_missing_file():
    """Test MSDS integration with missing file."""
    msds = MSDSIntegration('nonexistent_file.json')
    
    # Should initialize with empty database
    assert len(msds.msds_db) == 0
    assert msds.get_chemical_info('chlorine') is None


def test_msds_integration_invalid_format():
    """Test MSDS integration with invalid file format."""
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
        temp_path = f.name
    
    try:
        with pytest.raises(ValueError, match="Unsupported database format"):
            MSDSIntegration(temp_path)
    finally:
        Path(temp_path).unlink()


def test_ammonia_exposure_limits(msds_json_file):
    """Test ammonia exposure limits."""
    msds = MSDSIntegration(msds_json_file)
    
    ammonia_info = msds.get_chemical_info('ammonia')
    
    assert ammonia_info is not None
    assert ammonia_info.exposure_limits['TWA'] == 25.0
    assert ammonia_info.exposure_limits['STEL'] == 35.0
    assert ammonia_info.exposure_limits['IDLH'] == 300.0


def test_emergency_procedures_present(msds_json_file):
    """Test that emergency procedures are loaded correctly."""
    msds = MSDSIntegration(msds_json_file)
    
    chlorine_info = msds.get_chemical_info('chlorine')
    
    assert chlorine_info is not None
    assert 'Evacuate area immediately' in chlorine_info.emergency_procedures
    assert 'Activate emergency ventilation systems' in chlorine_info.emergency_procedures


def test_ppe_requirements_present(msds_json_file):
    """Test that PPE requirements are loaded correctly."""
    msds = MSDSIntegration(msds_json_file)
    
    chlorine_info = msds.get_chemical_info('chlorine')
    
    assert chlorine_info is not None
    assert 'Self-contained breathing apparatus (SCBA)' in chlorine_info.ppe_requirements
    assert 'Chemical-resistant suit' in chlorine_info.ppe_requirements
