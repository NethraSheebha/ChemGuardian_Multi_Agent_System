"""Unit tests for SOP integration."""

import json
import pytest
import tempfile
from pathlib import Path
from src.integrations.sop_integration import SOPIntegration


@pytest.fixture
def sample_sop_data():
    """Sample SOP data for testing."""
    return {
        "zone_a": {
            "mild": [
                "Notify shift supervisor immediately",
                "Increase monitoring frequency"
            ],
            "medium": [
                "Activate emergency ventilation",
                "Evacuate non-essential personnel"
            ],
            "high": [
                "ACTIVATE EMERGENCY ALARM",
                "Evacuate all personnel immediately"
            ]
        },
        "zone_b": {
            "mild": [
                "Check reactor temperature",
                "Document incident"
            ],
            "medium": [
                "Reduce reactor feed rate",
                "Isolate affected process lines"
            ],
            "high": [
                "Initiate emergency shutdown",
                "Notify local authorities"
            ]
        }
    }


@pytest.fixture
def sop_json_file(sample_sop_data):
    """Create temporary JSON file with SOP data."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_sop_data, f)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    Path(temp_path).unlink()


def test_sop_integration_initialization(sop_json_file):
    """Test SOP integration initialization."""
    sop = SOPIntegration(sop_json_file)
    
    assert len(sop.sop_db) == 2
    assert 'zone_a' in sop.sop_db
    assert 'zone_b' in sop.sop_db


def test_get_procedures_found(sop_json_file):
    """Test retrieving procedures that exist."""
    sop = SOPIntegration(sop_json_file)
    
    procedures = sop.get_procedures('zone_a', 'mild')
    
    assert len(procedures) == 2
    assert 'Notify shift supervisor immediately' in procedures
    assert 'Increase monitoring frequency' in procedures


def test_get_procedures_case_insensitive(sop_json_file):
    """Test case-insensitive zone and severity lookup."""
    sop = SOPIntegration(sop_json_file)
    
    # Test various cases
    assert len(sop.get_procedures('ZONE_A', 'MILD')) == 2
    assert len(sop.get_procedures('Zone_A', 'Mild')) == 2
    assert len(sop.get_procedures('zone_a', 'mild')) == 2
    assert len(sop.get_procedures('  zone_a  ', '  mild  ')) == 2


def test_get_procedures_zone_not_found(sop_json_file):
    """Test retrieving procedures for non-existent zone."""
    sop = SOPIntegration(sop_json_file)
    
    procedures = sop.get_procedures('zone_x', 'mild')
    
    assert len(procedures) == 0


def test_get_procedures_severity_not_found(sop_json_file):
    """Test retrieving procedures for non-existent severity."""
    sop = SOPIntegration(sop_json_file)
    
    procedures = sop.get_procedures('zone_a', 'critical')
    
    assert len(procedures) == 0


def test_get_procedures_invalid_severity(sop_json_file):
    """Test retrieving procedures with invalid severity level."""
    sop = SOPIntegration(sop_json_file)
    
    procedures = sop.get_procedures('zone_a', 'invalid')
    
    assert len(procedures) == 0


def test_get_all_zones(sop_json_file):
    """Test getting list of all zones."""
    sop = SOPIntegration(sop_json_file)
    
    zones = sop.get_all_zones()
    
    assert len(zones) == 2
    assert 'zone_a' in zones
    assert 'zone_b' in zones


def test_get_severities_for_zone(sop_json_file):
    """Test getting severities for a specific zone."""
    sop = SOPIntegration(sop_json_file)
    
    severities = sop.get_severities_for_zone('zone_a')
    
    assert len(severities) == 3
    assert 'mild' in severities
    assert 'medium' in severities
    assert 'high' in severities


def test_get_severities_for_nonexistent_zone(sop_json_file):
    """Test getting severities for non-existent zone."""
    sop = SOPIntegration(sop_json_file)
    
    severities = sop.get_severities_for_zone('zone_x')
    
    assert len(severities) == 0


def test_high_severity_procedures(sop_json_file):
    """Test retrieving high severity procedures."""
    sop = SOPIntegration(sop_json_file)
    
    procedures = sop.get_procedures('zone_a', 'high')
    
    assert len(procedures) == 2
    assert 'ACTIVATE EMERGENCY ALARM' in procedures
    assert 'Evacuate all personnel immediately' in procedures


def test_medium_severity_procedures(sop_json_file):
    """Test retrieving medium severity procedures."""
    sop = SOPIntegration(sop_json_file)
    
    procedures = sop.get_procedures('zone_b', 'medium')
    
    assert len(procedures) == 2
    assert 'Reduce reactor feed rate' in procedures
    assert 'Isolate affected process lines' in procedures


def test_sop_integration_missing_file():
    """Test SOP integration with missing file."""
    sop = SOPIntegration('nonexistent_file.json')
    
    # Should initialize with empty database
    assert len(sop.sop_db) == 0
    assert len(sop.get_procedures('zone_a', 'mild')) == 0


def test_sop_integration_invalid_format():
    """Test SOP integration with invalid file format."""
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
        temp_path = f.name
    
    try:
        with pytest.raises(ValueError, match="Unsupported database format"):
            SOPIntegration(temp_path)
    finally:
        Path(temp_path).unlink()


def test_multiple_zones_different_severities(sop_json_file):
    """Test that different zones can have different procedures."""
    sop = SOPIntegration(sop_json_file)
    
    zone_a_mild = sop.get_procedures('zone_a', 'mild')
    zone_b_mild = sop.get_procedures('zone_b', 'mild')
    
    # Both should have procedures
    assert len(zone_a_mild) > 0
    assert len(zone_b_mild) > 0
    
    # But they should be different
    assert zone_a_mild != zone_b_mild


def test_all_severity_levels_present(sop_json_file):
    """Test that all severity levels are present for zone_a."""
    sop = SOPIntegration(sop_json_file)
    
    mild = sop.get_procedures('zone_a', 'mild')
    medium = sop.get_procedures('zone_a', 'medium')
    high = sop.get_procedures('zone_a', 'high')
    
    assert len(mild) > 0
    assert len(medium) > 0
    assert len(high) > 0
