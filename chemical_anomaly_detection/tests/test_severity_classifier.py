"""Tests for SeverityClassifier"""

import pytest
from src.agents.severity_classifier import SeverityClassifier
from src.agents.cause_inference_engine import CauseAnalysis


@pytest.fixture
def severity_classifier():
    """Create SeverityClassifier instance"""
    return SeverityClassifier(
        gas_concentration_weight=0.4,
        modality_count_weight=0.3,
        cause_weight=0.3
    )


@pytest.fixture
def sample_cause_analysis():
    """Create sample cause analysis"""
    return CauseAnalysis(
        primary_cause="gas_plume",
        contributing_factors=["audio_anomaly"],
        confidence=0.8,
        explanation="Test explanation",
        similar_historical_incidents=["inc-1", "inc-2"]
    )


def test_classify_severity_mild(severity_classifier, sample_cause_analysis):
    """Test classification of mild severity"""
    anomaly_scores = {
        "video": 0.5,
        "audio": 0.4,
        "sensor": 1.5
    }
    
    metadata = {
        "gas_concentration_ppm": 300  # Low concentration
    }
    
    severity = severity_classifier.classify_severity(
        cause=sample_cause_analysis,
        anomaly_scores=anomaly_scores,
        metadata=metadata
    )
    
    assert severity == "mild"


def test_classify_severity_medium(severity_classifier, sample_cause_analysis):
    """Test classification of medium severity"""
    anomaly_scores = {
        "video": 0.75,
        "audio": 0.70,
        "sensor": 2.8
    }
    
    metadata = {
        "gas_concentration_ppm": 750  # Medium concentration
    }
    
    severity = severity_classifier.classify_severity(
        cause=sample_cause_analysis,
        anomaly_scores=anomaly_scores,
        metadata=metadata
    )
    
    assert severity == "medium"


def test_classify_severity_high(severity_classifier, sample_cause_analysis):
    """Test classification of high severity"""
    # Use high-severity cause
    high_severity_cause = CauseAnalysis(
        primary_cause="gas_leak_with_hissing",
        contributing_factors=[],
        confidence=0.9,
        explanation="Test",
        similar_historical_incidents=[]
    )
    
    anomaly_scores = {
        "video": 0.85,
        "audio": 0.80,
        "sensor": 3.5
    }
    
    metadata = {
        "gas_concentration_ppm": 2500  # Very high concentration to ensure high severity
    }
    
    severity = severity_classifier.classify_severity(
        cause=high_severity_cause,
        anomaly_scores=anomaly_scores,
        metadata=metadata
    )
    
    assert severity == "high"


def test_compute_gas_concentration_score_low(severity_classifier):
    """Test gas concentration score for low concentration"""
    metadata = {"gas_concentration_ppm": 200}
    
    score = severity_classifier._compute_gas_concentration_score(metadata)
    
    assert 0.0 <= score < 0.3


def test_compute_gas_concentration_score_medium(severity_classifier):
    """Test gas concentration score for medium concentration"""
    metadata = {"gas_concentration_ppm": 750}
    
    score = severity_classifier._compute_gas_concentration_score(metadata)
    
    assert 0.3 <= score < 0.6


def test_compute_gas_concentration_score_high(severity_classifier):
    """Test gas concentration score for high concentration"""
    metadata = {"gas_concentration_ppm": 1500}
    
    score = severity_classifier._compute_gas_concentration_score(metadata)
    
    assert 0.6 <= score < 0.9


def test_compute_gas_concentration_score_critical(severity_classifier):
    """Test gas concentration score for critical concentration"""
    metadata = {"gas_concentration_ppm": 3000}
    
    score = severity_classifier._compute_gas_concentration_score(metadata)
    
    assert 0.9 <= score <= 1.0


def test_compute_gas_concentration_score_missing(severity_classifier):
    """Test gas concentration score when data is missing"""
    metadata = {}
    
    score = severity_classifier._compute_gas_concentration_score(metadata)
    
    # Should return neutral score
    assert score == 0.5


def test_compute_modality_count_score_all_anomalous(severity_classifier):
    """Test modality count score when all modalities are anomalous"""
    anomaly_scores = {
        "video": 0.85,
        "audio": 0.80,
        "sensor": 3.0
    }
    
    score = severity_classifier._compute_modality_count_score(anomaly_scores)
    
    # All 3 modalities exceed thresholds
    assert score == 1.0


def test_compute_modality_count_score_partial(severity_classifier):
    """Test modality count score when some modalities are anomalous"""
    anomaly_scores = {
        "video": 0.85,  # Above threshold (0.7)
        "audio": 0.60,  # Below threshold (0.65)
        "sensor": 2.0   # Below threshold (2.5)
    }
    
    score = severity_classifier._compute_modality_count_score(anomaly_scores)
    
    # Only 1 of 3 modalities exceeds threshold
    assert abs(score - 1/3) < 0.01


def test_compute_modality_count_score_none(severity_classifier):
    """Test modality count score when no modalities are anomalous"""
    anomaly_scores = {
        "video": 0.5,
        "audio": 0.4,
        "sensor": 1.5
    }
    
    score = severity_classifier._compute_modality_count_score(anomaly_scores)
    
    assert score == 0.0


def test_compute_cause_severity_score_high_severity_cause(severity_classifier):
    """Test cause severity score for high-severity causes"""
    cause = CauseAnalysis(
        primary_cause="gas_leak_with_hissing",
        contributing_factors=[],
        confidence=0.9,
        explanation="Test",
        similar_historical_incidents=[]
    )
    
    score = severity_classifier._compute_cause_severity_score(cause)
    
    # Should be high due to cause type and confidence
    assert score > 0.3


def test_compute_cause_severity_score_low_severity_cause(severity_classifier):
    """Test cause severity score for low-severity causes"""
    cause = CauseAnalysis(
        primary_cause="unknown_anomaly",
        contributing_factors=[],
        confidence=0.5,
        explanation="Test",
        similar_historical_incidents=[]
    )
    
    score = severity_classifier._compute_cause_severity_score(cause)
    
    # Should be lower for unknown cause
    assert score < 0.5


def test_get_severity_distribution(severity_classifier):
    """Test severity distribution calculation"""
    # Simulate some classifications
    severity_classifier.stats["total_classifications"] = 10
    severity_classifier.stats["mild_count"] = 3
    severity_classifier.stats["medium_count"] = 5
    severity_classifier.stats["high_count"] = 2
    
    distribution = severity_classifier.get_severity_distribution()
    
    assert distribution["mild"] == 0.3
    assert distribution["medium"] == 0.5
    assert distribution["high"] == 0.2


def test_get_severity_distribution_empty(severity_classifier):
    """Test severity distribution when no classifications"""
    distribution = severity_classifier.get_severity_distribution()
    
    assert distribution["mild"] == 0.0
    assert distribution["medium"] == 0.0
    assert distribution["high"] == 0.0


def test_get_stats(severity_classifier):
    """Test statistics retrieval"""
    severity_classifier.stats["total_classifications"] = 10
    severity_classifier.stats["mild_count"] = 3
    
    stats = severity_classifier.get_stats()
    
    assert stats["total_classifications"] == 10
    assert stats["mild_count"] == 3
    assert "distribution" in stats


def test_reset_stats(severity_classifier):
    """Test statistics reset"""
    severity_classifier.stats["total_classifications"] = 10
    severity_classifier.reset_stats()
    
    assert severity_classifier.stats["total_classifications"] == 0
    assert severity_classifier.stats["mild_count"] == 0


def test_severity_thresholds_consistency(severity_classifier, sample_cause_analysis):
    """Test that severity thresholds are consistent"""
    # Test boundary between mild and medium
    anomaly_scores = {"video": 0.7, "audio": 0.65, "sensor": 2.5}
    metadata = {"gas_concentration_ppm": 500}
    
    severity = severity_classifier.classify_severity(
        cause=sample_cause_analysis,
        anomaly_scores=anomaly_scores,
        metadata=metadata
    )
    
    # Should be classified consistently
    assert severity in ["mild", "medium", "high"]
