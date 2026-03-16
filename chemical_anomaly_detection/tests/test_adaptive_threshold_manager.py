"""Tests for Adaptive Threshold Manager"""

import pytest
import numpy as np
from datetime import datetime

from src.agents.adaptive_threshold_manager import (
    AdaptiveThresholdManager,
    ThresholdConfig,
    FeedbackRecord
)


class TestThresholdConfig:
    """Test ThresholdConfig dataclass"""
    
    def test_config_initialization(self):
        """Test basic initialization"""
        config = ThresholdConfig(
            modality="video",
            initial_threshold=0.7,
            min_threshold=0.3,
            max_threshold=0.95
        )
        
        assert config.modality == "video"
        assert config.initial_threshold == 0.7
        assert config.current_threshold == 0.7  # Should match initial
        assert config.min_threshold == 0.3
        assert config.max_threshold == 0.95


class TestFeedbackRecord:
    """Test FeedbackRecord dataclass"""
    
    def test_false_positive(self):
        """Test false positive detection"""
        record = FeedbackRecord(
            timestamp=datetime.utcnow().isoformat(),
            modality="video",
            distance_score=0.8,
            predicted_anomaly=True,
            actual_anomaly=False
        )
        
        assert record.is_false_positive is True
        assert record.is_false_negative is False
        assert record.is_correct is False
    
    def test_false_negative(self):
        """Test false negative detection"""
        record = FeedbackRecord(
            timestamp=datetime.utcnow().isoformat(),
            modality="audio",
            distance_score=0.5,
            predicted_anomaly=False,
            actual_anomaly=True
        )
        
        assert record.is_false_positive is False
        assert record.is_false_negative is True
        assert record.is_correct is False
    
    def test_true_positive(self):
        """Test true positive"""
        record = FeedbackRecord(
            timestamp=datetime.utcnow().isoformat(),
            modality="sensor",
            distance_score=3.0,
            predicted_anomaly=True,
            actual_anomaly=True
        )
        
        assert record.is_false_positive is False
        assert record.is_false_negative is False
        assert record.is_correct is True
    
    def test_true_negative(self):
        """Test true negative"""
        record = FeedbackRecord(
            timestamp=datetime.utcnow().isoformat(),
            modality="video",
            distance_score=0.4,
            predicted_anomaly=False,
            actual_anomaly=False
        )
        
        assert record.is_false_positive is False
        assert record.is_false_negative is False
        assert record.is_correct is True


class TestAdaptiveThresholdManager:
    """Test AdaptiveThresholdManager class"""
    
    @pytest.fixture
    def manager(self):
        """Create threshold manager instance"""
        return AdaptiveThresholdManager(
            video_threshold=0.7,
            audio_threshold=0.65,
            sensor_threshold=2.5,
            learning_rate=0.05,
            feedback_window_size=100
        )
    
    def test_manager_initialization(self):
        """Test manager initialization"""
        manager = AdaptiveThresholdManager(
            video_threshold=0.8,
            audio_threshold=0.7,
            sensor_threshold=3.0
        )
        
        assert manager.thresholds["video"].current_threshold == 0.8
        assert manager.thresholds["audio"].current_threshold == 0.7
        assert manager.thresholds["sensor"].current_threshold == 3.0
        assert manager.learning_rate == 0.05
        assert manager.stats["total_predictions"] == 0
    
    def test_is_anomaly_single_modality(self, manager):
        """Test anomaly detection with single modality"""
        # Video score above threshold
        is_anomaly, per_modality = manager.is_anomaly({"video": 0.8})
        
        assert is_anomaly is True
        assert per_modality["video"] is True
        assert manager.stats["total_predictions"] == 1
    
    def test_is_anomaly_below_threshold(self, manager):
        """Test normal detection (below threshold)"""
        # Video score below threshold
        is_anomaly, per_modality = manager.is_anomaly({"video": 0.5})
        
        assert is_anomaly is False
        assert per_modality["video"] is False
    
    def test_is_anomaly_multiple_modalities(self, manager):
        """Test anomaly detection with multiple modalities"""
        scores = {
            "video": 0.8,  # Above threshold (0.7)
            "audio": 0.7,  # Above threshold (0.65)
            "sensor": 3.0  # Above threshold (2.5)
        }
        
        is_anomaly, per_modality = manager.is_anomaly(scores)
        
        assert is_anomaly is True
        assert per_modality["video"] is True
        assert per_modality["audio"] is True
        assert per_modality["sensor"] is True
    
    def test_is_anomaly_mixed_results(self, manager):
        """Test with some modalities above, some below threshold"""
        scores = {
            "video": 0.8,  # Above threshold
            "audio": 0.5,  # Below threshold
            "sensor": 2.0  # Below threshold
        }
        
        is_anomaly, per_modality = manager.is_anomaly(scores)
        
        assert is_anomaly is True  # Any modality is sufficient
        assert per_modality["video"] is True
        assert per_modality["audio"] is False
        assert per_modality["sensor"] is False
    
    def test_is_anomaly_multi_modality_voting(self, manager):
        """Test multi-modality voting requirement"""
        scores = {
            "video": 0.8,  # Above threshold
            "audio": 0.5,  # Below threshold
            "sensor": 2.0  # Below threshold
        }
        
        # Require at least 2 modalities to agree
        is_anomaly, per_modality = manager.is_anomaly(
            scores,
            require_multi_modality=True,
            min_modalities=2
        )
        
        assert is_anomaly is False  # Only 1 modality detected anomaly
    
    def test_is_anomaly_multi_modality_voting_pass(self, manager):
        """Test multi-modality voting with sufficient agreement"""
        scores = {
            "video": 0.8,  # Above threshold
            "audio": 0.7,  # Above threshold
            "sensor": 2.0  # Below threshold
        }
        
        # Require at least 2 modalities to agree
        is_anomaly, per_modality = manager.is_anomaly(
            scores,
            require_multi_modality=True,
            min_modalities=2
        )
        
        assert is_anomaly is True  # 2 modalities detected anomaly
    
    def test_add_feedback(self, manager):
        """Test adding feedback"""
        manager.add_feedback(
            modality="video",
            distance_score=0.8,
            predicted_anomaly=True,
            actual_anomaly=True
        )
        
        assert len(manager.feedback_history["video"]) == 1
        assert manager.stats["total_feedback"] == 1
        assert manager.stats["true_positives"] == 1
    
    def test_add_feedback_false_positive(self, manager):
        """Test adding false positive feedback"""
        manager.add_feedback(
            modality="audio",
            distance_score=0.7,
            predicted_anomaly=True,
            actual_anomaly=False
        )
        
        assert manager.stats["false_positives"] == 1
        assert manager.stats["total_feedback"] == 1
    
    def test_add_feedback_false_negative(self, manager):
        """Test adding false negative feedback"""
        manager.add_feedback(
            modality="sensor",
            distance_score=2.0,
            predicted_anomaly=False,
            actual_anomaly=True
        )
        
        assert manager.stats["false_negatives"] == 1
        assert manager.stats["total_feedback"] == 1
    
    def test_update_thresholds_insufficient_feedback(self, manager):
        """Test threshold update with insufficient feedback"""
        # Add only 5 feedbacks (need 10)
        for i in range(5):
            manager.add_feedback("video", 0.8, True, True)
        
        updated = manager.update_thresholds()
        
        assert len(updated) == 0  # No updates
        assert manager.stats["threshold_updates"] == 0
    
    def test_update_thresholds_high_false_positives(self, manager):
        """Test threshold increase with high false positive rate"""
        # Add many false positives
        for i in range(20):
            manager.add_feedback("video", 0.8, True, False)  # False positive
        
        # Add some true negatives
        for i in range(5):
            manager.add_feedback("video", 0.5, False, False)  # True negative
        
        old_threshold = manager.thresholds["video"].current_threshold
        updated = manager.update_thresholds()
        new_threshold = manager.thresholds["video"].current_threshold
        
        assert "video" in updated
        assert new_threshold > old_threshold  # Threshold should increase
        assert manager.stats["threshold_updates"] > 0
    
    def test_update_thresholds_high_false_negatives(self, manager):
        """Test threshold decrease with high false negative rate"""
        # Add many false negatives
        for i in range(20):
            manager.add_feedback("audio", 0.5, False, True)  # False negative
        
        # Add some true positives
        for i in range(5):
            manager.add_feedback("audio", 0.8, True, True)  # True positive
        
        old_threshold = manager.thresholds["audio"].current_threshold
        updated = manager.update_thresholds()
        new_threshold = manager.thresholds["audio"].current_threshold
        
        assert "audio" in updated
        assert new_threshold < old_threshold  # Threshold should decrease
        assert manager.stats["threshold_updates"] > 0
    
    def test_update_thresholds_clamping(self, manager):
        """Test that thresholds are clamped to min/max"""
        # Try to push threshold very high
        for i in range(100):
            manager.add_feedback("video", 0.9, True, False)  # Many false positives
        
        manager.update_thresholds()
        
        # Should not exceed max threshold
        assert manager.thresholds["video"].current_threshold <= manager.thresholds["video"].max_threshold
    
    def test_get_current_thresholds(self, manager):
        """Test getting current thresholds"""
        thresholds = manager.get_current_thresholds()
        
        assert "video" in thresholds
        assert "audio" in thresholds
        assert "sensor" in thresholds
        assert thresholds["video"] == 0.7
        assert thresholds["audio"] == 0.65
        assert thresholds["sensor"] == 2.5
    
    def test_get_performance_metrics_empty(self, manager):
        """Test performance metrics with no feedback"""
        metrics = manager.get_performance_metrics()
        
        assert len(metrics) == 0  # No metrics without feedback
    
    def test_get_performance_metrics(self, manager):
        """Test performance metrics calculation"""
        # Add mixed feedback
        manager.add_feedback("video", 0.8, True, True)  # TP
        manager.add_feedback("video", 0.8, True, True)  # TP
        manager.add_feedback("video", 0.8, True, False)  # FP
        manager.add_feedback("video", 0.5, False, False)  # TN
        manager.add_feedback("video", 0.5, False, True)  # FN
        
        metrics = manager.get_performance_metrics()
        
        assert "video" in metrics
        assert metrics["video"]["true_positives"] == 2
        assert metrics["video"]["false_positives"] == 1
        assert metrics["video"]["false_negatives"] == 1
        assert metrics["video"]["true_negatives"] == 1
        assert metrics["video"]["total_samples"] == 5
        
        # Check calculated metrics
        assert 0 <= metrics["video"]["precision"] <= 1
        assert 0 <= metrics["video"]["recall"] <= 1
        assert 0 <= metrics["video"]["f1_score"] <= 1
        assert 0 <= metrics["video"]["accuracy"] <= 1
    
    def test_get_stats(self, manager):
        """Test getting statistics"""
        manager.add_feedback("video", 0.8, True, True)
        manager.add_feedback("audio", 0.7, True, False)
        
        stats = manager.get_stats()
        
        assert stats["total_feedback"] == 2
        assert stats["true_positives"] == 1
        assert stats["false_positives"] == 1
        assert "current_thresholds" in stats
        assert "feedback_counts" in stats
    
    def test_reset_feedback_single_modality(self, manager):
        """Test resetting feedback for single modality"""
        manager.add_feedback("video", 0.8, True, True)
        manager.add_feedback("audio", 0.7, True, True)
        
        manager.reset_feedback("video")
        
        assert len(manager.feedback_history["video"]) == 0
        assert len(manager.feedback_history["audio"]) == 1
    
    def test_reset_feedback_all(self, manager):
        """Test resetting all feedback"""
        manager.add_feedback("video", 0.8, True, True)
        manager.add_feedback("audio", 0.7, True, True)
        manager.add_feedback("sensor", 3.0, True, True)
        
        manager.reset_feedback()
        
        assert len(manager.feedback_history["video"]) == 0
        assert len(manager.feedback_history["audio"]) == 0
        assert len(manager.feedback_history["sensor"]) == 0
    
    def test_feedback_window_size_limit(self, manager):
        """Test that feedback history respects window size"""
        # Add more feedbacks than window size
        for i in range(150):
            manager.add_feedback("video", 0.8, True, True)
        
        # Should only keep last 100 (window size)
        assert len(manager.feedback_history["video"]) == 100
    
    def test_adaptive_behavior_over_time(self, manager):
        """Test that thresholds adapt over time with feedback"""
        initial_threshold = manager.thresholds["video"].current_threshold
        
        # Simulate high false positive rate over multiple updates
        for batch in range(5):
            # Add batch of false positives
            for i in range(20):
                manager.add_feedback("video", 0.75, True, False)
            
            # Update thresholds
            manager.update_thresholds()
        
        final_threshold = manager.thresholds["video"].current_threshold
        
        # Threshold should have increased
        assert final_threshold > initial_threshold
