"""
CrewAI Implementation of Anomaly Detection Agent
Converts Pythonic agent to CrewAI while preserving all logic
"""

from crewai import Agent, Task, Crew
from crewai.tools import tool
from typing import Dict, Any, Optional, List
from collections import deque
import logging

from qdrant_client import QdrantClient

from src.agents.input_collection_agent import MultimodalEmbedding
from src.agents.similarity_search_engine import SimilaritySearchEngine
from src.agents.adaptive_threshold_manager import AdaptiveThresholdManager
from src.agents.storage_manager import StorageManager
from src.agents.anomaly_detection_agent import AnomalyDetectionResult


logger = logging.getLogger(__name__)


class AnomalyDetectionTools:
    """Tools for Anomaly Detection CrewAI Agent"""
    
    def __init__(
        self,
        similarity_search: SimilaritySearchEngine,
        threshold_manager: AdaptiveThresholdManager,
        storage_manager: StorageManager,
        high_severity_min_modalities: int = 2,
        borderline_threshold_pct: float = 0.1,
        temporal_confirmation_windows: int = 3
    ):
        self.similarity_search = similarity_search
        self.threshold_manager = threshold_manager
        self.storage_manager = storage_manager
        self.high_severity_min_modalities = high_severity_min_modalities
        self.borderline_threshold_pct = borderline_threshold_pct
        self.temporal_confirmation_windows = temporal_confirmation_windows
        
        # Temporal confirmation tracking (preserves original logic)
        self.temporal_history: Dict[str, deque] = {
            "video": deque(maxlen=temporal_confirmation_windows),
            "audio": deque(maxlen=temporal_confirmation_windows),
            "sensor": deque(maxlen=temporal_confirmation_windows)
        }
        
        self.stats = {
            "total_processed": 0,
            "anomalies_detected": 0,
            "normal_detected": 0,
            "high_severity_anomalies": 0,
            "temporal_confirmations": 0,
            "multi_modality_confirmations": 0
        }
    
    # Note: Not using @tool decorators due to type incompatibility
    # Methods are called directly by the crew
    async def search_and_score(
        self,
        embedding: MultimodalEmbedding,
        shift: Optional[str] = None,
        equipment_id: Optional[str] = None,
        plant_zone: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Search baseline collection and compute anomaly scores.
        Uses similarity search to find closest normal patterns.
        
        Args:
            embedding: Multimodal embedding to analyze
            shift: Filter by shift
            equipment_id: Filter by equipment ID
            plant_zone: Filter by plant zone
            
        Returns:
            Dictionary with search results and anomaly scores
        """
        try:
            search_results, anomaly_scores = await self.similarity_search.search_and_score(
                embedding=embedding,
                shift=shift,
                equipment_id=equipment_id,
                plant_zone=plant_zone
            )
            
            return {
                "success": True,
                "anomaly_scores": anomaly_scores,
                "search_results": search_results
            }
        except Exception as e:
            logger.error(f"Search and score failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "anomaly_scores": {},
                "search_results": {}
            }
    
    def apply_thresholds(
        self,
        anomaly_scores: Dict[str, float],
        require_multi_modality: bool = False
    ) -> Dict[str, Any]:
        """
        Apply adaptive thresholds to anomaly scores.
        Determines if scores indicate anomalous behavior.
        
        Args:
            anomaly_scores: Per-modality anomaly scores
            require_multi_modality: Whether to require multiple modalities
            
        Returns:
            Dictionary with anomaly decision and per-modality results
        """
        is_anomaly, per_modality = self.threshold_manager.is_anomaly(
            distance_scores=anomaly_scores,
            require_multi_modality=require_multi_modality
        )
        
        return {
            "is_anomaly": is_anomaly,
            "per_modality_decisions": per_modality,
            "anomaly_count": sum(per_modality.values()),
            "thresholds": self.threshold_manager.get_current_thresholds()
        }
    
    def check_temporal(
        self,
        per_modality: Dict[str, bool]
    ) -> Dict[str, Any]:
        """
        Check if temporal confirmation is achieved.
        Requires consecutive anomaly detections over multiple windows.
        
        Args:
            per_modality: Current per-modality decisions
            
        Returns:
            Dictionary with temporal confirmation status
        """
        max_consecutive = 0
        
        for modality, is_anomaly in per_modality.items():
            if modality not in self.temporal_history:
                continue
            
            history = self.temporal_history[modality]
            consecutive = sum(1 for d in reversed(history) if d)
            
            if is_anomaly:
                consecutive += 1
            
            max_consecutive = max(max_consecutive, consecutive)
        
        is_confirmed = max_consecutive >= self.temporal_confirmation_windows
        
        # Update history
        for modality, is_anomaly in per_modality.items():
            if modality in self.temporal_history:
                self.temporal_history[modality].append(is_anomaly)
        
        if is_confirmed:
            self.stats["temporal_confirmations"] += 1
        
        return {
            "is_confirmed": is_confirmed,
            "max_consecutive": max_consecutive,
            "required_windows": self.temporal_confirmation_windows
        }
    
    async def store_embedding(
        self,
        embedding: MultimodalEmbedding,
        is_anomaly: bool,
        anomaly_scores: Dict[str, float],
        confidence: float
    ) -> Dict[str, Any]:
        """
        Store embedding in Qdrant with anomaly label.
        
        Args:
            embedding: Multimodal embedding
            is_anomaly: Whether it's an anomaly
            anomaly_scores: Per-modality scores
            confidence: Confidence score
            
        Returns:
            Dictionary with storage status
        """
        try:
            await self.storage_manager.store_embedding(
                embedding=embedding,
                is_anomaly=is_anomaly,
                anomaly_scores=anomaly_scores,
                confidence=confidence
            )
            
            # Update stats
            self.stats["total_processed"] += 1
            if is_anomaly:
                self.stats["anomalies_detected"] += 1
            else:
                self.stats["normal_detected"] += 1
            
            return {
                "success": True,
                "stored": True
            }
        except Exception as e:
            logger.error(f"Storage failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get tool statistics"""
        stats = self.stats.copy()
        if stats["total_processed"] > 0:
            stats["anomaly_rate"] = stats["anomalies_detected"] / stats["total_processed"]
        else:
            stats["anomaly_rate"] = 0.0
        return stats


class AnomalyDetectionCrew:
    """
    CrewAI Crew for Anomaly Detection
    
    Maintains all logic from original AnomalyDetectionAgent:
    - Similarity search against baselines
    - Adaptive threshold application
    - Multi-modality confirmation for high severity
    - Temporal confirmation for borderline cases
    - Storage with anomaly labels
    """
    
    def __init__(
        self,
        qdrant_client: QdrantClient,
        similarity_search_engine: SimilaritySearchEngine,
        adaptive_threshold_manager: AdaptiveThresholdManager,
        storage_manager: StorageManager,
        processing_interval: float = 1.0,
        high_severity_min_modalities: int = 2,
        borderline_threshold_pct: float = 0.1,
        temporal_confirmation_windows: int = 3
    ):
        """
        Initialize Anomaly Detection Crew
        
        Args:
            qdrant_client: Qdrant client instance
            similarity_search_engine: Similarity search engine
            adaptive_threshold_manager: Adaptive threshold manager
            storage_manager: Storage manager
            processing_interval: Processing interval
            high_severity_min_modalities: Min modalities for high severity
            borderline_threshold_pct: Borderline threshold percentage
            temporal_confirmation_windows: Windows for temporal confirmation
        """
        self.qdrant_client = qdrant_client
        self.processing_interval = processing_interval
        
        # Initialize tools (preserves original logic)
        self.tools = AnomalyDetectionTools(
            similarity_search=similarity_search_engine,
            threshold_manager=adaptive_threshold_manager,
            storage_manager=storage_manager,
            high_severity_min_modalities=high_severity_min_modalities,
            borderline_threshold_pct=borderline_threshold_pct,
            temporal_confirmation_windows=temporal_confirmation_windows
        )
        
        # Note: Agent creation removed to avoid LLM requirement
        # The crew works without CrewAI Agent objects, using direct method calls
        self.agent = None
        
        logger.info(
            f"Initialized AnomalyDetectionCrew: "
            f"interval={processing_interval}s, "
            f"high_severity_min={high_severity_min_modalities}"
        )
    
    async def detect_anomaly(
        self,
        embedding: MultimodalEmbedding,
        shift: Optional[str] = None,
        equipment_id: Optional[str] = None,
        plant_zone: Optional[str] = None
    ) -> AnomalyDetectionResult:
        """
        Detect anomaly in embedding (preserves original interface)
        
        Args:
            embedding: Multimodal embedding
            shift: Filter by shift
            equipment_id: Filter by equipment ID
            plant_zone: Filter by plant zone
            
        Returns:
            AnomalyDetectionResult
        """
        try:
            # Step 1: Search and score
            search_result = await self.tools.search_and_score(
                embedding=embedding,
                shift=shift,
                equipment_id=equipment_id,
                plant_zone=plant_zone
            )
            
            if not search_result["success"]:
                logger.warning("Search and score failed")
                return AnomalyDetectionResult(
                    embedding=embedding,
                    is_anomaly=False,
                    anomaly_scores={},
                    per_modality_decisions={},
                    confidence=0.0,
                    requires_temporal_confirmation=False
                )
            
            anomaly_scores = search_result["anomaly_scores"]
            
            # Step 2: Apply thresholds
            threshold_result = self.tools.apply_thresholds(
                anomaly_scores=anomaly_scores,
                require_multi_modality=False
            )
            
            is_anomaly = threshold_result["is_anomaly"]
            per_modality = threshold_result["per_modality_decisions"]
            anomaly_count = threshold_result["anomaly_count"]
            
            # Step 3: Check for high severity (multi-modality confirmation)
            is_high_severity = anomaly_count >= self.tools.high_severity_min_modalities
            
            if is_high_severity:
                high_sev_result = self.tools.apply_thresholds(
                    anomaly_scores=anomaly_scores,
                    require_multi_modality=True
                )
                is_anomaly = high_sev_result["is_anomaly"]
                
                if is_anomaly:
                    self.tools.stats["high_severity_anomalies"] += 1
                    self.tools.stats["multi_modality_confirmations"] += 1
            
            # Step 4: Check temporal confirmation for borderline cases
            requires_temporal = self._is_borderline(anomaly_scores)
            temporal_confirmed = False
            temporal_count = 0
            
            if requires_temporal and is_anomaly:
                temporal_result = self.tools.check_temporal(per_modality)
                temporal_confirmed = temporal_result["is_confirmed"]
                temporal_count = temporal_result["max_consecutive"]
                
                if not temporal_confirmed:
                    is_anomaly = False
            
            # Step 5: Compute confidence
            confidence = self._compute_confidence(
                anomaly_scores=anomaly_scores,
                per_modality=per_modality,
                is_high_severity=is_high_severity,
                temporal_confirmed=temporal_confirmed
            )
            
            # Step 6: Store embedding
            await self.tools.store_embedding(
                embedding=embedding,
                is_anomaly=is_anomaly,
                anomaly_scores=anomaly_scores,
                confidence=confidence
            )
            
            return AnomalyDetectionResult(
                embedding=embedding,
                is_anomaly=is_anomaly,
                anomaly_scores=anomaly_scores,
                per_modality_decisions=per_modality,
                confidence=confidence,
                requires_temporal_confirmation=requires_temporal,
                temporal_confirmation_count=temporal_count
            )
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            raise
    
    def _is_borderline(self, anomaly_scores: Dict[str, float]) -> bool:
        """Check if scores are borderline (preserves original logic)"""
        thresholds = self.tools.threshold_manager.get_current_thresholds()
        
        for modality, score in anomaly_scores.items():
            if modality not in thresholds:
                continue
            
            threshold = thresholds[modality]
            lower = threshold * (1 - self.tools.borderline_threshold_pct)
            upper = threshold * (1 + self.tools.borderline_threshold_pct)
            
            if lower <= score <= upper:
                return True
        
        return False
    
    def _compute_confidence(
        self,
        anomaly_scores: Dict[str, float],
        per_modality: Dict[str, bool],
        is_high_severity: bool,
        temporal_confirmed: bool
    ) -> float:
        """Compute confidence score (preserves original logic)"""
        if not anomaly_scores:
            return 0.0
        
        thresholds = self.tools.threshold_manager.get_current_thresholds()
        
        modality_confidences = []
        for modality, score in anomaly_scores.items():
            if modality not in thresholds:
                continue
            
            threshold = thresholds[modality]
            if score > threshold:
                confidence = min((score - threshold) / threshold, 1.0)
                modality_confidences.append(confidence)
        
        if not modality_confidences:
            return 0.0
        
        base_confidence = sum(modality_confidences) / len(modality_confidences)
        modality_boost = min(sum(per_modality.values()) / len(per_modality), 1.0) * 0.2
        severity_boost = 0.1 if is_high_severity else 0.0
        temporal_boost = 0.1 if temporal_confirmed else 0.0
        
        return min(base_confidence + modality_boost + severity_boost + temporal_boost, 1.0)
    
    def create_task(
        self,
        description: str,
        expected_output: str,
        context: Optional[List[Task]] = None
    ) -> Task:
        """Create a CrewAI task for anomaly detection"""
        return Task(
            description=description,
            expected_output=expected_output,
            agent=self.agent,
            context=context or []
        )
    
    def get_crew(self, tasks: List[Task]) -> Crew:
        """Get CrewAI Crew with tasks"""
        return Crew(
            agents=[self.agent],
            tasks=tasks,
            verbose=True
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get crew statistics"""
        return {
            "tools": self.tools.get_stats(),
            "threshold_manager": self.tools.threshold_manager.get_stats(),
            "search_engine": self.tools.similarity_search.get_stats()
        }


if __name__ == "__main__":
    print("AnomalyDetectionCrew - CrewAI implementation ready")
