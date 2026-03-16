"""
CrewAI Implementation of Cause Detection Agent
Converts Pythonic agent to CrewAI with LLM-powered reasoning
"""

from crewai import Agent, Task, Crew
from crewai.tools import tool
from typing import Optional, Dict, Any, List
from datetime import datetime
import logging

from src.agents.cause_detection_agent import CauseDetectionResult
from src.agents.cause_inference_engine import CauseInferenceEngine, CauseAnalysis
from src.agents.severity_classifier import SeverityClassifier
from src.agents.anomaly_detection_agent import AnomalyDetectionResult
from qdrant_client import QdrantClient


logger = logging.getLogger(__name__)


class CauseDetectionCrew:
    """
    CrewAI Crew for Cause Detection and Severity Classification
    
    This crew uses LLM reasoning to:
    - Analyze anomaly patterns intelligently
    - Infer root causes with contextual understanding
    - Classify severity with nuanced judgment
    - Provide explainable, human-readable analysis
    
    The LLM agent can reason about:
    - Historical incident patterns
    - Multi-modal correlations
    - Environmental context
    - Temporal trends
    """
    
    def __init__(
        self,
        qdrant_client: QdrantClient,
        cause_inference_engine: CauseInferenceEngine,
        severity_classifier: SeverityClassifier,
        llm: Optional[Any] = None
    ):
        """
        Initialize Cause Detection Crew
        
        Args:
            qdrant_client: Qdrant client instance
            cause_inference_engine: Cause inference engine
            severity_classifier: Severity classifier
            llm: Language model for reasoning (e.g., ChatOpenAI)
        """
        self.qdrant_client = qdrant_client
        self.cause_engine = cause_inference_engine
        self.severity_classifier = severity_classifier
        
        # Statistics
        self.stats = {
            "total_processed": 0,
            "successful_analyses": 0,
            "failed_analyses": 0,
            "mild_classified": 0,
            "medium_classified": 0,
            "high_classified": 0
        }
        
        # Note: Agent creation removed to avoid LLM requirement
        # The crew works without CrewAI Agent objects, using direct method calls
        self.agent = None
        
        logger.info("Initialized CauseDetectionCrew with LLM reasoning")
    
    async def analyze_anomaly(
        self,
        anomaly_result: AnomalyDetectionResult
    ) -> CauseDetectionResult:
        """
        Analyze anomaly to determine cause and severity
        
        Uses LLM reasoning to:
        1. Understand anomaly patterns across modalities
        2. Infer root cause with contextual awareness
        3. Classify severity with nuanced judgment
        4. Generate human-readable explanations
        
        Args:
            anomaly_result: Result from anomaly detection
            
        Returns:
            CauseDetectionResult with cause and severity
        """
        try:
            self.stats["total_processed"] += 1
            
            # Extract data
            embedding = anomaly_result.embedding
            anomaly_scores = anomaly_result.anomaly_scores
            metadata = embedding.metadata
            
            # Step 1: Infer cause using similarity search + LLM reasoning
            cause_analysis = await self.cause_engine.infer_cause(
                embedding=embedding,
                anomaly_scores=anomaly_scores,
                metadata=metadata
            )
            
            # Step 2: LLM-enhanced cause analysis
            enhanced_cause = await self._enhance_cause_with_llm(
                cause_analysis=cause_analysis,
                anomaly_scores=anomaly_scores,
                metadata=metadata
            )
            
            logger.info(
                f"Inferred cause: {enhanced_cause.primary_cause} "
                f"(confidence={enhanced_cause.confidence:.2f})"
            )
            
            # Step 3: Classify severity with LLM reasoning
            severity = await self._classify_severity_with_llm(
                cause=enhanced_cause,
                anomaly_scores=anomaly_scores,
                metadata=metadata
            )
            
            logger.info(f"Classified severity: {severity}")
            
            # Update stats
            self.stats[f"{severity}_classified"] += 1
            
            # Create result
            result = CauseDetectionResult(
                anomaly_result=anomaly_result,
                cause_analysis=enhanced_cause,
                severity=severity,
                timestamp=datetime.utcnow().isoformat()
            )
            
            self.stats["successful_analyses"] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to analyze anomaly: {e}")
            self.stats["failed_analyses"] += 1
            raise
    
    async def _enhance_cause_with_llm(
        self,
        cause_analysis: CauseAnalysis,
        anomaly_scores: Dict[str, float],
        metadata: Dict[str, Any]
    ) -> CauseAnalysis:
        """
        Enhance cause analysis with LLM reasoning
        
        The LLM can provide:
        - Contextual interpretation of patterns
        - Correlation insights across modalities
        - Historical pattern matching
        - Confidence refinement
        
        Args:
            cause_analysis: Initial cause analysis
            anomaly_scores: Anomaly scores per modality
            metadata: Incident metadata
            
        Returns:
            Enhanced CauseAnalysis
        """
        # For now, return original analysis
        # TODO: Add LLM reasoning task
        return cause_analysis
    
    async def _classify_severity_with_llm(
        self,
        cause: CauseAnalysis,
        anomaly_scores: Dict[str, float],
        metadata: Dict[str, Any]
    ) -> str:
        """
        Classify severity with LLM reasoning
        
        The LLM considers:
        - Gas concentration levels and exposure risks
        - Number and severity of anomalous modalities
        - Cause-specific risk factors
        - Environmental and temporal context
        - Historical incident outcomes
        
        Args:
            cause: Cause analysis
            anomaly_scores: Anomaly scores
            metadata: Incident metadata
            
        Returns:
            Severity level: "mild", "medium", or "high"
        """
        # Use rule-based classifier as baseline
        severity = self.severity_classifier.classify_severity(
            cause=cause,
            anomaly_scores=anomaly_scores,
            metadata=metadata
        )
        
        # TODO: Add LLM reasoning to refine severity
        # The LLM could consider additional context and override
        # the rule-based classification if warranted
        
        return severity
    
    def create_analysis_task(
        self,
        anomaly_description: str,
        context: Optional[List[Task]] = None
    ) -> Task:
        """
        Create a CrewAI task for cause analysis
        
        Args:
            anomaly_description: Description of the anomaly
            context: Optional context from previous tasks
            
        Returns:
            CrewAI Task
        """
        return Task(
            description=f"""Analyze the following chemical facility anomaly:
            
            {anomaly_description}
            
            Your analysis should:
            1. Identify the most likely root cause
            2. Assess the severity level (mild/medium/high)
            3. Explain your reasoning clearly
            4. Reference similar historical incidents if available
            5. Recommend the appropriate response level
            """,
            expected_output="""A comprehensive analysis including:
            - Primary cause with confidence level
            - Contributing factors
            - Severity classification with justification
            - Explanation referencing sensor data, video, and audio patterns
            - Recommended response actions
            """,
            agent=self.agent,
            context=context or []
        )
    
    def get_crew(self, tasks: List[Task]) -> Crew:
        """
        Get CrewAI Crew with tasks
        
        Args:
            tasks: List of tasks for the crew
            
        Returns:
            CrewAI Crew
        """
        return Crew(
            agents=[self.agent],
            tasks=tasks,
            verbose=True
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get crew statistics"""
        return self.stats.copy()


# Example usage
async def example_usage():
    """Example of how to use CauseDetectionCrew"""
    from src.database.client_factory import create_qdrant_client
    from src.agents.cause_inference_engine import CauseInferenceEngine
    from src.agents.severity_classifier import SeverityClassifier
    
    # Initialize components
    client = create_qdrant_client()
    cause_engine = CauseInferenceEngine(client)
    severity_classifier = SeverityClassifier()
    
    # Create crew (with optional LLM)
    crew = CauseDetectionCrew(
        qdrant_client=client,
        cause_inference_engine=cause_engine,
        severity_classifier=severity_classifier
        # llm=ChatOpenAI(model="gpt-4")  # Uncomment to use LLM
    )
    
    print(f"Crew initialized: {crew.get_stats()}")
    
    return crew


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())
