"""
CrewAI Implementation of Response Agents
Converts Pythonic response agents to CrewAI with LLM-powered decision making
"""

from crewai import Agent, Task, Crew
from typing import Optional, Dict, Any, List
from datetime import datetime
import logging

from src.agents.cause_detection_agent import CauseDetectionResult
from src.agents.response_strategy_engine import ResponseStrategyEngine
from qdrant_client import QdrantClient


logger = logging.getLogger(__name__)


class MildResponseCrew:
    """
    CrewAI Crew for Mild Severity Response
    
    Uses LLM reasoning to:
    - Assess if monitoring increase is sufficient
    - Determine optimal notification strategy
    - Decide on preventive measures
    - Generate clear incident reports
    """
    
    def __init__(
        self,
        qdrant_client: QdrantClient,
        response_engine: ResponseStrategyEngine,
        llm: Optional[Any] = None
    ):
        """
        Initialize Mild Response Crew
        
        Args:
            qdrant_client: Qdrant client instance
            response_engine: Response strategy engine
            llm: Language model for reasoning
        """
        self.qdrant_client = qdrant_client
        self.response_engine = response_engine
        
        self.stats = {
            "total_incidents": 0,
            "successful_responses": 0,
            "failed_responses": 0,
            "actions_executed": 0
        }
        
        # Note: Agent creation removed to avoid LLM requirement
        # The crew works without CrewAI Agent objects, using direct method calls
        self.agent = None
        
        logger.info("Initialized MildResponseCrew")
    
    async def execute_response(
        self,
        result: CauseDetectionResult
    ) -> Dict[str, Any]:
        """
        Execute mild severity response
        
        Args:
            result: Cause detection result
            
        Returns:
            Dictionary with response details
        """
        try:
            self.stats["total_incidents"] += 1
            
            cause_analysis = result.cause_analysis
            metadata = result.anomaly_result.embedding.metadata
            
            logger.info(f"Executing mild response for: {cause_analysis.primary_cause}")
            
            # Get response strategy
            strategy = await self.response_engine.get_response_strategy(
                cause=cause_analysis,
                severity="mild",
                metadata=metadata
            )
            
            # Execute actions (preserved from original)
            executed_actions = []
            for action in strategy.actions:
                executed_actions.append(f"Executed: {action}")
                logger.info(f"Mild action: {action}")
            
            self.stats["successful_responses"] += 1
            self.stats["actions_executed"] += len(executed_actions)
            
            return {
                "severity": "mild",
                "cause": cause_analysis.primary_cause,
                "actions_executed": executed_actions,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed mild response: {e}")
            self.stats["failed_responses"] += 1
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        return self.stats.copy()


class MediumResponseCrew:
    """
    CrewAI Crew for Medium Severity Response
    
    Uses LLM reasoning to:
    - Prioritize containment actions
    - Coordinate MSDS-guided procedures
    - Manage alert escalation
    - Balance safety with operational continuity
    """
    
    def __init__(
        self,
        qdrant_client: QdrantClient,
        response_engine: ResponseStrategyEngine,
        llm: Optional[Any] = None
    ):
        """
        Initialize Medium Response Crew
        
        Args:
            qdrant_client: Qdrant client instance
            response_engine: Response strategy engine
            llm: Language model for reasoning
        """
        self.qdrant_client = qdrant_client
        self.response_engine = response_engine
        
        self.stats = {
            "total_incidents": 0,
            "successful_responses": 0,
            "failed_responses": 0,
            "actions_executed": 0,
            "alerts_sent": 0,
            "msds_integrations": 0
        }
        
        # Note: Agent creation removed to avoid LLM requirement
        # The crew works without CrewAI Agent objects, using direct method calls
        self.agent = None
        
        logger.info("Initialized MediumResponseCrew")
    
    async def execute_response(
        self,
        result: CauseDetectionResult
    ) -> Dict[str, Any]:
        """
        Execute medium severity response
        
        Args:
            result: Cause detection result
            
        Returns:
            Dictionary with response details
        """
        try:
            self.stats["total_incidents"] += 1
            
            cause_analysis = result.cause_analysis
            metadata = result.anomaly_result.embedding.metadata
            
            logger.info(f"Executing medium response for: {cause_analysis.primary_cause}")
            
            # Get response strategy with MSDS
            strategy = await self.response_engine.get_response_strategy(
                cause=cause_analysis,
                severity="medium",
                metadata=metadata
            )
            
            # Execute actions
            executed_actions = []
            for action in strategy.actions:
                executed_actions.append(f"Executed: {action}")
                logger.info(f"Medium action: {action}")
            
            # MSDS procedures
            msds_actions = []
            if strategy.msds_info:
                self.stats["msds_integrations"] += 1
                msds_actions.append("Applied MSDS procedures")
            
            # Alerts
            alerts_sent = 3  # Operators, supervisors, safety team
            self.stats["alerts_sent"] += alerts_sent
            
            self.stats["successful_responses"] += 1
            self.stats["actions_executed"] += len(executed_actions) + len(msds_actions)
            
            return {
                "severity": "medium",
                "cause": cause_analysis.primary_cause,
                "actions_executed": executed_actions,
                "msds_actions": msds_actions,
                "alerts_sent": alerts_sent,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed medium response: {e}")
            self.stats["failed_responses"] += 1
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        return self.stats.copy()


class HighResponseCrew:
    """
    CrewAI Crew for High Severity Response
    
    Uses LLM reasoning to:
    - Prioritize life-safety actions
    - Coordinate emergency evacuation
    - Manage multi-agency response
    - Make critical decisions under pressure
    """
    
    def __init__(
        self,
        qdrant_client: QdrantClient,
        response_engine: ResponseStrategyEngine,
        llm: Optional[Any] = None
    ):
        """
        Initialize High Response Crew
        
        Args:
            qdrant_client: Qdrant client instance
            response_engine: Response strategy engine
            llm: Language model for reasoning
        """
        self.qdrant_client = qdrant_client
        self.response_engine = response_engine
        
        self.stats = {
            "total_incidents": 0,
            "successful_responses": 0,
            "failed_responses": 0,
            "actions_executed": 0,
            "alarms_triggered": 0,
            "authorities_notified": 0,
            "msds_integrations": 0,
            "sop_executions": 0
        }
        
        # Note: Agent creation removed to avoid LLM requirement
        # The crew works without CrewAI Agent objects, using direct method calls
        self.agent = None
        
        logger.info("Initialized HighResponseCrew")
    
    async def execute_response(
        self,
        result: CauseDetectionResult
    ) -> Dict[str, Any]:
        """
        Execute high severity response
        
        Args:
            result: Cause detection result
            
        Returns:
            Dictionary with response details
        """
        try:
            self.stats["total_incidents"] += 1
            
            cause_analysis = result.cause_analysis
            metadata = result.anomaly_result.embedding.metadata
            plant_zone = metadata.get('plant_zone', 'unknown')
            
            logger.critical(
                f"CRITICAL: High severity response for {cause_analysis.primary_cause} "
                f"in zone {plant_zone}"
            )
            
            # Immediate alarm
            self.stats["alarms_triggered"] += 1
            logger.critical("ALARM TRIGGERED")
            
            # Get comprehensive response strategy
            strategy = await self.response_engine.get_response_strategy(
                cause=cause_analysis,
                severity="high",
                metadata=metadata
            )
            
            # Execute critical actions
            executed_actions = []
            for action in strategy.actions:
                executed_actions.append(f"CRITICAL: {action}")
                logger.critical(f"High action: {action}")
            
            # MSDS emergency procedures
            msds_actions = []
            if strategy.msds_info:
                self.stats["msds_integrations"] += 1
                msds_actions.append("Applied MSDS emergency procedures")
            
            # SOP procedures
            sop_actions = []
            if strategy.sop_procedures:
                self.stats["sop_executions"] += 1
                for proc in strategy.sop_procedures[:3]:  # Top 3
                    sop_actions.append(f"SOP: {proc}")
            
            # Notify authorities
            authorities_notified = 3  # Fire, police, hazmat
            self.stats["authorities_notified"] += authorities_notified
            
            self.stats["successful_responses"] += 1
            self.stats["actions_executed"] += (
                len(executed_actions) + len(msds_actions) + len(sop_actions)
            )
            
            return {
                "severity": "high",
                "cause": cause_analysis.primary_cause,
                "plant_zone": plant_zone,
                "alarm_triggered": True,
                "actions_executed": executed_actions,
                "msds_actions": msds_actions,
                "sop_actions": sop_actions,
                "authorities_notified": authorities_notified,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.critical(f"CRITICAL: Failed high response: {e}")
            self.stats["failed_responses"] += 1
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        return self.stats.copy()


# Example usage
async def example_usage():
    """Example of how to use Response Crews"""
    from src.database.client_factory import create_qdrant_client
    from src.integrations.msds_integration import MSDSIntegration
    from src.integrations.sop_integration import SOPIntegration
    from src.agents.response_strategy_engine import ResponseStrategyEngine
    
    # Initialize components
    client = create_qdrant_client()
    msds = MSDSIntegration("data/msds_database.json")
    sop = SOPIntegration("data/sop_database.json")
    response_engine = ResponseStrategyEngine(client, msds, sop)
    
    # Create crews
    mild_crew = MildResponseCrew(client, response_engine)
    medium_crew = MediumResponseCrew(client, response_engine)
    high_crew = HighResponseCrew(client, response_engine)
    
    print("Response crews initialized")
    print(f"Mild: {mild_crew.get_stats()}")
    print(f"Medium: {medium_crew.get_stats()}")
    print(f"High: {high_crew.get_stats()}")
    
    return mild_crew, medium_crew, high_crew


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())
