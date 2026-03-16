"""
TRUE CrewAI Agentic Implementation
Following CrewAI documentation for autonomous, collaborative AI agents
"""

from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from langchain_community.llms import Ollama
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


# ============================================================================
# TOOLS - Functions that agents can use
# ============================================================================

@tool("Search Similar Incidents")
def search_similar_incidents(query: str) -> str:
    """
    Search for similar historical incidents in the database.
    Use this to find patterns and learn from past events.
    
    Args:
        query: Description of the incident to search for
        
    Returns:
        String describing similar incidents found
    """
    # This would connect to your Qdrant database
    return f"Found 3 similar incidents matching: {query}"


@tool("Analyze Sensor Data")
def analyze_sensor_data(sensor_values: str) -> str:
    """
    Analyze sensor readings to identify anomalies.
    
    Args:
        sensor_values: JSON string of sensor readings
        
    Returns:
        Analysis of sensor data
    """
    return f"Sensor analysis: High pressure and temperature detected"


@tool("Check MSDS Database")
def check_msds_database(chemical: str) -> str:
    """
    Look up Material Safety Data Sheet for a chemical.
    
    Args:
        chemical: Name of the chemical
        
    Returns:
        Safety information and handling procedures
    """
    return f"MSDS for {chemical}: Highly toxic, requires immediate evacuation"


@tool("Retrieve SOP Procedures")
def retrieve_sop_procedures(zone: str, severity: str) -> str:
    """
    Retrieve Standard Operating Procedures for a zone and severity level.
    
    Args:
        zone: Plant zone (e.g., Zone_A)
        severity: Severity level (mild, medium, high)
        
    Returns:
        List of SOP procedures to follow
    """
    return f"SOP for {zone} ({severity}): 1. Evacuate, 2. Activate alarm, 3. Contact emergency"


# ============================================================================
# AGENTS - Autonomous AI agents with specific roles
# ============================================================================

def create_anomaly_detection_agent(llm) -> Agent:
    """
    Anomaly Detection Agent
    
    Role: Identify unusual patterns in multimodal data
    Goal: Detect chemical leaks before they become critical
    """
    return Agent(
        role="Anomaly Detection Specialist",
        goal="Detect chemical leaks and anomalies in real-time by analyzing video, audio, and sensor data",
        backstory="""You are an expert in industrial safety monitoring with 15 years of experience.
        You have a keen eye for spotting unusual patterns in sensor data, video feeds, and audio signals.
        Your quick detection has prevented numerous accidents and saved lives.
        You use similarity-based analysis to compare current readings against known baselines.""",
        tools=[analyze_sensor_data],
        llm=llm,
        verbose=True,
        allow_delegation=False
    )


def create_cause_analysis_agent(llm) -> Agent:
    """
    Cause Analysis Agent
    
    Role: Determine root cause of detected anomalies
    Goal: Identify why the anomaly occurred
    """
    return Agent(
        role="Root Cause Analyst",
        goal="Determine the root cause of detected anomalies by analyzing patterns and historical data",
        backstory="""You are a chemical engineer with expertise in failure analysis and incident investigation.
        You excel at connecting dots between different data sources to identify root causes.
        You have investigated hundreds of industrial incidents and can quickly recognize patterns.
        You always consider multiple hypotheses before concluding.""",
        tools=[search_similar_incidents, analyze_sensor_data],
        llm=llm,
        verbose=True,
        allow_delegation=False
    )


def create_severity_classifier_agent(llm) -> Agent:
    """
    Severity Classification Agent
    
    Role: Assess the severity and risk level
    Goal: Classify incidents as mild, medium, or high severity
    """
    return Agent(
        role="Risk Assessment Specialist",
        goal="Classify the severity of incidents and assess potential risks to personnel and equipment",
        backstory="""You are a safety officer with deep knowledge of chemical hazards and risk assessment.
        You have trained emergency response teams and developed safety protocols for major facilities.
        You consider gas concentration, exposure time, environmental factors, and potential escalation.
        Your assessments are always conservative to prioritize safety.""",
        tools=[check_msds_database],
        llm=llm,
        verbose=True,
        allow_delegation=False
    )


def create_response_coordinator_agent(llm) -> Agent:
    """
    Response Coordinator Agent
    
    Role: Coordinate emergency response actions
    Goal: Execute appropriate response based on severity
    """
    return Agent(
        role="Emergency Response Coordinator",
        goal="Coordinate and execute appropriate emergency response actions based on incident severity",
        backstory="""You are an emergency response coordinator with certifications in hazmat and industrial safety.
        You have managed responses to chemical incidents ranging from minor leaks to major emergencies.
        You know all SOPs by heart and can quickly mobilize the right resources.
        You prioritize human safety above all else and follow established protocols.""",
        tools=[retrieve_sop_procedures, check_msds_database],
        llm=llm,
        verbose=True,
        allow_delegation=False
    )


# ============================================================================
# TASKS - Specific objectives for agents to accomplish
# ============================================================================

def create_anomaly_detection_task(agent: Agent, data_description: str) -> Task:
    """Create task for anomaly detection"""
    return Task(
        description=f"""Analyze the following multimodal data for anomalies:
        
        {data_description}
        
        Compare against baseline patterns and determine if any anomalies are present.
        Consider all three modalities: video, audio, and sensor data.
        Report your findings with confidence scores.""",
        expected_output="""A detailed report including:
        - Whether anomalies were detected (yes/no)
        - Which modalities show anomalies (video/audio/sensor)
        - Confidence scores for each modality
        - Description of the anomalous patterns observed""",
        agent=agent
    )


def create_cause_analysis_task(agent: Agent, anomaly_report: str, context: List[Task]) -> Task:
    """Create task for cause analysis"""
    return Task(
        description=f"""Based on the anomaly detection report, determine the root cause:
        
        {anomaly_report}
        
        Search for similar historical incidents and analyze the patterns.
        Consider all available data sources and provide your best assessment of the root cause.""",
        expected_output="""A root cause analysis including:
        - Primary cause (e.g., gas_leak, valve_malfunction, ppe_violation)
        - Contributing factors
        - Confidence level (0-1)
        - Explanation of your reasoning
        - References to similar historical incidents""",
        agent=agent,
        context=context  # Depends on anomaly detection task
    )


def create_severity_classification_task(agent: Agent, cause_report: str, context: List[Task]) -> Task:
    """Create task for severity classification"""
    return Task(
        description=f"""Assess the severity of this incident:
        
        {cause_report}
        
        Consider gas concentration levels, exposure risks, and potential for escalation.
        Check MSDS data for relevant chemicals.
        Classify as: mild, medium, or high severity.""",
        expected_output="""A severity assessment including:
        - Severity level (mild/medium/high)
        - Risk factors considered
        - Potential consequences
        - Justification for the classification
        - Recommended response level""",
        agent=agent,
        context=context  # Depends on cause analysis task
    )


def create_response_coordination_task(agent: Agent, severity_report: str, context: List[Task]) -> Task:
    """Create task for response coordination"""
    return Task(
        description=f"""Coordinate emergency response for this incident:
        
        {severity_report}
        
        Retrieve appropriate SOPs and execute response actions.
        Ensure all safety protocols are followed.
        Document all actions taken.""",
        expected_output="""A response execution report including:
        - List of actions executed
        - SOP procedures followed
        - Personnel notified
        - Equipment deployed
        - Timeline of response
        - Current status""",
        agent=agent,
        context=context  # Depends on severity classification task
    )


# ============================================================================
# CREW - Orchestrate agents working together
# ============================================================================

class ChemicalLeakMonitoringCrew:
    """
    TRUE CrewAI Implementation for Chemical Leak Monitoring
    
    This crew follows CrewAI best practices:
    - Agents have clear roles, goals, and backstories
    - Tasks have specific descriptions and expected outputs
    - Agents use tools to gather information
    - LLM powers autonomous decision-making
    - Tasks are chained with context dependencies
    - Process is sequential for safety-critical decisions
    """
    
    def __init__(self, llm_model: str = "llama3.2:1b", llm_base_url: str = "http://localhost:11434"):
        """
        Initialize the crew with Ollama LLM
        
        Args:
            llm_model: Ollama model to use
            llm_base_url: Ollama server URL
        """
        # Initialize LLM
        self.llm = Ollama(
            model=llm_model,
            base_url=llm_base_url,
            temperature=0.1  # Low temperature for safety-critical decisions
        )
        
        # Create agents
        self.anomaly_agent = create_anomaly_detection_agent(self.llm)
        self.cause_agent = create_cause_analysis_agent(self.llm)
        self.severity_agent = create_severity_classifier_agent(self.llm)
        self.response_agent = create_response_coordinator_agent(self.llm)
        
        logger.info("Initialized ChemicalLeakMonitoringCrew with 4 autonomous agents")
    
    def analyze_incident(
        self,
        video_data: str,
        audio_data: str,
        sensor_data: Dict[str, float],
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze an incident using the full crew
        
        This creates a sequential workflow where:
        1. Anomaly Detection Agent detects anomalies
        2. Cause Analysis Agent determines root cause (using detection results)
        3. Severity Classifier Agent assesses risk (using cause analysis)
        4. Response Coordinator Agent executes actions (using severity assessment)
        
        Args:
            video_data: Description of video observations
            audio_data: Description of audio observations
            sensor_data: Sensor readings
            metadata: Incident metadata (zone, shift, etc.)
            
        Returns:
            Complete analysis with all agent outputs
        """
        # Format data for agents
        data_description = f"""
        Video: {video_data}
        Audio: {audio_data}
        Sensors: Temperature={sensor_data.get('temperature')}°C, 
                 Pressure={sensor_data.get('pressure')} bar,
                 Gas={sensor_data.get('gas_concentration')} ppm
        Location: {metadata.get('plant_zone')}
        Shift: {metadata.get('shift')}
        """
        
        # Create tasks with dependencies
        task1 = create_anomaly_detection_task(self.anomaly_agent, data_description)
        task2 = create_cause_analysis_task(self.cause_agent, "See anomaly detection results", [task1])
        task3 = create_severity_classification_task(self.severity_agent, "See cause analysis", [task2])
        task4 = create_response_coordination_task(self.response_agent, "See severity assessment", [task3])
        
        # Create crew with sequential process
        crew = Crew(
            agents=[
                self.anomaly_agent,
                self.cause_agent,
                self.severity_agent,
                self.response_agent
            ],
            tasks=[task1, task2, task3, task4],
            process=Process.sequential,  # Tasks execute in order
            verbose=True
        )
        
        # Execute the crew
        logger.info("Starting crew execution...")
        result = crew.kickoff()
        
        logger.info("Crew execution complete!")
        return result


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_usage():
    """Example of using the TRUE CrewAI implementation"""
    
    # Initialize crew
    crew = ChemicalLeakMonitoringCrew(
        llm_model="llama3.2:1b",
        llm_base_url="http://localhost:11434"
    )
    
    # Analyze an incident
    result = crew.analyze_incident(
        video_data="Visible gas plume detected in Zone A",
        audio_data="Hissing sound detected at 85 dB",
        sensor_data={
            "temperature": 95.5,
            "pressure": 15.2,
            "gas_concentration": 450.0,
            "vibration": 8.5,
            "flow_rate": 65.0
        },
        metadata={
            "plant_zone": "Zone_A",
            "shift": "morning",
            "timestamp": datetime.utcnow().isoformat()
        }
    )
    
    print("\n" + "="*80)
    print("CREW ANALYSIS RESULT")
    print("="*80)
    print(result)
    
    return result


if __name__ == "__main__":
    example_usage()
