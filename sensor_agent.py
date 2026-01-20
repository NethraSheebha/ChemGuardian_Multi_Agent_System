from dotenv import load_dotenv
load_dotenv()

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from crewai import Agent, Task, Crew, LLM
from qdrant_client import QdrantClient

from sensor_pipeline import SensorPipeline
from utils.sensor_stream_simulator import SensorStreamSimulator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/sensor_agent.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class SensorIntelligenceAgent:
    """CrewAI-based agent for intelligent sensor monitoring"""
    
    def __init__(
        self,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        collection_name: str = "sensor_patterns",
        window_size: int = 50,
        anomaly_threshold: float = -0.5
    ):
        """
        Initialize Sensor Intelligence Agent
        
        Args:
            qdrant_host: Qdrant server host
            qdrant_port: Qdrant server port
            collection_name: Target Qdrant collection
            window_size: Number of readings per analysis window
            anomaly_threshold: Anomaly score threshold for alerts
        """
        self.qdrant = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.collection_name = collection_name
        self.window_size = window_size
        self.anomaly_threshold = anomaly_threshold
        
        # Initialize sensor processing pipeline
        self.pipeline = SensorPipeline(
            window_size=window_size,
            anomaly_threshold=anomaly_threshold
        )
        
        # Create models directory
        Path("models").mkdir(exist_ok=True)
        
        # Initialize CrewAI agent
        self._setup_agent()
        
        logger.info("Sensor Intelligence Agent initialized")

    def _setup_agent(self):
        """Configure CrewAI agent and tasks"""
        
        api_key = os.getenv("OPENROUTER_API_KEY")

        llm = LLM(
            model="openrouter/openai/gpt-4o-mini",
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )

        # Define the sensor monitoring agent
        self.agent = Agent(
            role="Sensor Intelligence Analyst",
            goal="Monitor real-time sensor data streams to detect anomalous patterns "
                 "that may indicate chemical leaks or equipment failures",
            backstory="You are an expert in time-series analysis and industrial "
                     "sensor monitoring. You analyze gas concentration, pressure, "
                     "temperature, and humidity patterns to identify early warning "
                     "signs of chemical incidents.",
            llm=llm,
            verbose=True,
            allow_delegation=False
        )
        
        # Define monitoring task
        self.monitoring_task = Task(
            description="Continuously analyze sensor data streams, extract time-series "
                       "features, detect anomalies using machine learning, and store "
                       "anomalous patterns in the knowledge base for incident correlation.",
            agent=self.agent,
            expected_output="JSON alerts for detected anomalies with severity assessment"
        )
    
    def _assess_severity(self, metrics: Dict) -> str:
        """
        Assess severity level based on sensor metrics
        
        Args:
            metrics: Dictionary of sensor metrics
            
        Returns:
            Severity level string
        """
        gas_ppm = metrics.get("gas_ppm_mean", 0)
        
        if gas_ppm > 800:
            return "CRITICAL"
        elif gas_ppm > 400:
            return "HIGH"
        elif gas_ppm > 150:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _create_alert(
        self,
        sensor_id: str,
        zone: str,
        timestamp: str,
        anomaly_score: float,
        metrics: Dict
    ) -> Dict:
        """
        Create alert JSON for anomaly
        
        Args:
            sensor_id: Sensor identifier
            zone: Factory zone
            timestamp: Event timestamp
            anomaly_score: Anomaly detection score
            metrics: Sensor metrics dictionary
            
        Returns:
            Alert dictionary
        """
        severity = self._assess_severity(metrics)
        
        alert = {
            "event": "sensor_anomaly",
            "sensor_id": sensor_id,
            "zone": zone,
            "timestamp": timestamp,
            "score": round(anomaly_score, 4),
            "severity": severity,
            "metrics": {
                "gas_ppm_mean": round(metrics.get("gas_ppm_mean", 0), 2),
                "pressure_mean": round(metrics.get("pressure_mean", 0), 2),
                "temp_mean": round(metrics.get("temp_mean", 0), 2),
                "humidity_mean": round(metrics.get("humidity_mean", 0), 2)
            }
        }
        
        return alert
    
    def _store_anomaly(
        self,
        sensor_id: str,
        zone: str,
        timestamp: str,
        embedding: List[float],
        anomaly_score: float,
        metrics: Dict
    ):
        """
        Store anomaly pattern in Qdrant
        
        Args:
            sensor_id: Sensor identifier
            zone: Factory zone
            timestamp: Event timestamp
            embedding: 128-dim feature embedding
            anomaly_score: Anomaly score
            metrics: Sensor metrics
        """
        from qdrant_client.models import PointStruct
        import hashlib
        import time
        
        # Generate unique ID
        point_id = hashlib.md5(
            f"{sensor_id}_{timestamp}_{anomaly_score}".encode()
        ).hexdigest()
        
        point = PointStruct(
            id=point_id,
            vector=embedding,
            payload={
                "sensor_id": sensor_id,
                "zone": zone,
                "timestamp": timestamp,
                "anomaly_score": anomaly_score,
                "gas_ppm_mean": metrics.get("gas_ppm_mean", 0),
                "pressure_mean": metrics.get("pressure_mean", 0),
                "temp_mean": metrics.get("temp_mean", 0),
                "humidity_mean": metrics.get("humidity_mean", 0),
                "severity": self._assess_severity(metrics),
                "ingestion_time": time.time(),
                "source": "sensor_agent"
            }
        )
        
        self.qdrant.upsert(
            collection_name=self.collection_name,
            points=[point]
        )
        
        logger.info(f"Stored anomaly pattern: {sensor_id} in {zone}")
    
    def process_window(
        self,
        window_data: List[Dict],
        sensor_id: str,
        zone: str
    ) -> Optional[Dict]:
        """
        Process a sensor data window
        
        Args:
            window_data: List of sensor readings
            sensor_id: Sensor identifier
            zone: Factory zone
            
        Returns:
            Alert dictionary if anomaly detected, None otherwise
        """
        try:
            # Analyze window
            result = self.pipeline.analyze_window(window_data, sensor_id)
            
            if result is None:
                return None
            
            is_anomaly, score, embedding, metrics = result
            
            if is_anomaly and score <= self.anomaly_threshold:
                # Get timestamp from latest reading
                timestamp = window_data[-1]["timestamp"]
                
                # Store in Qdrant
                self._store_anomaly(
                    sensor_id=sensor_id,
                    zone=zone,
                    timestamp=timestamp,
                    embedding=embedding,
                    anomaly_score=score,
                    metrics=metrics
                )
                
                # Create alert
                alert = self._create_alert(
                    sensor_id=sensor_id,
                    zone=zone,
                    timestamp=timestamp,
                    anomaly_score=score,
                    metrics=metrics
                )
                
                return alert
            
            return None
            
        except Exception as e:
            logger.error(f"Error processing window for {sensor_id}: {e}")
            return None
    
    def run_from_csv(
        self,
        csv_path: str = "data/sensors/sensor_stream.csv",
        output_alerts: str = "outputs/sensor_alerts.jsonl"
    ):
        """
        Run agent on CSV sensor data (simulated real-time)
        
        Args:
            csv_path: Path to sensor CSV file
            output_alerts: Path to output alerts file
        """
        logger.info("="*60)
        logger.info("SENSOR INTELLIGENCE AGENT - CSV MODE")
        logger.info("="*60)
        
        # Create output directory
        Path(output_alerts).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize simulator
        simulator = SensorStreamSimulator(csv_path)
        
        # Load data
        if not simulator.load_data():
            logger.error("Failed to load sensor data")
            return
        
        # Training phase
        logger.info("\nPhase 1: Training anomaly detection model...")
        training_windows = simulator.get_training_windows(
            num_windows=10,
            window_size=self.window_size
        )
        
        if training_windows:
            self.pipeline.train_models(training_windows)
            logger.info("Models trained successfully")
        else:
            logger.error("Failed to generate training data")
            return
        
        # Monitoring phase
        logger.info("\nPhase 2: Real-time monitoring...")
        
        alert_count = 0
        total_windows = 0
        
        with open(output_alerts, 'w') as f:
            for window_batch in simulator.stream_windows(
                window_size=self.window_size,
                batch_size=5
            ):
                for sensor_id, zone, window_data in window_batch:
                    total_windows += 1
                    
                    # Process window
                    alert = self.process_window(window_data, sensor_id, zone)
                    
                    if alert:
                        alert_count += 1
                        
                        # Write alert to file
                        f.write(json.dumps(alert) + "\n")
                        f.flush()
                        
                        # Log alert
                        logger.warning(
                            f"🚨 ANOMALY DETECTED - {alert['severity']} - "
                            f"{sensor_id} in {zone} - "
                            f"Gas: {alert['metrics']['gas_ppm_mean']} ppm"
                        )
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("MONITORING SUMMARY")
        logger.info("="*60)
        logger.info(f"Total windows analyzed: {total_windows}")
        logger.info(f"Anomalies detected: {alert_count}")
        logger.info(f"Alerts saved to: {output_alerts}")
        logger.info("="*60)
    
    def run_from_mqtt(
        self,
        mqtt_broker: str = "localhost",
        mqtt_port: int = 1883,
        topic: str = "factory/sensors/#"
    ):
        """
        Run agent on MQTT sensor stream (production mode)
        
        Args:
            mqtt_broker: MQTT broker host
            mqtt_port: MQTT broker port
            topic: MQTT topic pattern
        """
        logger.info("="*60)
        logger.info("SENSOR INTELLIGENCE AGENT - MQTT MODE")
        logger.info("="*60)
        logger.info(f"Connecting to {mqtt_broker}:{mqtt_port}")
        logger.info(f"Subscribing to topic: {topic}")
        
        # This would be implemented with paho-mqtt in production
        # For now, this is a structural stub
        
        logger.warning("MQTT mode not yet implemented - use CSV mode for testing")
        logger.info("To implement: use paho.mqtt.client to subscribe and buffer")


def main():
    """Main execution"""
    try:
        # Initialize agent
        agent = SensorIntelligenceAgent(
            qdrant_host="localhost",
            qdrant_port=6333,
            window_size=50,
            anomaly_threshold=-0.5
        )
        
        # Run in CSV mode (simulated real-time)
        agent.run_from_csv(
            csv_path="data/sensors/sensor_stream.csv",
            output_alerts="outputs/sensor_alerts.jsonl"
        )
        
        logger.info("\nSensor monitoring completed successfully!")
        
    except Exception as e:
        logger.error(f"Sensor agent failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()