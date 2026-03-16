"""
Anomaly Detection on Test Data
Processes anomalous_1.mp4 and anomalous_sensor.csv to detect anomalies
"""

import asyncio
import cv2
import pandas as pd
import os
from datetime import datetime
from pathlib import Path

from src.database.client_factory import create_qdrant_client
from src.agents.input_collection_agent import InputCollectionAgent, EmbeddingGenerator
from src.agents.anomaly_detection_agent import AnomalyDetectionAgent
from src.agents.cause_detection_agent import CauseDetectionAgent
from src.agents.high_response_agent import HighResponseAgent
from src.agents.medium_response_agent import MediumResponseAgent
from src.agents.mild_response_agent import MildResponseAgent

from src.models.video_processor import VideoProcessor
from src.models.audio_processor import AudioProcessor
from src.models.sensor_processor import SensorProcessor
from src.models.sensor_adapter import SensorEmbeddingAdapter

from src.agents.similarity_search_engine import SimilaritySearchEngine
from src.agents.adaptive_threshold_manager import AdaptiveThresholdManager
from src.agents.storage_manager import StorageManager
from src.agents.cause_inference_engine import CauseInferenceEngine
from src.agents.severity_classifier import SeverityClassifier
from src.agents.response_strategy_engine import ResponseStrategyEngine

from src.integrations.msds_integration import MSDSIntegration
from src.integrations.sop_integration import SOPIntegration


class DetectionResults:
    def __init__(self):
        self.total_processed = 0
        self.anomalies_detected = 0
        self.normal_detected = 0
        self.anomaly_details = []


async def detect_anomalies():
    """Detect anomalies in test data"""
    print("\n" + "="*80)
    print("ANOMALY DETECTION ON TEST DATA")
    print("="*80)
    
    results = DetectionResults()
    
    # Check files
    video_path = Path("anomalous_1.mp4")
    sensor_path = Path("anomalous_sensor.csv")
    
    print(f"\n[1] Checking files...")
    if not video_path.exists():
        print(f"  [ERROR] Video file not found: {video_path}")
        return False
    print(f"  [OK] Video: {video_path} ({video_path.stat().st_size / 1024:.1f} KB)")
    
    if not sensor_path.exists():
        print(f"  [ERROR] Sensor file not found: {sensor_path}")
        return False
    print(f"  [OK] Sensor: {sensor_path}")
    
    # Load sensor data
    df_sensor = pd.read_csv(sensor_path)
    print(f"\n[2] Loaded {len(df_sensor)} sensor readings")
    print(f"    Columns: {list(df_sensor.columns)}")
    
    # Connect to Qdrant Cloud
    print(f"\n[3] Connecting to Qdrant Cloud...")
    try:
        qdrant_client = create_qdrant_client()
        
        # Verify baselines
        baselines_info = qdrant_client.get_collection("baselines")
        print(f"    [OK] Connected - {baselines_info.points_count} baseline points available")
    except Exception as e:
        print(f"    [ERROR] Failed to connect: {e}")
        return False
    
    # Initialize processors
    print(f"\n[4] Initializing processors...")
    video_proc = VideoProcessor(device="cpu", timeout=2.0)
    panns_checkpoint = os.getenv("PANNS_CHECKPOINT_PATH")
    audio_proc = AudioProcessor(device="cpu", timeout=2.0, checkpoint_path=panns_checkpoint)
    adapter = SensorEmbeddingAdapter()
    sensor_proc = SensorProcessor(adapter=adapter)
    print(f"    [OK] Video: {video_proc.model_name}")
    print(f"    [OK] Audio: {audio_proc.model_name}")
    print(f"    [OK] Sensor: initialized")
    
    # Initialize agents
    print(f"\n[5] Initializing detection agents...")
    embedding_gen = EmbeddingGenerator(
        video_processor=video_proc,
        audio_processor=audio_proc,
        sensor_processor=sensor_proc
    )
    
    input_agent = InputCollectionAgent(embedding_gen, 1.0, 100)
    
    similarity_search = SimilaritySearchEngine(qdrant_client)
    threshold_manager = AdaptiveThresholdManager(
        video_threshold=0.7,
        audio_threshold=0.65,
        sensor_threshold=2.5
    )
    storage_manager = StorageManager(qdrant_client)
    
    anomaly_agent = AnomalyDetectionAgent(
        qdrant_client=qdrant_client,
        similarity_search_engine=similarity_search,
        adaptive_threshold_manager=threshold_manager,
        storage_manager=storage_manager,
        processing_interval=1.0
    )
    
    cause_engine = CauseInferenceEngine(qdrant_client)
    severity_classifier = SeverityClassifier()
    
    msds_path = os.getenv("MSDS_DATABASE_PATH", "data/msds_database.json")
    sop_path = os.getenv("SOP_DATABASE_PATH", "data/sop_database.json")
    
    msds_integration = MSDSIntegration(msds_path)
    sop_integration = SOPIntegration(sop_path)
    response_engine = ResponseStrategyEngine(
        qdrant_client=qdrant_client,
        msds_integration=msds_integration,
        sop_integration=sop_integration
    )
    
    mild_agent = MildResponseAgent(qdrant_client, response_engine, 1.0)
    medium_agent = MediumResponseAgent(qdrant_client, response_engine, 1.0)
    high_agent = HighResponseAgent(qdrant_client, response_engine, 1.0)
    
    cause_agent = CauseDetectionAgent(
        qdrant_client=qdrant_client,
        cause_inference_engine=cause_engine,
        severity_classifier=severity_classifier,
        processing_interval=1.0
    )
    print(f"    [OK] All agents initialized")
    
    # Process video frames with sensor data
    print(f"\n" + "="*80)
    print("PROCESSING VIDEO FRAMES")
    print("="*80)
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("[ERROR] Failed to open video file")
        return False
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\nVideo info: {frame_count} frames @ {fps:.2f} FPS")
    
    # Process first 5 frames
    frames_to_process = min(5, frame_count)
    print(f"Processing first {frames_to_process} frames...\n")
    
    for i in range(frames_to_process):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get corresponding sensor reading
        sensor_idx = min(i, len(df_sensor) - 1)
        sensor_reading = {
            "timestamp": datetime.now(),
            "temperature_celsius": float(df_sensor.iloc[sensor_idx]["temperature_celsius"]),
            "pressure_bar": float(df_sensor.iloc[sensor_idx]["pressure_bar"]),
            "gas_concentration_ppm": float(df_sensor.iloc[sensor_idx]["gas_concentration_ppm"]),
            "vibration_mm_s": float(df_sensor.iloc[sensor_idx]["vibration_mm_s"]),
            "flow_rate_lpm": float(df_sensor.iloc[sensor_idx]["flow_rate_lpm"])
        }
        
        metadata = {
            "plant_zone": "Zone_A",
            "shift": "morning",
            "camera_id": "anomalous_cam",
            "frame_number": i
        }
        
        try:
            # Generate embedding
            embedding = await input_agent.process_data_point(
                video_frame=frame,
                sensor_reading=sensor_reading,
                metadata=metadata
            )
            
            if embedding and embedding.has_any_modality():
                results.total_processed += 1
                modalities = embedding.get_available_modalities()
                
                # Detect anomaly
                anomaly_result = await anomaly_agent.detect_anomaly(embedding)
                
                if anomaly_result.is_anomaly:
                    results.anomalies_detected += 1
                    print(f"Frame {i+1}: [ANOMALY DETECTED]")
                    print(f"  Modalities: {', '.join(modalities)}")
                    print(f"  Sensor: T={sensor_reading['temperature_celsius']:.1f}C, "
                          f"P={sensor_reading['pressure_bar']:.1f}bar, "
                          f"Gas={sensor_reading['gas_concentration_ppm']:.1f}ppm")
                    print(f"  Anomaly scores: {anomaly_result.anomaly_scores}")
                    print(f"  Confidence: {anomaly_result.confidence:.3f}")
                    
                    # Analyze cause
                    cause_result = await cause_agent.analyze_anomaly(anomaly_result)
                    print(f"  Cause: {cause_result.cause_analysis.primary_cause}")
                    print(f"  Severity: {cause_result.severity}")
                    print(f"  Cause confidence: {cause_result.cause_analysis.confidence:.3f}")
                    
                    # Execute response
                    if cause_result.severity == "mild":
                        response = await mild_agent.execute_response(cause_result)
                    elif cause_result.severity == "medium":
                        response = await medium_agent.execute_response(cause_result)
                    else:
                        response = await high_agent.execute_response(cause_result)
                    
                    print(f"  Response: {len(response.get('actions_taken', []))} actions executed")
                    
                    results.anomaly_details.append({
                        "frame": i+1,
                        "cause": cause_result.cause_analysis.primary_cause,
                        "severity": cause_result.severity,
                        "confidence": anomaly_result.confidence
                    })
                else:
                    results.normal_detected += 1
                    print(f"Frame {i+1}: [NORMAL] - No anomaly detected")
                
                print()  # Blank line
                
        except Exception as e:
            print(f"Frame {i+1}: [ERROR] {e}")
            print()
    
    cap.release()
    
    # Final Summary
    print("="*80)
    print("DETECTION SUMMARY")
    print("="*80)
    print(f"\nTotal data points processed: {results.total_processed}")
    print(f"Anomalies detected: {results.anomalies_detected}")
    print(f"Normal operation: {results.normal_detected}")
    print(f"Detection rate: {results.anomalies_detected/results.total_processed*100:.1f}%")
    
    if results.anomaly_details:
        print(f"\nDetected Anomalies:")
        for detail in results.anomaly_details:
            print(f"  Frame {detail['frame']}: {detail['cause']} "
                  f"(severity: {detail['severity']}, confidence: {detail['confidence']:.3f})")
    
    print(f"\n" + "="*80)
    if results.anomalies_detected > 0:
        print("SUCCESS: System detected anomalies in test data!")
    else:
        print("WARNING: No anomalies detected in test data")
    print("="*80)
    
    qdrant_client.close()
    return results.anomalies_detected > 0


if __name__ == "__main__":
    print("\nStarting anomaly detection task...")
    success = asyncio.run(detect_anomalies())
    print(f"\nTask completed: {'SUCCESS' if success else 'FAILED'}")
    exit(0 if success else 1)
