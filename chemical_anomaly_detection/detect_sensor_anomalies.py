"""
Simple Anomaly Detection - Sensor Data Only
Processes anomalous_sensor.csv to detect anomalies using Qdrant Cloud baselines
"""

import asyncio
import pandas as pd
import os
from datetime import datetime

from src.database.client_factory import create_qdrant_client
from src.agents.input_collection_agent import InputCollectionAgent, EmbeddingGenerator
from src.agents.anomaly_detection_agent import AnomalyDetectionAgent
from src.models.sensor_processor import SensorProcessor
from src.models.sensor_adapter import SensorEmbeddingAdapter
from src.agents.similarity_search_engine import SimilaritySearchEngine
from src.agents.adaptive_threshold_manager import AdaptiveThresholdManager
from src.agents.storage_manager import StorageManager


async def detect_sensor_anomalies():
    """Detect anomalies in sensor data"""
    print("\n" + "="*80)
    print("SENSOR ANOMALY DETECTION")
    print("="*80)
    
    # Load sensor data
    print("\n[1] Loading sensor data...")
    df_sensor = pd.read_csv("anomalous_sensor.csv")
    print(f"    Loaded {len(df_sensor)} sensor readings")
    print(f"    Columns: {list(df_sensor.columns)}")
    
    # Show sample data
    print(f"\n    Sample values:")
    print(f"      Temperature: {df_sensor['temperature_celsius'].mean():.1f}C (range: {df_sensor['temperature_celsius'].min():.1f}-{df_sensor['temperature_celsius'].max():.1f})")
    print(f"      Pressure: {df_sensor['pressure_bar'].mean():.1f}bar (range: {df_sensor['pressure_bar'].min():.1f}-{df_sensor['pressure_bar'].max():.1f})")
    print(f"      Gas: {df_sensor['gas_concentration_ppm'].mean():.1f}ppm (range: {df_sensor['gas_concentration_ppm'].min():.1f}-{df_sensor['gas_concentration_ppm'].max():.1f})")
    
    # Connect to Qdrant Cloud
    print(f"\n[2] Connecting to Qdrant Cloud...")
    qdrant_client = create_qdrant_client()
    
    baselines_info = qdrant_client.get_collection("baselines")
    print(f"    [OK] Connected - {baselines_info.points_count} baseline points available")
    
    # Initialize components
    print(f"\n[3] Initializing components...")
    adapter = SensorEmbeddingAdapter()
    sensor_proc = SensorProcessor(adapter=adapter)
    
    embedding_gen = EmbeddingGenerator(
        video_processor=None,
        audio_processor=None,
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
    print(f"    [OK] Components initialized")
    
    # Process sensor readings
    print(f"\n[4] Processing 10 sensor readings...")
    print("="*80)
    
    anomalies_detected = 0
    normal_detected = 0
    anomaly_scores_list = []
    
    for i in range(10):
        sensor_reading = {
            "timestamp": datetime.now(),
            "temperature_celsius": float(df_sensor.iloc[i]["temperature_celsius"]),
            "pressure_bar": float(df_sensor.iloc[i]["pressure_bar"]),
            "gas_concentration_ppm": float(df_sensor.iloc[i]["gas_concentration_ppm"]),
            "vibration_mm_s": float(df_sensor.iloc[i]["vibration_mm_s"]),
            "flow_rate_lpm": float(df_sensor.iloc[i]["flow_rate_lpm"])
        }
        
        metadata = {
            "plant_zone": "Zone_A",
            "shift": "morning",
            "equipment_id": f"sensor_{i+1:02d}"
        }
        
        # Generate embedding
        embedding = await input_agent.process_data_point(
            sensor_reading=sensor_reading,
            metadata=metadata
        )
        
        if embedding:
            # Detect anomaly
            result = await anomaly_agent.detect_anomaly(embedding)
            
            sensor_score = result.anomaly_scores.get('sensor', 0)
            anomaly_scores_list.append(sensor_score)
            
            if result.is_anomaly:
                anomalies_detected += 1
                print(f"\nReading {i+1}: [ANOMALY DETECTED]")
                print(f"  Values: T={sensor_reading['temperature_celsius']:.1f}C, "
                      f"P={sensor_reading['pressure_bar']:.1f}bar, "
                      f"Gas={sensor_reading['gas_concentration_ppm']:.1f}ppm, "
                      f"Vib={sensor_reading['vibration_mm_s']:.1f}mm/s, "
                      f"Flow={sensor_reading['flow_rate_lpm']:.1f}lpm")
                print(f"  Anomaly score: {sensor_score:.3f}")
                print(f"  Confidence: {result.confidence:.3f}")
            else:
                normal_detected += 1
                print(f"Reading {i+1}: [NORMAL] - Score: {sensor_score:.3f}")
    
    # Summary
    print("\n" + "="*80)
    print("DETECTION SUMMARY")
    print("="*80)
    print(f"Total readings processed: 10")
    print(f"Anomalies detected: {anomalies_detected}")
    print(f"Normal operation: {normal_detected}")
    print(f"Detection rate: {anomalies_detected/10*100:.1f}%")
    print(f"Average anomaly score: {sum(anomaly_scores_list)/len(anomaly_scores_list):.3f}")
    print(f"Max anomaly score: {max(anomaly_scores_list):.3f}")
    print(f"Min anomaly score: {min(anomaly_scores_list):.3f}")
    
    print(f"\n" + "="*80)
    if anomalies_detected > 0:
        print(f"SUCCESS: Detected {anomalies_detected} anomalies in test data!")
        print("The system is working correctly with Qdrant Cloud baselines.")
    else:
        print("WARNING: No anomalies detected")
        print("This may indicate threshold adjustment is needed.")
    print("="*80)
    
    qdrant_client.close()
    return anomalies_detected > 0


if __name__ == "__main__":
    print("\nStarting sensor anomaly detection...")
    try:
        success = asyncio.run(detect_sensor_anomalies())
        print(f"\nTask completed: {'SUCCESS' if success else 'NEEDS REVIEW'}")
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTask interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n\nTask failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
