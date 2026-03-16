"""
Example: Using CrewAI Agents for Chemical Leak Monitoring
Demonstrates how to use the converted CrewAI agents
"""

import asyncio
import numpy as np
from datetime import datetime

from src.database.client_factory import create_qdrant_client
from src.models.video_processor import VideoProcessor
from src.models.audio_processor import AudioProcessor
from src.models.sensor_processor import SensorProcessor

# Import CrewAI agents
from src.crewai_agents.input_collection_crew import InputCollectionCrew
from src.crewai_agents.anomaly_detection_crew import AnomalyDetectionCrew

# Import supporting components (unchanged)
from src.agents.similarity_search_engine import SimilaritySearchEngine
from src.agents.adaptive_threshold_manager import AdaptiveThresholdManager
from src.agents.storage_manager import StorageManager


async def main():
    """
    Example workflow using CrewAI agents
    
    This demonstrates:
    1. Input Collection Crew - Multimodal embedding generation
    2. Anomaly Detection Crew - Anomaly detection with adaptive thresholds
    """
    
    print("="*80)
    print("CrewAI Chemical Leak Monitoring System - Example")
    print("="*80)
    
    # ========================================================================
    # STEP 1: Initialize Components
    # ========================================================================
    print("\n[1] Initializing components...")
    
    # Connect to Qdrant Cloud
    qdrant_client = create_qdrant_client()
    print("    ✓ Connected to Qdrant Cloud")
    
    # Initialize processors
    video_proc = VideoProcessor(device="cpu")
    audio_proc = AudioProcessor(
        device="cpu",
        checkpoint_path="C:/Users/maryj/Downloads/Cnn14_mAP=0.431.pth"
    )
    sensor_proc = SensorProcessor()
    print("    ✓ Initialized processors (video, audio, sensor)")
    
    # Initialize supporting components
    similarity_search = SimilaritySearchEngine(qdrant_client)
    threshold_manager = AdaptiveThresholdManager(
        video_threshold=0.7,
        audio_threshold=0.65,
        sensor_threshold=2.5
    )
    storage_manager = StorageManager(qdrant_client)
    print("    ✓ Initialized supporting components")
    
    # ========================================================================
    # STEP 2: Create CrewAI Agents
    # ========================================================================
    print("\n[2] Creating CrewAI agents...")
    
    # Input Collection Crew
    input_crew = InputCollectionCrew(
        video_processor=video_proc,
        audio_processor=audio_proc,
        sensor_processor=sensor_proc,
        processing_interval=1.0
    )
    print("    ✓ Created Input Collection Crew")
    
    # Anomaly Detection Crew
    anomaly_crew = AnomalyDetectionCrew(
        qdrant_client=qdrant_client,
        similarity_search_engine=similarity_search,
        adaptive_threshold_manager=threshold_manager,
        storage_manager=storage_manager,
        processing_interval=1.0,
        high_severity_min_modalities=2,
        temporal_confirmation_windows=3
    )
    print("    ✓ Created Anomaly Detection Crew")
    
    # ========================================================================
    # STEP 3: Process Sample Data
    # ========================================================================
    print("\n[3] Processing sample data...")
    
    # Generate sample data
    video_frame = np.random.rand(224, 224, 3).astype(np.float32)
    audio_data = np.random.randn(32000).astype(np.float32)  # 1 second at 32kHz
    sensor_reading = {
        "temperature": 25.5,
        "pressure": 101.3,
        "gas_concentration": 0.05,
        "vibration": 0.02,
        "flow_rate": 10.5
    }
    metadata = {
        "plant_zone": "Zone_A",
        "shift": "morning",
        "equipment_id": "EQUIP_001",
        "camera_id": "CAM_001"
    }
    
    print("    ✓ Generated sample data (video, audio, sensor)")
    
    # ========================================================================
    # STEP 4: Input Collection (CrewAI)
    # ========================================================================
    print("\n[4] Input Collection Crew - Generating embeddings...")
    
    embedding = await input_crew.process_data_point(
        video_frame=video_frame,
        audio_data=(audio_data, 32000),
        sensor_reading=sensor_reading,
        metadata=metadata
    )
    
    if embedding:
        modalities = embedding.get_available_modalities()
        print(f"    ✓ Generated embedding with modalities: {', '.join(modalities)}")
        print(f"    ✓ Timestamp: {embedding.timestamp}")
    else:
        print("    ✗ Failed to generate embedding")
        return
    
    # ========================================================================
    # STEP 5: Anomaly Detection (CrewAI)
    # ========================================================================
    print("\n[5] Anomaly Detection Crew - Detecting anomalies...")
    
    result = await anomaly_crew.detect_anomaly(
        embedding=embedding,
        shift=metadata.get("shift"),
        equipment_id=metadata.get("equipment_id"),
        plant_zone=metadata.get("plant_zone")
    )
    
    print(f"    ✓ Detection complete")
    print(f"    • Is Anomaly: {result.is_anomaly}")
    print(f"    • Confidence: {result.confidence:.3f}")
    print(f"    • Anomaly Scores:")
    for modality, score in result.anomaly_scores.items():
        status = "ANOMALY" if result.per_modality_decisions.get(modality, False) else "NORMAL"
        print(f"      - {modality}: {score:.3f} [{status}]")
    
    if result.requires_temporal_confirmation:
        print(f"    • Temporal Confirmation: {result.temporal_confirmation_count}/3 windows")
    
    # ========================================================================
    # STEP 6: Statistics
    # ========================================================================
    print("\n[6] Crew Statistics:")
    
    input_stats = input_crew.get_stats()
    print(f"\n    Input Collection Crew:")
    print(f"      • Total processed: {input_stats['tools']['total_processed']}")
    print(f"      • Successful: {input_stats['tools']['successful_embeddings']}")
    print(f"      • Failed: {input_stats['tools']['failed_embeddings']}")
    
    anomaly_stats = anomaly_crew.get_stats()
    print(f"\n    Anomaly Detection Crew:")
    print(f"      • Total processed: {anomaly_stats['tools']['total_processed']}")
    print(f"      • Anomalies detected: {anomaly_stats['tools']['anomalies_detected']}")
    print(f"      • Normal detected: {anomaly_stats['tools']['normal_detected']}")
    print(f"      • Anomaly rate: {anomaly_stats['tools']['anomaly_rate']:.1%}")
    
    # ========================================================================
    # STEP 7: Cleanup
    # ========================================================================
    print("\n[7] Cleanup...")
    qdrant_client.close()
    print("    ✓ Closed Qdrant connection")
    
    print("\n" + "="*80)
    print("✅ CrewAI Example Complete!")
    print("="*80)
    
    return result


async def batch_processing_example():
    """
    Example: Processing multiple data points in batch
    """
    print("\n" + "="*80)
    print("Batch Processing Example")
    print("="*80)
    
    # Initialize (same as above, abbreviated)
    qdrant_client = create_qdrant_client()
    video_proc = VideoProcessor(device="cpu")
    sensor_proc = SensorProcessor()
    
    input_crew = InputCollectionCrew(
        video_processor=video_proc,
        sensor_processor=sensor_proc
    )
    
    anomaly_crew = AnomalyDetectionCrew(
        qdrant_client=qdrant_client,
        similarity_search_engine=SimilaritySearchEngine(qdrant_client),
        adaptive_threshold_manager=AdaptiveThresholdManager(),
        storage_manager=StorageManager(qdrant_client)
    )
    
    print("\n[1] Processing 5 data points...")
    
    anomalies_detected = 0
    
    for i in range(5):
        # Generate sample data
        video_frame = np.random.rand(224, 224, 3).astype(np.float32)
        sensor_reading = {
            "temperature": 25.0 + np.random.randn(),
            "pressure": 101.0 + np.random.randn(),
            "gas_concentration": 0.05 + np.random.randn() * 0.01,
            "vibration": 0.02 + np.random.randn() * 0.005,
            "flow_rate": 10.0 + np.random.randn()
        }
        
        # Process
        embedding = await input_crew.process_data_point(
            video_frame=video_frame,
            sensor_reading=sensor_reading,
            metadata={"plant_zone": "Zone_A", "index": i}
        )
        
        if embedding:
            result = await anomaly_crew.detect_anomaly(embedding)
            
            status = "ANOMALY" if result.is_anomaly else "NORMAL"
            print(f"    Point {i+1}: [{status}] Confidence: {result.confidence:.3f}")
            
            if result.is_anomaly:
                anomalies_detected += 1
    
    print(f"\n[2] Summary:")
    print(f"    • Total processed: 5")
    print(f"    • Anomalies detected: {anomalies_detected}")
    print(f"    • Detection rate: {anomalies_detected/5*100:.1f}%")
    
    qdrant_client.close()
    print("\n✅ Batch processing complete!")


if __name__ == "__main__":
    print("\n🚀 Starting CrewAI Example...\n")
    
    # Run main example
    result = asyncio.run(main())
    
    # Uncomment to run batch processing example
    # asyncio.run(batch_processing_example())
    
    print("\n✅ All examples complete!")
