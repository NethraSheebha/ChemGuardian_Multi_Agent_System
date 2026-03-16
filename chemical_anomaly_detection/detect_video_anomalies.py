"""
Video Anomaly Detection
Processes anomalous_1.mp4 to detect visual anomalies using Qdrant Cloud baselines
"""

import asyncio
import cv2
import os
from datetime import datetime

from src.database.client_factory import create_qdrant_client
from src.agents.input_collection_agent import InputCollectionAgent, EmbeddingGenerator
from src.agents.anomaly_detection_agent import AnomalyDetectionAgent
from src.models.video_processor import VideoProcessor
from src.agents.similarity_search_engine import SimilaritySearchEngine
from src.agents.adaptive_threshold_manager import AdaptiveThresholdManager
from src.agents.storage_manager import StorageManager


async def detect_video_anomalies():
    """Detect anomalies in video frames"""
    print("\n" + "="*80)
    print("VIDEO ANOMALY DETECTION - anomalous_1.mp4")
    print("="*80)
    
    # Check video file
    video_path = "anomalous_1.mp4"
    if not os.path.exists(video_path):
        print(f"\n[ERROR] Video file not found: {video_path}")
        return False
    
    print(f"\n[1] Video file found: {video_path}")
    print(f"    Size: {os.path.getsize(video_path) / 1024:.1f} KB")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Failed to open video file")
        return False
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    
    print(f"\n    Video properties:")
    print(f"      Resolution: {width}x{height}")
    print(f"      FPS: {fps:.2f}")
    print(f"      Total frames: {frame_count}")
    print(f"      Duration: {duration:.2f} seconds")
    
    # Connect to Qdrant Cloud
    print(f"\n[2] Connecting to Qdrant Cloud...")
    qdrant_client = create_qdrant_client()
    
    baselines_info = qdrant_client.get_collection("baselines")
    print(f"    [OK] Connected - {baselines_info.points_count} baseline points available")
    
    # Initialize video processor
    print(f"\n[3] Initializing video processor...")
    video_proc = VideoProcessor(device="cpu", timeout=2.0)
    print(f"    [OK] Model: {video_proc.model_name}")
    print(f"    [OK] Embedding dimension: 512")
    
    # Initialize agents
    print(f"\n[4] Initializing detection agents...")
    embedding_gen = EmbeddingGenerator(
        video_processor=video_proc,
        audio_processor=None,
        sensor_processor=None
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
    print(f"    [OK] Agents initialized")
    
    # Process video frames
    print(f"\n[5] Processing video frames...")
    print("="*80)
    
    frames_to_process = min(10, frame_count)
    print(f"\nProcessing first {frames_to_process} frames...\n")
    
    anomalies_detected = 0
    normal_detected = 0
    video_scores = []
    
    for i in range(frames_to_process):
        ret, frame = cap.read()
        if not ret:
            print(f"Frame {i+1}: [ERROR] Failed to read frame")
            break
        
        metadata = {
            "plant_zone": "Zone_A",
            "shift": "morning",
            "camera_id": "anomalous_cam",
            "frame_number": i,
            "video_file": "anomalous_1.mp4"
        }
        
        try:
            # Generate embedding (video only)
            embedding = await input_agent.process_data_point(
                video_frame=frame,
                metadata=metadata
            )
            
            if embedding and embedding.has_any_modality():
                modalities = embedding.get_available_modalities()
                
                # Detect anomaly
                result = await anomaly_agent.detect_anomaly(embedding)
                
                video_score = result.anomaly_scores.get('video', 0)
                video_scores.append(video_score)
                
                if result.is_anomaly:
                    anomalies_detected += 1
                    print(f"Frame {i+1}: [ANOMALY DETECTED]")
                    print(f"  Modalities: {', '.join(modalities)}")
                    print(f"  Video anomaly score: {video_score:.3f}")
                    print(f"  Confidence: {result.confidence:.3f}")
                else:
                    normal_detected += 1
                    print(f"Frame {i+1}: [NORMAL] - Video score: {video_score:.3f}")
            else:
                print(f"Frame {i+1}: [ERROR] Failed to generate embedding")
                
        except Exception as e:
            print(f"Frame {i+1}: [ERROR] {e}")
    
    cap.release()
    
    # Summary
    print("\n" + "="*80)
    print("DETECTION SUMMARY")
    print("="*80)
    print(f"\nFrames processed: {frames_to_process}")
    print(f"Anomalies detected: {anomalies_detected}")
    print(f"Normal frames: {normal_detected}")
    print(f"Detection rate: {anomalies_detected/frames_to_process*100:.1f}%")
    
    if video_scores:
        print(f"\nVideo Anomaly Scores:")
        print(f"  Average: {sum(video_scores)/len(video_scores):.3f}")
        print(f"  Maximum: {max(video_scores):.3f}")
        print(f"  Minimum: {min(video_scores):.3f}")
        print(f"  Threshold: 0.7")
    
    print(f"\n" + "="*80)
    if anomalies_detected > 0:
        print(f"SUCCESS: Detected {anomalies_detected} visual anomalies!")
        print("The video contains anomalous content detected by the system.")
    else:
        print("INFO: No visual anomalies detected")
        print("The video appears normal based on baseline comparisons.")
    print("="*80)
    
    qdrant_client.close()
    return True


if __name__ == "__main__":
    print("\nStarting video anomaly detection...")
    try:
        success = asyncio.run(detect_video_anomalies())
        print(f"\nTask completed successfully")
        exit(0)
    except KeyboardInterrupt:
        print("\n\nTask interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n\nTask failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
