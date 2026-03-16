"""
Audio Anomaly Detection
Processes anomalous_audio.wav to detect audio anomalies using Qdrant Cloud baselines
"""

import asyncio
import librosa
import os
from datetime import datetime

from src.database.client_factory import create_qdrant_client
from src.agents.input_collection_agent import InputCollectionAgent, EmbeddingGenerator
from src.agents.anomaly_detection_agent import AnomalyDetectionAgent
from src.models.audio_processor import AudioProcessor
from src.agents.similarity_search_engine import SimilaritySearchEngine
from src.agents.adaptive_threshold_manager import AdaptiveThresholdManager
from src.agents.storage_manager import StorageManager


async def detect_audio_anomalies():
    """Detect anomalies in audio windows"""
    print("\n" + "="*80)
    print("AUDIO ANOMALY DETECTION - anomalous_audio.wav")
    print("="*80)
    
    # Check audio file
    audio_path = "anomalous_audio.wav"
    if not os.path.exists(audio_path):
        print(f"\n[ERROR] Audio file not found: {audio_path}")
        return False
    
    print(f"\n[1] Audio file found: {audio_path}")
    file_size = os.path.getsize(audio_path) / 1024
    print(f"    Size: {file_size:.1f} KB")
    
    # Load audio
    print(f"\n[2] Loading audio...")
    try:
        audio, sr = librosa.load(audio_path, sr=None)
        duration = len(audio) / sr
        print(f"    [OK] Loaded audio")
        print(f"    Sample rate: {sr} Hz")
        print(f"    Duration: {duration:.2f} seconds")
        print(f"    Samples: {len(audio)}")
    except Exception as e:
        print(f"    [ERROR] Failed to load audio: {e}")
        return False
    
    # Connect to Qdrant Cloud
    print(f"\n[3] Connecting to Qdrant Cloud...")
    qdrant_client = create_qdrant_client()
    
    baselines_info = qdrant_client.get_collection("baselines")
    print(f"    [OK] Connected - {baselines_info.points_count} baseline points available")
    
    # Check audio baselines
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    audio_baselines = qdrant_client.scroll(
        collection_name="baselines",
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="baseline_type",
                    match=MatchValue(value="audio_baseline")
                )
            ]
        ),
        limit=1
    )[0]
    
    if not audio_baselines:
        print(f"    [ERROR] No audio baselines found!")
        print(f"    Run: python regenerate_audio_real.py")
        return False
    
    print(f"    [OK] Audio baselines available")
    
    # Initialize audio processor with correct checkpoint
    print(f"\n[4] Initializing audio processor...")
    checkpoint_path = "C:/Users/maryj/Downloads/Cnn14_mAP=0.431.pth"
    audio_proc = AudioProcessor(
        device="cpu",
        timeout=5.0,
        checkpoint_path=checkpoint_path
    )
    
    model_info = audio_proc.get_model_info()
    print(f"    [OK] Model: {model_info['model_name']}")
    print(f"    [OK] Embedding dimension: {model_info['embedding_dim']}")
    
    # Check if real model
    if hasattr(audio_proc.model, '__class__'):
        model_class = audio_proc.model.__class__.__name__
        if 'Mock' in model_class:
            print(f"    [WARNING] Using mock model - results may not be accurate")
        else:
            print(f"    [OK] Using real PANNs model")
    
    # Initialize agents
    print(f"\n[5] Initializing detection agents...")
    embedding_gen = EmbeddingGenerator(
        video_processor=None,
        audio_processor=audio_proc,
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
    
    # Split audio into 1-second windows
    window_size = sr  # 1 second
    num_windows = int(duration)
    print(f"\n[6] Processing audio windows...")
    print("="*80)
    
    print(f"\nProcessing {num_windows} audio windows (1 second each)...\n")
    
    anomalies_detected = 0
    normal_detected = 0
    audio_scores = []
    
    for i in range(num_windows):
        start = i * window_size
        end = start + window_size
        audio_window = audio[start:end]
        
        metadata = {
            "plant_zone": "Zone_A",
            "shift": "morning",
            "equipment_id": "AUDIO_001",
            "window_index": i,
            "audio_file": "anomalous_audio.wav"
        }
        
        try:
            # Generate embedding (audio only)
            embedding = await input_agent.process_data_point(
                audio_data=(audio_window, sr),
                metadata=metadata
            )
            
            if embedding and embedding.has_any_modality():
                modalities = embedding.get_available_modalities()
                
                # Detect anomaly
                result = await anomaly_agent.detect_anomaly(embedding)
                
                audio_score = result.anomaly_scores.get('audio', 0)
                audio_scores.append(audio_score)
                
                if result.is_anomaly:
                    anomalies_detected += 1
                    print(f"Window {i+1}: [ANOMALY DETECTED]")
                    print(f"  Modalities: {', '.join(modalities)}")
                    print(f"  Audio anomaly score: {audio_score:.3f}")
                    print(f"  Confidence: {result.confidence:.3f}")
                else:
                    normal_detected += 1
                    print(f"Window {i+1}: [NORMAL] - Audio score: {audio_score:.3f}")
            else:
                print(f"Window {i+1}: [ERROR] Failed to generate embedding")
                
        except Exception as e:
            print(f"Window {i+1}: [ERROR] {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*80)
    print("DETECTION SUMMARY")
    print("="*80)
    print(f"\nWindows processed: {num_windows}")
    print(f"Anomalies detected: {anomalies_detected}")
    print(f"Normal windows: {normal_detected}")
    print(f"Detection rate: {anomalies_detected/num_windows*100:.1f}%")
    
    if audio_scores:
        print(f"\nAudio Anomaly Scores:")
        print(f"  Average: {sum(audio_scores)/len(audio_scores):.3f}")
        print(f"  Maximum: {max(audio_scores):.3f}")
        print(f"  Minimum: {min(audio_scores):.3f}")
        print(f"  Threshold: 0.65")
        
        # Show distribution
        above_threshold = sum(1 for s in audio_scores if s > 0.65)
        print(f"\n  Windows above threshold: {above_threshold}/{len(audio_scores)}")
    
    print(f"\n" + "="*80)
    if anomalies_detected > 0:
        print(f"SUCCESS: Detected {anomalies_detected} audio anomalies!")
        print("The audio contains anomalous content detected by the system.")
        print("\nAnomaly Types Detected:")
        print("  - Unusual acoustic patterns")
        print("  - Abnormal sound signatures")
        print("  - Deviation from normal audio baselines")
    else:
        print("INFO: No audio anomalies detected")
        print("The audio appears normal based on baseline comparisons.")
    print("="*80)
    
    qdrant_client.close()
    return True


if __name__ == "__main__":
    print("\nStarting audio anomaly detection...")
    try:
        success = asyncio.run(detect_audio_anomalies())
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
