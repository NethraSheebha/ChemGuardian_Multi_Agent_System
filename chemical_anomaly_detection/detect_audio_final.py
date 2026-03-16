"""
Audio Anomaly Detection - Final Version
Processes anomalous_audio.wav to detect audio anomalies
"""

import asyncio
import librosa
import os
import numpy as np
from datetime import datetime

from src.database.client_factory import create_qdrant_client
from src.models.audio_processor import AudioProcessor
from src.agents.similarity_search_engine import SimilaritySearchEngine
from src.agents.input_collection_agent import MultimodalEmbedding
from qdrant_client.models import Filter, FieldCondition, MatchValue


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
    
    # Initialize audio processor
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
    
    # Initialize search engine
    print(f"\n[5] Initializing similarity search...")
    similarity_search = SimilaritySearchEngine(qdrant_client)
    audio_threshold = 0.65
    print(f"    [OK] Audio threshold: {audio_threshold}")
    
    # Split audio into 1-second windows
    window_size = sr  # 1 second
    num_windows = int(duration)
    print(f"\n[6] Processing {num_windows} audio windows...")
    print("="*80 + "\n")
    
    anomalies_detected = 0
    normal_detected = 0
    audio_scores = []
    failed_windows = 0
    
    for i in range(num_windows):
        start = i * window_size
        end = start + window_size
        audio_window = audio[start:end]
        
        try:
            # Generate embedding
            embedding = await audio_proc.process_audio(audio_window, sr)
            
            if embedding is not None:
                # Create multimodal embedding
                mm_embedding = MultimodalEmbedding(
                    video_embedding=None,
                    audio_embedding=embedding,
                    sensor_embedding=None,
                    timestamp=datetime.utcnow().isoformat(),
                    metadata={
                        "plant_zone": "Zone_A",
                        "shift": "morning",
                        "equipment_id": "AUDIO_001",
                        "window_index": i,
                        "audio_file": "anomalous_audio.wav"
                    }
                )
                
                # Search baselines
                search_results = await similarity_search.search_baselines(mm_embedding)
                
                # Compute anomaly scores
                anomaly_scores_dict = similarity_search.compute_anomaly_scores(search_results)
                
                audio_score = anomaly_scores_dict.get('audio', 0)
                audio_scores.append(audio_score)
                
                is_anomaly = audio_score > audio_threshold
                
                if is_anomaly:
                    anomalies_detected += 1
                    print(f"Window {i+1:2d}: [ANOMALY] Score: {audio_score:.3f} (threshold: {audio_threshold})")
                else:
                    normal_detected += 1
                    print(f"Window {i+1:2d}: [NORMAL]  Score: {audio_score:.3f}")
            else:
                failed_windows += 1
                print(f"Window {i+1:2d}: [FAILED] Could not generate embedding")
                
        except Exception as e:
            failed_windows += 1
            print(f"Window {i+1:2d}: [ERROR] {e}")
    
    # Summary
    print("\n" + "="*80)
    print("DETECTION SUMMARY")
    print("="*80)
    print(f"\nWindows processed: {num_windows}")
    print(f"Successful: {len(audio_scores)}")
    print(f"Failed: {failed_windows}")
    print(f"\nAnomalies detected: {anomalies_detected}")
    print(f"Normal windows: {normal_detected}")
    
    if len(audio_scores) > 0:
        detection_rate = (anomalies_detected / len(audio_scores)) * 100
        print(f"Detection rate: {detection_rate:.1f}%")
        
        print(f"\nAudio Anomaly Scores:")
        print(f"  Average: {np.mean(audio_scores):.3f}")
        print(f"  Maximum: {np.max(audio_scores):.3f}")
        print(f"  Minimum: {np.min(audio_scores):.3f}")
        print(f"  Std Dev: {np.std(audio_scores):.3f}")
        print(f"  Threshold: {audio_threshold}")
        
        # Show distribution
        above_threshold = sum(1 for s in audio_scores if s > audio_threshold)
        print(f"\n  Windows above threshold: {above_threshold}/{len(audio_scores)}")
    
    print(f"\n" + "="*80)
    if anomalies_detected > 0:
        print(f"✅ SUCCESS: Detected {anomalies_detected} audio anomalies!")
        print("\nThe audio contains anomalous acoustic patterns:")
        print("  • Unusual sound signatures")
        print("  • Abnormal acoustic events")
        print("  • Deviation from normal audio baselines")
        print("\nPossible causes:")
        print("  • Gas leak hissing sounds")
        print("  • Equipment malfunction noises")
        print("  • Alarm patterns")
        print("  • Abnormal silence periods")
    else:
        print("ℹ️  INFO: No audio anomalies detected")
        print("The audio appears normal based on baseline comparisons.")
    print("="*80)
    
    qdrant_client.close()
    return True


if __name__ == "__main__":
    print("\n🎵 Starting audio anomaly detection...")
    try:
        success = asyncio.run(detect_audio_anomalies())
        if success:
            print(f"\n✅ Task completed successfully!")
        else:
            print(f"\n❌ Task failed!")
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Task interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n\n❌ Task failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
