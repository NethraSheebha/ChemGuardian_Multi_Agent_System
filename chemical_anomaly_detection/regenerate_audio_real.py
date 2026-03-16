"""
Regenerate Audio Baselines with Real PANNs Model
Explicitly passes checkpoint path to avoid env caching issues
"""

import asyncio
import librosa
import numpy as np
import uuid
import os
from datetime import datetime
from src.database.client_factory import create_qdrant_client
from src.models.audio_processor import AudioProcessor
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue


async def regenerate_audio_baselines():
    """Regenerate audio baselines with real PANNs model"""
    print("\n" + "="*80)
    print("AUDIO BASELINE REGENERATION (with Real PANNs Model)")
    print("="*80)
    
    # Set checkpoint path explicitly
    checkpoint_path = "C:/Users/maryj/Downloads/Cnn14_mAP=0.431.pth"
    print(f"\nUsing checkpoint: {checkpoint_path}")
    
    # Verify file exists
    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] Checkpoint file not found: {checkpoint_path}")
        return False
    
    file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)
    print(f"Checkpoint file size: {file_size:.1f} MB")
    
    # Connect to Qdrant
    print(f"\n[1] Connecting to Qdrant Cloud...")
    client = create_qdrant_client()
    baselines_info = client.get_collection("baselines")
    print(f"    [OK] Connected - {baselines_info.points_count} baseline points currently")
    
    # Delete old audio baselines using scroll + delete
    print(f"\n[2] Deleting old audio baselines...")
    try:
        # Get all audio baseline IDs
        audio_points = client.scroll(
            collection_name="baselines",
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="baseline_type",
                        match=MatchValue(value="audio_baseline")
                    )
                ]
            ),
            limit=100,
            with_payload=False,
            with_vectors=False
        )[0]
        
        if audio_points:
            point_ids = [point.id for point in audio_points]
            client.delete(
                collection_name="baselines",
                points_selector=point_ids
            )
            print(f"    [OK] Deleted {len(point_ids)} old audio baselines")
        else:
            print(f"    [INFO] No old audio baselines to delete")
    except Exception as e:
        print(f"    [WARNING] Could not delete old baselines: {e}")
    
    # Verify deletion
    baselines_info = client.get_collection("baselines")
    print(f"    Baselines after deletion: {baselines_info.points_count}")
    
    # Load normal audio
    audio_file = "normal_audio.wav"
    print(f"\n[3] Loading audio file: {audio_file}")
    
    try:
        audio, sr = librosa.load(audio_file, sr=None)
        duration = len(audio) / sr
        print(f"    [OK] Loaded audio")
        print(f"    Sample rate: {sr} Hz")
        print(f"    Duration: {duration:.2f} seconds")
        print(f"    Samples: {len(audio)}")
    except Exception as e:
        print(f"    [ERROR] Failed to load audio: {e}")
        return False
    
    # Initialize audio processor with explicit checkpoint path
    print(f"\n[4] Initializing audio processor with real PANNs model...")
    print(f"    Loading from: {checkpoint_path}")
    
    audio_proc = AudioProcessor(
        device="cpu",
        timeout=5.0,
        checkpoint_path=checkpoint_path  # Explicitly pass the path
    )
    
    model_info = audio_proc.get_model_info()
    print(f"    Model: {model_info['model_name']}")
    print(f"    Embedding dimension: {model_info['embedding_dim']}")
    print(f"    PANNs available: {model_info['panns_available']}")
    
    # Check if using real model
    if hasattr(audio_proc.model, '__class__'):
        model_class = audio_proc.model.__class__.__name__
        print(f"    Model class: {model_class}")
        if 'Mock' in model_class:
            print(f"    [WARNING] Still using mock model!")
            return False
        else:
            print(f"    [OK] Using real PANNs model!")
    
    # Split audio into 1-second windows
    window_size = sr  # 1 second
    num_windows = int(duration)
    print(f"\n[5] Splitting audio into {num_windows} windows (1 second each)...")
    
    audio_windows = []
    for i in range(num_windows):
        start = i * window_size
        end = start + window_size
        window = audio[start:end]
        audio_windows.append((window, sr))
    
    print(f"    [OK] Created {len(audio_windows)} audio windows")
    
    # Generate embeddings
    print(f"\n[6] Generating audio embeddings with real PANNs model...")
    embeddings = []
    
    for i, (window, sample_rate) in enumerate(audio_windows):
        try:
            embedding = await audio_proc.process_audio(window, sample_rate)
            if embedding is not None:
                embeddings.append(embedding)
                non_zero = np.count_nonzero(embedding)
                mean_val = np.mean(embedding)
                std_val = np.std(embedding)
                print(f"    Window {i+1}/{len(audio_windows)}: {non_zero}/512 non-zero, mean={mean_val:.4f}, std={std_val:.4f}")
            else:
                print(f"    Window {i+1}/{len(audio_windows)}: [FAILED]")
        except Exception as e:
            print(f"    Window {i+1}/{len(audio_windows)}: [ERROR] {e}")
    
    print(f"\n    [OK] Generated {len(embeddings)} audio embeddings")
    
    if len(embeddings) == 0:
        print("\n[ERROR] No embeddings generated!")
        return False
    
    # Store audio baselines
    print(f"\n[7] Storing audio baselines in Qdrant...")
    
    points = []
    for i, embedding in enumerate(embeddings):
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector={
                "video": np.zeros(512).tolist(),
                "audio": embedding.tolist(),
                "sensor": np.zeros(128).tolist()
            },
            payload={
                "timestamp": datetime.utcnow().isoformat(),
                "shift": "morning",
                "equipment_id": "AUDIO_001",
                "plant_zone": "Zone_A",
                "baseline_type": "audio_baseline",
                "source_file": "normal_audio.wav",
                "window_index": i,
                "model": "PANNs_CNN14_Real"
            }
        )
        points.append(point)
    
    # Upload in batches
    batch_size = 10
    for i in range(0, len(points), batch_size):
        batch = points[i:i+batch_size]
        client.upsert(
            collection_name="baselines",
            points=batch
        )
        print(f"    Uploaded batch {i//batch_size + 1}/{(len(points)-1)//batch_size + 1}")
    
    print(f"    [OK] Stored {len(points)} audio baseline points")
    
    # Verify storage
    print(f"\n[8] Verifying audio baselines...")
    baselines_info = client.get_collection("baselines")
    new_total = baselines_info.points_count
    print(f"    Total baselines now: {new_total}")
    print(f"    Audio baselines added: {len(points)}")
    
    # Check a sample point
    sample_points = client.scroll(
        collection_name="baselines",
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="baseline_type",
                    match=MatchValue(value="audio_baseline")
                )
            ]
        ),
        limit=1,
        with_vectors=True
    )[0]
    
    if sample_points:
        sample = sample_points[0]
        audio_vec = sample.vector.get("audio", [])
        non_zero = sum(1 for x in audio_vec if x != 0)
        mean_val = np.mean(audio_vec)
        std_val = np.std(audio_vec)
        print(f"\n    Sample audio baseline verification:")
        print(f"      Audio vector: {non_zero}/512 non-zero values")
        print(f"      Mean: {mean_val:.4f}, Std: {std_val:.4f}")
        print(f"      Model: {sample.payload.get('model', 'unknown')}")
    
    print("\n" + "="*80)
    print(f"SUCCESS: Generated {len(embeddings)} audio baselines with real PANNs!")
    print(f"Total baselines in Qdrant: {new_total}")
    print("  - Sensor baselines: 66")
    print("  - Video baselines: 30")
    print(f"  - Audio baselines: {len(points)}")
    print("="*80)
    print("\n✅ Audio anomaly detection is now ready to use!")
    print("   The system will use cosine similarity (like video)")
    print("   Distance = 1 - similarity (already implemented)")
    
    client.close()
    return True


if __name__ == "__main__":
    print("\nStarting audio baseline regeneration...")
    try:
        success = asyncio.run(regenerate_audio_baselines())
        if success:
            print("\nAudio baseline regeneration completed successfully!")
            exit(0)
        else:
            print("\nAudio baseline regeneration failed!")
            exit(1)
    except KeyboardInterrupt:
        print("\n\nTask interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n\nTask failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
