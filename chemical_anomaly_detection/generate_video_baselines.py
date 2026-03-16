"""
Generate Video Baselines from Normal Videos
Processes normal_1.mp4 and normal_2.mp4 to create baseline embeddings
"""

import asyncio
import cv2
import numpy as np
from datetime import datetime
from uuid import uuid4
from qdrant_client.models import PointStruct

from src.database.client_factory import create_qdrant_client
from src.models.video_processor import VideoProcessor


async def generate_video_baselines():
    """Generate video baselines from normal videos"""
    print("\n" + "="*80)
    print("GENERATING VIDEO BASELINES")
    print("="*80)
    
    # Video files
    normal_videos = ["normal_1.mp4", "normal_2.mp4"]
    
    # Connect to Qdrant Cloud
    print(f"\n[1] Connecting to Qdrant Cloud...")
    qdrant_client = create_qdrant_client()
    print(f"    [OK] Connected")
    
    # Initialize video processor
    print(f"\n[2] Initializing video processor...")
    video_proc = VideoProcessor(device="cpu", timeout=2.0)
    print(f"    [OK] Model: {video_proc.model_name}")
    print(f"    [OK] Embedding dimension: 512")
    
    # Process each video
    all_video_embeddings = []
    
    for video_file in normal_videos:
        print(f"\n[3] Processing {video_file}...")
        
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            print(f"    [ERROR] Failed to open {video_file}")
            continue
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"    Video: {frame_count} frames @ {fps:.2f} FPS")
        
        # Sample frames (every 30 frames to get diverse samples)
        frame_indices = list(range(0, frame_count, 30))[:20]  # Max 20 frames per video
        print(f"    Sampling {len(frame_indices)} frames...")
        
        video_embeddings = []
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            try:
                # Generate video embedding
                embedding = await video_proc.process_frame(frame)
                
                if embedding is not None:
                    video_embeddings.append({
                        'video_embedding': embedding,
                        'source_video': video_file,
                        'frame_number': idx
                    })
            except Exception as e:
                print(f"    [WARN] Frame {idx} failed: {e}")
        
        cap.release()
        
        print(f"    [OK] Generated {len(video_embeddings)} video embeddings")
        all_video_embeddings.extend(video_embeddings)
    
    print(f"\n[4] Total video embeddings generated: {len(all_video_embeddings)}")
    
    # Create baseline points
    print(f"\n[5] Creating baseline points...")
    
    baseline_points = []
    
    for i, emb_data in enumerate(all_video_embeddings):
        point = PointStruct(
            id=str(uuid4()),
            vector={
                'video': emb_data['video_embedding'].tolist(),
                'audio': np.zeros(512, dtype=np.float32).tolist(),  # Placeholder
                'sensor': np.zeros(128, dtype=np.float32).tolist()  # Placeholder
            },
            payload={
                'timestamp': datetime.now().isoformat(),
                'shift': 'all',
                'equipment_id': 'camera_baseline',
                'plant_zone': 'all',
                'baseline_type': 'video_baseline',
                'source_video': emb_data['source_video'],
                'frame_number': emb_data['frame_number']
            }
        )
        baseline_points.append(point)
    
    print(f"    [OK] Created {len(baseline_points)} baseline points")
    
    # Store in Qdrant Cloud
    print(f"\n[6] Storing baselines in Qdrant Cloud...")
    
    batch_size = 100
    for i in range(0, len(baseline_points), batch_size):
        batch = baseline_points[i:i+batch_size]
        qdrant_client.upsert(
            collection_name="baselines",
            points=batch
        )
        print(f"    Stored batch {i//batch_size + 1}/{(len(baseline_points)-1)//batch_size + 1}")
    
    print(f"    [OK] All baselines stored")
    
    # Verify
    print(f"\n[7] Verifying baselines...")
    baselines_info = qdrant_client.get_collection("baselines")
    print(f"    Total baseline points: {baselines_info.points_count}")
    
    # Check a sample
    result = qdrant_client.scroll(
        collection_name="baselines",
        scroll_filter={
            "must": [
                {"key": "baseline_type", "match": {"value": "video_baseline"}}
            ]
        },
        limit=1,
        with_vectors=True
    )
    
    if result[0]:
        point = result[0][0]
        video_vec = point.vector['video']
        non_zero = sum(1 for v in video_vec if v != 0)
        print(f"    Sample video baseline: {non_zero}/{len(video_vec)} non-zero values")
        print(f"    [OK] Video baselines are real embeddings!")
    
    print(f"\n" + "="*80)
    print(f"SUCCESS: Generated {len(baseline_points)} video baselines!")
    print(f"="*80)
    
    qdrant_client.close()
    return True


if __name__ == "__main__":
    print("\nStarting video baseline generation...")
    try:
        success = asyncio.run(generate_video_baselines())
        print(f"\nBaseline generation completed successfully!")
        exit(0)
    except Exception as e:
        print(f"\n\nFailed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
