"""Check what's actually stored in Qdrant data collection"""

import os
from qdrant_client import QdrantClient

def check_data_collection():
    """Check the data collection contents"""
    print("\n" + "="*80)
    print("Qdrant Data Collection Check")
    print("="*80)
    
    # Connect to Qdrant
    qdrant_client = QdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", "6333"))
    )
    
    # Get collection info
    print("\n[1] Collection Info:")
    try:
        info = qdrant_client.get_collection("data")
        print(f"  Points count: {info.points_count}")
        print(f"  Vectors config: {list(info.config.params.vectors.keys())}")
    except Exception as e:
        print(f"  ERROR: {e}")
        return
    
    # Scroll through all points
    print("\n[2] Scanning all points...")
    offset = None
    total_points = 0
    video_present = 0
    audio_present = 0
    sensor_present = 0
    
    while True:
        result = qdrant_client.scroll(
            collection_name="data",
            limit=100,
            offset=offset,
            with_vectors=True,
            with_payload=True
        )
        
        points, next_offset = result
        
        if not points:
            break
        
        for point in points:
            total_points += 1
            vectors = list(point.vector.keys())
            
            if "video" in vectors:
                video_present += 1
            if "audio" in vectors:
                audio_present += 1
            if "sensor" in vectors:
                sensor_present += 1
            
            # Print first 5 points in detail
            if total_points <= 5:
                print(f"\n  Point {total_points}:")
                print(f"    ID: {point.id}")
                print(f"    Vectors: {vectors}")
                for v in vectors:
                    print(f"      - {v}: {len(point.vector[v])} dims")
                if point.payload:
                    print(f"    Payload keys: {list(point.payload.keys())}")
                    if "modality_status" in point.payload:
                        print(f"    Modality status: {point.payload['modality_status']}")
        
        if next_offset is None:
            break
        offset = next_offset
    
    # Summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print(f"Total points: {total_points}")
    print(f"Points with video: {video_present} ({video_present/total_points*100:.1f}%)")
    print(f"Points with audio: {audio_present} ({audio_present/total_points*100:.1f}%)")
    print(f"Points with sensor: {sensor_present} ({sensor_present/total_points*100:.1f}%)")
    
    if video_present == 0 and total_points > 0:
        print("\nWARNING: No points have video embeddings!")
    elif video_present < total_points:
        print(f"\nWARNING: {total_points - video_present} points are missing video embeddings!")
    else:
        print("\nSUCCESS: All points have video embeddings!")
    
    qdrant_client.close()


if __name__ == "__main__":
    check_data_collection()
