"""Check video baselines specifically"""

from src.database.client_factory import create_qdrant_client

def check_video_baselines():
    print("\n" + "="*80)
    print("Checking Video Baselines")
    print("="*80)
    
    client = create_qdrant_client()
    
    # Get collection info
    info = client.get_collection("baselines")
    print(f"\nTotal baseline points: {info.points_count}")
    
    # Get video baselines specifically
    print(f"\nFetching video baselines...")
    result = client.scroll(
        collection_name="baselines",
        scroll_filter={
            "must": [
                {"key": "baseline_type", "match": {"value": "video_baseline"}}
            ]
        },
        limit=5,
        with_vectors=True,
        with_payload=True
    )
    
    points, _ = result
    
    print(f"Found {len(points)} video baseline points")
    
    for i, point in enumerate(points):
        print(f"\nVideo Baseline {i+1}:")
        print(f"  ID: {point.id}")
        
        video_vec = point.vector['video']
        non_zero = sum(1 for v in video_vec if v != 0)
        print(f"  Video: {len(video_vec)} dims, {non_zero} non-zero values")
        
        if point.payload:
            print(f"  Source: {point.payload.get('source_video', 'unknown')}")
            print(f"  Frame: {point.payload.get('frame_number', 'unknown')}")
    
    # Also check sensor baselines
    print(f"\n" + "-"*80)
    print("Checking sensor baselines...")
    result2 = client.scroll(
        collection_name="baselines",
        scroll_filter={
            "must": [
                {"key": "baseline_type", "match": {"value": "global_baseline"}}
            ]
        },
        limit=2,
        with_vectors=True
    )
    
    points2, _ = result2
    print(f"Found {len(points2)} sensor baseline points (showing 2)")
    
    for i, point in enumerate(points2):
        sensor_vec = point.vector['sensor']
        video_vec = point.vector['video']
        sensor_non_zero = sum(1 for v in sensor_vec if v != 0)
        video_non_zero = sum(1 for v in video_vec if v != 0)
        print(f"  Point {i+1}: Sensor {sensor_non_zero}/128, Video {video_non_zero}/512")
    
    client.close()
    print("\n" + "="*80)

if __name__ == "__main__":
    check_video_baselines()
