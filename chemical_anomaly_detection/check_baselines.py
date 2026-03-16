"""Check what's in the baselines collection"""

from src.database.client_factory import create_qdrant_client

def check_baselines():
    print("\n" + "="*80)
    print("Checking Baselines Collection")
    print("="*80)
    
    # Connect
    client = create_qdrant_client()
    
    # Get collection info
    info = client.get_collection("baselines")
    print(f"\nCollection: baselines")
    print(f"Points: {info.points_count}")
    print(f"Vectors: {list(info.config.params.vectors.keys())}")
    
    # Get a sample point
    print(f"\nFetching sample points...")
    result = client.scroll(
        collection_name="baselines",
        limit=3,
        with_vectors=True,
        with_payload=True
    )
    
    points, _ = result
    
    for i, point in enumerate(points):
        print(f"\nPoint {i+1}:")
        print(f"  ID: {point.id}")
        print(f"  Vectors available: {list(point.vector.keys())}")
        
        for modality, vec in point.vector.items():
            vec_array = vec if isinstance(vec, list) else [vec]
            non_zero = sum(1 for v in vec_array if v != 0)
            print(f"    {modality}: {len(vec_array)} dims, {non_zero} non-zero values")
        
        if point.payload:
            print(f"  Payload: {point.payload}")
    
    client.close()
    print("\n" + "="*80)

if __name__ == "__main__":
    check_baselines()
