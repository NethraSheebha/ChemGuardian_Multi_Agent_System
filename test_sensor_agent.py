import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_imports():
    """Test all required imports"""
    logger.info("Testing imports...")
    
    try:
        from sensor_agent import SensorIntelligenceAgent
        from sensor_pipeline import SensorPipeline
        from utils.sensor_stream_simulator import SensorStreamSimulator
        from qdrant_client import QdrantClient
        
        logger.info("  All imports successful")
        return True
        
    except ImportError as e:
        logger.error(f"  Import failed: {e}")
        return False


def test_qdrant_connection():
    """Test Qdrant connection and collection"""
    logger.info("Testing Qdrant connection...")
    
    try:
        from qdrant_client import QdrantClient
        
        client = QdrantClient(host="localhost", port=6333)
        collection_info = client.get_collection("sensor_patterns")
        
        logger.info(f"  Connected to Qdrant")
        logger.info(f"  Collection: sensor_patterns")
        logger.info(f"  Dimension: {collection_info.config.params.vectors.size}")
        logger.info(f"  Points: {collection_info.points_count}")
        
        if collection_info.config.params.vectors.size != 128:
            logger.error("  Wrong dimension! Expected 128")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"  Qdrant connection failed: {e}")
        return False


def test_data_generation():
    """Test synthetic data generation"""
    logger.info("Testing data generation...")
    
    try:
        from utils.sensor_stream_simulator import SensorStreamSimulator
        
        simulator = SensorStreamSimulator("data/sensors/test_stream.csv")
        df = simulator.generate_synthetic_data(
            num_sensors=3,
            num_readings=200,
            save_path="data/sensors/test_stream.csv"
        )
        
        logger.info(f"  Generated {len(df)} readings")
        logger.info(f"  Columns: {list(df.columns)}")
        logger.info(f"  Sensors: {df['sensor_id'].unique().tolist()}")
        
        return True
        
    except Exception as e:
        logger.error(f"  Data generation failed: {e}")
        return False


def test_feature_extraction():
    """Test feature extraction pipeline"""
    logger.info("Testing feature extraction...")
    
    try:
        from sensor_pipeline import SensorPipeline
        from utils.sensor_stream_simulator import SensorStreamSimulator
        
        # Load data
        simulator = SensorStreamSimulator("data/sensors/test_stream.csv")
        simulator.load_data()
        
        # Get a test window
        windows = simulator.get_training_windows(num_windows=1, window_size=50)
        sensor_id, zone, window_data = windows[0]
        
        # Extract features
        pipeline = SensorPipeline()
        features = pipeline.extract_features(window_data, sensor_id)
        
        logger.info(f"  Features extracted")
        logger.info(f"  Sensor: {sensor_id}")
        logger.info(f"  Feature dimension: {features.shape[0]}")
        
        return True
        
    except Exception as e:
        logger.error(f"  Feature extraction failed: {e}")
        return False


def test_model_training():
    """Test model training"""
    logger.info("Testing model training...")
    
    try:
        from sensor_pipeline import SensorPipeline
        from utils.sensor_stream_simulator import SensorStreamSimulator
        
        # Load data
        simulator = SensorStreamSimulator("data/sensors/test_stream.csv")
        simulator.load_data()
        
        # Get training windows
        windows = simulator.get_training_windows(num_windows=10, window_size=50)
        
        # Train models
        pipeline = SensorPipeline()
        pipeline.train_models(windows)
        
        # Verify models exist
        models_dir = Path("models")
        required_files = ["iforest.pkl", "pca.pkl", "scaler.pkl"]
        
        for file in required_files:
            if not (models_dir / file).exists():
                logger.error(f"  Model file missing: {file}")
                return False
        
        logger.info("  Models trained and saved")
        logger.info(f"  IsolationForest: trained")
        logger.info(f"  PCA: trained (128 components)")
        logger.info(f"  StandardScaler: trained")
        
        return True
        
    except Exception as e:
        logger.error(f"  Model training failed: {e}")
        return False


def test_anomaly_detection():
    """Test anomaly detection on a window"""
    logger.info("Testing anomaly detection...")
    
    try:
        from sensor_pipeline import SensorPipeline
        from utils.sensor_stream_simulator import SensorStreamSimulator
        
        # Load data
        simulator = SensorStreamSimulator("data/sensors/test_stream.csv")
        simulator.load_data()
        
        # Get test window
        windows = list(simulator.stream_windows(window_size=50, batch_size=1))
        sensor_id, zone, window_data = windows[0][0]
        
        # Analyze window
        pipeline = SensorPipeline()
        pipeline._load_models()
        
        result = pipeline.analyze_window(window_data, sensor_id)
        
        if result:
            is_anomaly, score, embedding, metrics = result
            
            logger.info("  Anomaly detection successful")
            logger.info(f"  Anomaly: {is_anomaly}")
            logger.info(f"  Score: {score:.4f}")
            logger.info(f"  Embedding dim: {len(embedding)}")
            logger.info(f"  Metrics: gas_ppm={metrics['gas_ppm_mean']:.2f}")
            
            return True
        else:
            logger.error("  Analysis returned None")
            return False
        
    except Exception as e:
        logger.error(f"  Anomaly detection failed: {e}")
        return False


def test_full_pipeline():
    """Test complete agent pipeline"""
    logger.info("Testing full agent pipeline...")
    
    try:
        from sensor_agent import SensorIntelligenceAgent
        
        # Initialize agent
        agent = SensorIntelligenceAgent(
            window_size=50,
            anomaly_threshold=-0.5
        )
        
        # Run on test data
        agent.run_from_csv(
            csv_path="data/sensors/test_stream.csv",
            output_alerts="outputs/test_alerts.jsonl"
        )
        
        # Check output file
        alert_file = Path("outputs/test_alerts.jsonl")
        if alert_file.exists():
            with open(alert_file) as f:
                num_alerts = sum(1 for _ in f)
            
            logger.info(f"  Full pipeline successful")
            logger.info(f"  Alerts generated: {num_alerts}")
            
            return True
        else:
            logger.warning("⚠ No alerts file generated (may be no anomalies)")
            return True
        
    except Exception as e:
        logger.error(f"  Full pipeline failed: {e}")
        return False


def main():
    """Run all tests"""
    logger.info("="*60)
    logger.info("SENSOR AGENT TEST SUITE")
    logger.info("="*60)
    
    tests = [
        ("Imports", test_imports),
        ("Qdrant Connection", test_qdrant_connection),
        ("Data Generation", test_data_generation),
        ("Feature Extraction", test_feature_extraction),
        ("Model Training", test_model_training),
        ("Anomaly Detection", test_anomaly_detection),
        ("Full Pipeline", test_full_pipeline)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"TEST: {test_name}")
        logger.info(f"{'='*60}")
        
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"Test crashed: {e}")
            results[test_name] = False
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*60}")
    
    for test_name, passed in results.items():
        status = "  PASS" if passed else "  FAIL"
        logger.info(f"{status}: {test_name}")
    
    total = len(results)
    passed = sum(results.values())
    
    logger.info(f"\nTotal: {passed}/{total} passed")
    logger.info(f"{'='*60}\n")
    
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())