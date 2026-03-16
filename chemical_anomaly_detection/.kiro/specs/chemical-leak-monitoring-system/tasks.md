# Implementation Plan: Chemical Leak Monitoring System

## Overview

This implementation plan breaks down the multi-agent chemical leak monitoring system into discrete, incremental coding tasks. The system will be built in phases, starting with core infrastructure, then adding each agent sequentially, and finally integrating continual learning and fault tolerance features.

## Tasks

- [x] 1. Set up project structure and core infrastructure
  - Create directory structure for agents, models, database, and tests
  - Set up configuration management with Pydantic models and environment variables
  - Configure logging framework with structured JSON logging
  - Set up Qdrant client and connection management
  - Create base classes for agents with async support
  - _Requirements: 13.1, 13.2, 13.3, 13.4, 12.6, 14.1, 14.4_

- [ ]\* 1.1 Write unit tests for configuration validation
  - Test missing required parameters fail fast
  - Test invalid parameter values are rejected
  - Test environment variable loading
  - _Requirements: 13.2, 13.3_

- [x] 2. Implement Qdrant database schema and collections
  - [x] 2.1 Create baselines collection with multivector schema
    - Define vector configurations for video (512-dim, Cosine), audio (512-dim, Cosine), sensor (128-dim, Euclidean)
    - Define payload schema with timestamp, shift, equipment_id, plant_zone, baseline_type
    - Configure HNSW indexing and payload indexes
    - _Requirements: 3.1, 3.3, 3.4, 4.2, 4.3_
  - [x] 2.2 Create data collection with multivector schema
    - Define vector configurations for all modalities
    - Define payload schema with timestamp, location, camera_id, sensor_ids, plant_zone, is_anomaly, anomaly_scores, cause, severity, operator_feedback
    - Configure indexing for time-window and location-based queries
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_
  - [x] 2.3 Create labeled_anomalies collection
    - Define vector configurations for all modalities
    - Define payload schema with ground_truth_cause, ground_truth_severity, chemical_detected, operator_notes, training_weight
    - _Requirements: 8.1_
  - [x] 2.4 Create response_strategies collection
    - Define vector configuration for incident_embedding (128-dim, Cosine)
    - Define payload schema with incident_id, cause, severity, plant_zone, successful_actions, failed_actions, effectiveness_score, response_time_seconds, outcome
    - Configure indexing for severity, plant_zone, effectiveness_score
    - _Requirements: 7.1, 7.2, 7.3, 7.4_

- [ ]\* 2.5 Write property test for multivector storage
  - **Property 10: Multivector storage**
  - **Validates: Requirements 3.1, 3.3**

- [ ]\* 2.6 Write property test for payload completeness
  - **Property 11: Payload completeness**
  - **Validates: Requirements 3.2**

- [ ]\* 2.7 Write property test for time-window query correctness
  - **Property 12: Time-window query correctness**
  - **Validates: Requirements 3.5**

- [ ]\* 2.8 Write property test for location-based query correctness
  - **Property 13: Location-based query correctness**
  - **Validates: Requirements 3.6**

- [x] 3. Implement sensor embedding adapter and processor
  - [x] 3.1 Create SensorEmbeddingAdapter neural network
    - Implement PyTorch module with Input(5) -> Dense(64, ReLU) -> Dense(128, tanh) architecture
    - Add forward pass with normalization
    - Define normalization parameters (means and stds from training data)
    - _Requirements: 2.3, 2.4_
  - [x] 3.2 Create SensorProcessor class
    - Implement async process method for sensor readings
    - Add Pydantic validation for sensor data (SensorReading model)
    - Implement noise filtering for outliers (>3 std devs)
    - _Requirements: 2.3, 2.4, 10.1, 14.4, 14.5_

- [ ]\* 3.3 Write property test for sensor embedding transformation
  - **Property 4: Sensor embedding transformation**
  - **Validates: Requirements 2.3, 2.8**

- [ ]\* 3.4 Write property test for sensor normalization
  - **Property 5: Sensor normalization**
  - **Validates: Requirements 2.4**

- [ ]\* 3.5 Write property test for embedding dimensionality
  - **Property 6: Embedding dimensionality (sensor)**
  - **Validates: Requirements 2.3**

- [ ]\* 3.6 Write property test for noise filtering
  - **Property 42: Noise filtering**
  - **Validates: Requirements 10.1**

- [x] 4. Implement video processor
  - [x] 4.1 Create VideoProcessor class with MobileNetV3-Small
    - Load pre-trained MobileNetV3-Small model from torchvision
    - Implement async process_frame method
    - Add preprocessing (resize to 224x224, normalize)
    - Extract features from penultimate layer (1024-dim)
    - _Requirements: 2.1_
  - [x] 4.2 Add error handling and fallback for video processing
    - Implement retry logic with timeout (1 second max)
    - Return None for failed frames with logging
    - _Requirements: 9.1, 14.2, 14.3_

- [ ]\* 4.3 Write property test for embedding dimensionality
  - **Property 6: Embedding dimensionality (video)**
  - **Validates: Requirements 2.1**

- [ ]\* 4.4 Write unit tests for video processing edge cases
  - Test invalid frame shapes are rejected
  - Test corrupted frames are handled gracefully
  - _Requirements: 14.5_

- [x] 5. Implement audio processor
  - [x] 5.1 Create AudioProcessor class with PANNs CNN14
    - Load pre-trained PANNs CNN14 model
    - Implement async process_audio method
    - Add spectral preprocessing (mel-spectrogram with 128 bins)
    - Extract embeddings (2048-dim)
    - _Requirements: 2.2, 2.5_
  - [x] 5.2 Add error handling and fallback for audio processing
    - Implement retry logic with timeout
    - Return None for failed audio windows with logging
    - _Requirements: 9.2, 14.2, 14.3_

- [ ]\* 5.3 Write property test for spectral preprocessing
  - **Property 7: Spectral preprocessing**
  - **Validates: Requirements 2.5**

- [ ]\* 5.4 Write property test for embedding dimensionality
  - **Property 6: Embedding dimensionality (audio)**
  - **Validates: Requirements 2.2**

- [x] 6. Implement Input Collection Agent
  - [x] 6.1 Create EmbeddingGenerator class
    - Implement async generate method for multimodal embeddings
    - Add parallel processing for video, audio, and sensor data using asyncio.gather
    - Create MultimodalEmbedding dataclass with metadata
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 11.2_
  - [x] 6.2 Create InputCollectionAgent class
    - Implement main processing loop with 1-second intervals
    - Add direct pass to Anomaly Detection Agent (no intermediate storage)
    - Implement queue-based buffering for backpressure handling
    - _Requirements: 1.5, 1.6_

- [ ]\* 6.3 Write property test for parallel processing
  - **Property 3: Parallel processing**
  - **Validates: Requirements 11.2**

- [ ]\* 6.4 Write property test for direct embedding pass
  - **Property 8: Direct embedding pass**
  - **Validates: Requirements 1.5**

- [ ]\* 6.5 Write property test for end-to-end latency
  - **Property 1: End-to-end processing latency**
  - **Validates: Requirements 1.4, 1.6, 11.5**

- [ ]\* 6.6 Write property test for graceful degradation
  - **Property 37: Graceful degradation with partial modalities**
  - **Validates: Requirements 9.1, 9.2, 9.3, 9.6**

- [x] 7. Checkpoint - Ensure embedding generation works end-to-end
  - Ensure all tests pass, ask the user if questions arise.

- [x] 8. Implement baseline management
  - [x] 8.1 Create BaselineManager class
    - Implement method to generate baselines from normal operating data
    - Add shift-specific and equipment-specific baseline tagging
    - Implement rolling baseline updates with drift detection
    - Store baselines in Qdrant baselines collection
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

- [ ]\* 8.2 Write property test for baseline source validation
  - **Property 14: Baseline source validation**
  - **Validates: Requirements 4.1**

- [ ]\* 8.3 Write property test for shift-specific baseline tagging
  - **Property 15: Shift-specific baseline tagging**
  - **Validates: Requirements 4.2**

- [ ]\* 8.4 Write property test for equipment-specific baseline tagging
  - **Property 16: Equipment-specific baseline tagging**
  - **Validates: Requirements 4.3**

- [ ]\* 8.5 Write property test for rolling baseline updates
  - **Property 17: Rolling baseline updates**
  - **Validates: Requirements 4.4**

- [x] 9. Implement adaptive threshold manager
  - [x] 9.1 Create AdaptiveThresholdManager class
    - Initialize with per-modality thresholds (video: 0.7, audio: 0.65, sensor: 2.5)
    - Implement update_thresholds method with exponential moving average
    - Add false positive/negative tracking with sliding windows
    - Implement is_anomaly method with multi-modality voting
    - _Requirements: 4.6, 5.3, 5.4_

- [ ]\* 9.2 Write property test for adaptive threshold non-static
  - **Property 18: Adaptive threshold non-static**
  - **Validates: Requirements 4.6, 5.8**

- [ ]\* 9.3 Write property test for threshold adaptation direction
  - **Property 19: Threshold adaptation direction**
  - **Validates: Requirements 5.4, 10.3**

- [x] 10. Implement similarity search engine
  - [x] 10.1 Create SimilaritySearchEngine class
    - Implement async search_baselines method with per-modality queries
    - Add compute_anomaly_scores method (min distance to baseline)
    - Support filtered searches by shift, equipment_id, plant_zone
    - _Requirements: 5.1, 5.2, 5.7_

- [ ]\* 10.2 Write property test for per-modality anomaly scoring
  - **Property 20: Per-modality anomaly scoring**
  - **Validates: Requirements 5.7**

- [x] 11. Implement Anomaly Detection Agent
  - [x] 11.1 Create StorageManager class
    - Implement async store_embedding method
    - Add logic to store with is_anomaly flag and anomaly_scores
    - Trigger Cause Detection Agent for anomalies
    - _Requirements: 5.5, 5.6_
  - [x] 11.2 Create AnomalyDetectionAgent class
    - Implement main detection loop receiving embeddings from Input Collection Agent
    - Integrate SimilaritySearchEngine and AdaptiveThresholdManager
    - Add multi-modality confirmation for high-severity anomalies
    - Implement temporal confirmation for borderline anomalies (3 consecutive windows)
    - _Requirements: 5.1, 5.2, 5.3, 10.2, 10.4_

- [ ]\* 11.3 Write property test for anomaly flagging and storage
  - **Property 21: Anomaly flagging and storage**
  - **Validates: Requirements 5.5**

- [ ]\* 11.4 Write property test for normal data storage
  - **Property 22: Normal data storage**
  - **Validates: Requirements 5.6**

- [ ]\* 11.5 Write property test for multi-modality confirmation
  - **Property 23: Multi-modality confirmation for high severity**
  - **Validates: Requirements 10.2**

- [ ]\* 11.6 Write property test for temporal confirmation
  - **Property 24: Temporal confirmation for borderline anomalies**
  - **Validates: Requirements 10.4**

- [x] 12. Checkpoint - Ensure anomaly detection works end-to-end
  - Ensure all tests pass, ask the user if questions arise.

- [x] 13. Implement Cause Detection Agent
  - [x] 13.1 Create CauseInferenceEngine class
    - Implement async infer_cause method with similarity search
    - Add \_search_similar_incidents method querying labeled_anomalies collection
    - Implement \_aggregate_causes with weighted voting
    - Add \_generate_explanation method with references to similar incidents
    - _Requirements: 6.1, 6.2, 6.3, 6.5, 6.6_
  - [x] 13.2 Create SeverityClassifier class
    - Implement classify_severity method
    - Add \_compute_severity_score with gas concentration, modality count, and cause-specific factors
    - Classify as mild (<0.5), medium (0.5-0.8), or high (>0.8)
    - _Requirements: 6.4_
  - [x] 13.3 Create CauseDetectionAgent class
    - Integrate CauseInferenceEngine and SeverityClassifier
    - Implement main processing loop receiving anomalies from Anomaly Detection Agent
    - Route to appropriate Risk Response Agent based on severity
    - _Requirements: 6.1, 6.4_

- [ ]\* 13.4 Write property test for cause identification completeness
  - **Property 25: Cause identification completeness**
  - **Validates: Requirements 6.2**

- [ ]\* 13.5 Write property test for severity classification validity
  - **Property 26: Severity classification validity**
  - **Validates: Requirements 6.4**

- [ ]\* 13.6 Write property test for explainability requirement
  - **Property 27: Explainability requirement**
  - **Validates: Requirements 6.5**

- [ ]\* 13.7 Write property test for metadata incorporation
  - **Property 28: Metadata incorporation**
  - **Validates: Requirements 6.6**

- [x] 14. Implement MSDS and SOP integration
  - [x] 14.1 Create MSDSIntegration class
    - Load MSDS database from JSON/SQLite
    - Implement get_chemical_info method for Chlorine, Ammonia, MIC, acidic/toxic gases
    - Return ChemicalInfo with exposure limits, emergency procedures, PPE requirements
    - _Requirements: 7.5_
  - [x] 14.2 Create SOPIntegration class
    - Load SOP database organized by plant_zone and severity
    - Implement get_procedures method
    - Return zone-specific and severity-specific procedures
    - _Requirements: 7.6_

- [x] 15. Implement Risk Response Agents
  - [x] 15.1 Create ResponseStrategyEngine class
    - Implement async get_response_strategy method
    - Add \_search_similar_responses querying response_strategies collection
    - Implement \_aggregate_actions with weighted voting by similarity and effectiveness
    - Integrate MSDS and SOP information
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_
  - [x] 15.2 Create MildResponseAgent class
    - Initialize with ResponseStrategyEngine
    - Implement execute_response method
    - Execute recommended actions and log incident
    - _Requirements: 7.1, 7.7_
  - [x] 15.3 Create MediumResponseAgent class
    - Initialize with ResponseStrategyEngine
    - Implement execute_response method with MSDS integration
    - Execute recommended actions and log incident
    - _Requirements: 7.2, 7.7_
  - [x] 15.4 Create HighResponseAgent class
    - Initialize with ResponseStrategyEngine
    - Implement execute_response method with emergency alarm trigger
    - Execute recommended actions, SOPs, and notify authorities
    - _Requirements: 7.3, 7.7_

- [ ]\* 15.5 Write property test for severity-based routing
  - **Property 9: Severity-based routing**
  - **Validates: Requirements 7.1, 7.2, 7.3**

- [ ]\* 15.6 Write property test for MSDS integration
  - **Property 29: MSDS integration**
  - **Validates: Requirements 7.4**

- [ ]\* 15.7 Write property test for SOP integration
  - **Property 30: SOP integration**
  - **Validates: Requirements 7.5**

- [ ]\* 15.8 Write property test for response action logging
  - **Property 31: Response action logging**
  - **Validates: Requirements 7.6**

- [x] 16. Checkpoint - Ensure cause detection and response work end-to-end
  - Ensure all tests pass, ask the user if questions arise.

- [x] 17. Implement continual learning system
  - [x] 17.1 Create LabeledAnomalyStore class
    - Implement store_labeled_anomaly method
    - Add operator feedback tracking (confirmed/false_positive/false_negative)
    - Store in labeled_anomalies collection with ground truth labels
    - _Requirements: 8.1_
  - [x] 17.2 Create RetrainingManager class
    - Implement trigger logic based on labeled data count threshold (1000 samples)
    - Add retrain_embedding_adapters method
    - Implement catastrophic forgetting prevention (replay buffer with old samples)
    - Update adaptive thresholds based on new validation data
    - _Requirements: 8.2, 8.3, 8.4_
  - [x] 17.3 Create ModelVersionManager class
    - Implement version tracking for embedding models and baseline collections
    - Add deploy_new_version method with backward compatibility checks
    - Store version metadata in Qdrant
    - _Requirements: 8.5, 8.6_

- [ ]\* 17.4 Write property test for labeled anomaly storage
  - **Property 32: Labeled anomaly storage**
  - **Validates: Requirements 8.1**

- [ ]\* 17.5 Write property test for retraining trigger
  - **Property 33: Retraining trigger**
  - **Validates: Requirements 8.2**

- [ ]\* 17.6 Write property test for catastrophic forgetting prevention
  - **Property 34: Catastrophic forgetting prevention**
  - **Validates: Requirements 8.4**

- [ ]\* 17.7 Write property test for model versioning
  - **Property 35: Model versioning**
  - **Validates: Requirements 8.5**

- [ ]\* 17.8 Write property test for backward compatibility
  - **Property 36: Backward compatibility**
  - **Validates: Requirements 8.6**

- [ ] 18. Implement fault tolerance and error handling
  - [ ] 18.1 Add retry logic with exponential backoff
    - Create retry_with_backoff utility function
    - Apply to Qdrant operations (max 3 retries)
    - _Requirements: 14.2_
  - [ ] 18.2 Add circuit breaker for Qdrant connections
    - Create CircuitBreaker class
    - Implement state management (closed/open/half_open)
    - Apply to database operations
    - _Requirements: 9.5_
  - [ ] 18.3 Implement embedding queue for database unavailability
    - Create in-memory queue with max depth (1000)
    - Add periodic retry logic for queued embeddings
    - Persist queue to disk if memory limit reached
    - _Requirements: 9.5_
  - [ ] 18.4 Add confidence adjustment for partial modalities
    - Implement confidence reduction factor based on missing modalities
    - Update anomaly detection logic to adjust confidence scores
    - _Requirements: 9.7_
  - [ ] 18.5 Implement buffering for high network latency
    - Add latency monitoring
    - Switch to batch mode when latency > 2 seconds
    - Buffer data without dropping
    - _Requirements: 9.4_

- [ ]\* 18.6 Write property test for confidence adjustment
  - **Property 38: Confidence adjustment for partial data**
  - **Validates: Requirements 9.7**

- [ ]\* 18.7 Write property test for buffering under high latency
  - **Property 39: Buffering under high latency**
  - **Validates: Requirements 9.4**

- [ ]\* 18.8 Write property test for queueing during database unavailability
  - **Property 40: Queueing during database unavailability**
  - **Validates: Requirements 9.5**

- [ ]\* 18.9 Write property test for agent isolation
  - **Property 41: Agent isolation**
  - **Validates: Requirements 14.3**

- [ ] 19. Implement logging and metrics
  - [ ] 19.1 Create structured logging system
    - Configure JSON logging with required fields (timestamp, level, agent_name, message)
    - Add log handlers for processing, anomaly, and error logs
    - Implement context managers for log enrichment
    - _Requirements: 12.1, 12.2, 12.3, 12.6_
  - [ ] 19.2 Create metrics collection system
    - Implement metrics for throughput, latency, error_rate, queue_depth
    - Add Prometheus-compatible metrics exporter
    - Create metrics dashboard configuration
    - _Requirements: 12.4, 12.5_

- [ ]\* 19.3 Write property test for processing log completeness
  - **Property 44: Processing log completeness**
  - **Validates: Requirements 12.1**

- [ ]\* 19.4 Write property test for anomaly log completeness
  - **Property 45: Anomaly log completeness**
  - **Validates: Requirements 12.2**

- [ ]\* 19.5 Write property test for error log completeness
  - **Property 46: Error log completeness**
  - **Validates: Requirements 12.3**

- [ ]\* 19.6 Write property test for structured logging format
  - **Property 47: Structured logging format**
  - **Validates: Requirements 12.6**

- [ ]\* 19.7 Write property test for metrics exposure
  - **Property 48: Metrics exposure**
  - **Validates: Requirements 12.4**

- [ ] 20. Implement configuration validation and type safety
  - [ ] 20.1 Add comprehensive type hints
    - Review all functions and add type hints for parameters and return values
    - Use typing module for complex types (Optional, Dict, List, etc.)
    - _Requirements: 14.1_
  - [ ] 20.2 Add input validation with Pydantic
    - Create Pydantic models for all data structures
    - Add validators for ranges and constraints
    - _Requirements: 14.4, 14.5_
  - [ ] 20.3 Implement configuration validation
    - Add startup validation for all required environment variables
    - Implement fail-fast behavior with descriptive errors
    - Verify no hardcoded environment-specific values
    - _Requirements: 13.1, 13.2, 13.3, 13.5_

- [ ]\* 20.4 Write property test for environment variable configuration
  - **Property 49: Environment variable configuration**
  - **Validates: Requirements 13.1**

- [ ]\* 20.5 Write property test for configuration validation
  - **Property 50: Configuration validation**
  - **Validates: Requirements 13.2, 13.3**

- [ ]\* 20.6 Write property test for no hardcoded values
  - **Property 51: No hardcoded values**
  - **Validates: Requirements 13.5**

- [ ]\* 20.7 Write property test for type hint completeness
  - **Property 52: Type hint completeness**
  - **Validates: Requirements 14.1**

- [ ]\* 20.8 Write property test for exception handling
  - **Property 53: Exception handling**
  - **Validates: Requirements 14.2**

- [ ]\* 20.9 Write property test for input validation
  - **Property 43: Input validation**
  - **Validates: Requirements 14.5**

- [x] 21. Create Docker containerization
  - [x] 21.1 Create Dockerfile for each agent
    - Write Dockerfiles with Python 3.10+ base image
    - Include all dependencies from requirements.txt
    - Add health check endpoints
    - _Requirements: 15.1, 15.2, 15.4_
  - [x] 21.2 Create Docker Compose configuration
    - Define services for all agents and Qdrant
    - Configure networking and volumes
    - Add environment variable templates
    - Implement graceful shutdown handlers
    - _Requirements: 15.3, 15.5_

- [ ] 22. Implement performance benchmarking
  - [ ] 22.1 Create benchmark suite
    - Implement latency benchmarks (end-to-end, Qdrant query, embedding generation per modality)
    - Implement throughput benchmarks per modality
    - Implement memory usage benchmarks per agent
    - _Requirements: 18.1, 18.2, 18.3, 18.4, 18.5_
  - [ ] 22.2 Add performance validation
    - Verify 1-second processing intervals
    - Verify sub-2-second end-to-end latency
    - Generate performance report
    - _Requirements: 18.6_

- [ ]\* 22.3 Write property test for throughput maintenance
  - **Property 2: Throughput maintenance**
  - **Validates: Requirements 11.4**

- [ ]\* 22.4 Write property test for latency benchmarks
  - **Property 54: Latency benchmarks**
  - **Validates: Requirements 18.1, 18.3, 18.4**

- [ ]\* 22.5 Write property test for throughput benchmarks
  - **Property 55: Throughput benchmarks**
  - **Validates: Requirements 18.2**

- [ ]\* 22.6 Write property test for memory benchmarks
  - **Property 56: Memory benchmarks**
  - **Validates: Requirements 18.5**

- [ ]\* 22.7 Write property test for performance target compliance
  - **Property 57: Performance target compliance**
  - **Validates: Requirements 18.6**

- [x] 23. Create sample data and seed databases
  - [x] 23.1 Generate baseline embeddings from normal_sensor_data.csv
    - Process existing sensor data to create normal baselines
    - Generate shift-specific and equipment-specific baselines
    - Store in baselines collection
    - _Requirements: 4.1, 4.2, 4.3_
  - [x] 23.2 Create sample labeled anomalies
    - Generate synthetic anomaly embeddings for common causes
    - Add ground truth labels for gas_plume, audio_anomaly, pressure_spike, valve_malfunction, ppe_violation, human_panic
    - Store in labeled_anomalies collection
    - _Requirements: 6.2, 8.1_
  - [x] 23.3 Create sample response strategies
    - Generate response strategy embeddings for each severity level
    - Add successful_actions and effectiveness_scores
    - Store in response_strategies collection
    - _Requirements: 7.1, 7.2, 7.3_
  - [x] 23.4 Create MSDS and SOP databases
    - Create JSON database with MSDS info for Chlorine, Ammonia, MIC, acidic/toxic gases
    - Create JSON database with SOPs organized by plant_zone and severity
    - _Requirements: 7.4, 7.5_

- [x] 24. Integration testing and validation
  - [x] 24.1 Create end-to-end integration test
    - Test complete flow from video/audio/sensor input to response execution
    - Verify all agents communicate correctly
    - Validate data flows through all collections
    - _Requirements: All_
  - [x] 24.2 Test with factory_video.mp4 and anomalous_sensor.csv
    - Process real video and sensor data
    - Verify anomaly detection on known anomalous data
    - Validate cause inference and response selection
    - _Requirements: All_

- [ ] 25. Documentation and deployment guide
  - [ ] 25.1 Create README with setup instructions
    - Document environment variable configuration
    - Provide Docker Compose quickstart
    - Include performance benchmarking instructions
    - _Requirements: 13.1, 15.3_
  - [ ] 25.2 Document model selection justifications
    - Document video model choice (MobileNetV3-Small)
    - Document audio model choice (PANNs CNN14)
    - Document embedding fusion strategy (late fusion)
    - Document distance metrics per modality
    - _Requirements: 17.1, 17.2, 17.3, 17.4_
  - [ ] 25.3 Create architecture diagram
    - Generate Mermaid diagram from design document
    - Include in README
    - _Requirements: All_

- [ ] 26. Final checkpoint - Complete system validation
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties (minimum 100 iterations each)
- Unit tests validate specific examples and edge cases
- Integration tests validate end-to-end workflows
