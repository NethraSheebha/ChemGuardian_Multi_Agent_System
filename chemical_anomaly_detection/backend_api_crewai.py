"""
FastAPI Backend for Chemical Leak Monitoring System - CrewAI Version
Uses CrewAI agents with Ollama LLM for truly agentic behavior
"""

import asyncio
import logging
import base64
import io
import time
import os
from datetime import datetime
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

import cv2
import librosa
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.database.client_factory import create_qdrant_client
from src.models.video_processor import VideoProcessor
from src.models.audio_processor import AudioProcessor
from src.models.sensor_processor import SensorProcessor
from src.models.sensor_adapter import SensorEmbeddingAdapter

# Import CrewAI agents instead of Pythonic agents
from src.crewai_agents import (
    InputCollectionCrew,
    AnomalyDetectionCrew,
    CauseDetectionCrew,
    MildResponseCrew,
    MediumResponseCrew,
    HighResponseCrew
)

# Import supporting components
from src.agents.similarity_search_engine import SimilaritySearchEngine
from src.agents.adaptive_threshold_manager import AdaptiveThresholdManager
from src.agents.storage_manager import StorageManager
from src.agents.cause_inference_engine import CauseInferenceEngine
from src.agents.severity_classifier import SeverityClassifier
from src.agents.response_strategy_engine import ResponseStrategyEngine
from src.integrations.msds_integration import MSDSIntegration
from src.integrations.sop_integration import SOPIntegration

# Import Ollama LLM
try:
    from langchain_community.llms import Ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logging.warning("langchain_community not installed. LLM reasoning will be disabled.")


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Global state for agents
class AppState:
    input_crew: Optional[InputCollectionCrew] = None
    anomaly_crew: Optional[AnomalyDetectionCrew] = None
    cause_crew: Optional[CauseDetectionCrew] = None
    mild_crew: Optional[MildResponseCrew] = None
    medium_crew: Optional[MediumResponseCrew] = None
    high_crew: Optional[HighResponseCrew] = None
    
    # Data sources
    video_cap: Optional[cv2.VideoCapture] = None
    audio_data: Optional[tuple] = None
    sensor_df: Optional[pd.DataFrame] = None
    
    # Monitoring state
    monitoring: bool = False
    current_sample: int = 0
    samples_processed: int = 0
    anomalies_detected: int = 0
    start_time: Optional[float] = None
    frame_skip: int = 5  # Skip frames to reach blast faster (process every 5th frame)


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize CrewAI agents with Ollama LLM on startup"""
    logger.info("🤖 Initializing CrewAI monitoring system with Ollama LLM...")
    
    # Initialize Ollama LLM
    llm = None
    if OLLAMA_AVAILABLE:
        try:
            ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
            ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            
            logger.info(f"🦙 Connecting to Ollama: {ollama_base_url} (model: {ollama_model})")
            llm = Ollama(
                model=ollama_model,
                base_url=ollama_base_url,
                temperature=0.1  # Low temperature for safety-critical decisions
            )
            logger.info("✅ Ollama LLM initialized successfully")
        except Exception as e:
            logger.warning(f"⚠️  Failed to initialize Ollama: {e}")
            logger.warning("⚠️  Continuing without LLM reasoning")
            llm = None
    else:
        logger.warning("⚠️  langchain_community not installed. Install with: pip install langchain-community")
    
    # Initialize Qdrant client
    client = create_qdrant_client()
    
    # Initialize processors with increased timeouts for cloud processing
    video_proc = VideoProcessor(device="cpu", timeout=5.0)
    audio_proc = AudioProcessor(
        device="cpu",
        checkpoint_path="C:/Users/maryj/Downloads/Cnn14_mAP=0.431.pth",
        timeout=5.0
    )
    sensor_adapter = SensorEmbeddingAdapter()
    sensor_proc = SensorProcessor(adapter=sensor_adapter, enable_noise_filtering=False)
    
    # Initialize CrewAI Input Collection Crew
    state.input_crew = InputCollectionCrew(
        video_processor=video_proc,
        audio_processor=audio_proc,
        sensor_processor=sensor_proc,
        processing_interval=5.0
    )
    
    # Initialize CrewAI Anomaly Detection Crew
    state.anomaly_crew = AnomalyDetectionCrew(
        qdrant_client=client,
        similarity_search_engine=SimilaritySearchEngine(client, top_k=5, search_timeout=60.0),  # Increased timeout for cloud
        adaptive_threshold_manager=AdaptiveThresholdManager(
            video_threshold=0.7,
            audio_threshold=0.65,
            sensor_threshold=2.5
        ),
        storage_manager=StorageManager(client)
    )
    
    # Initialize CrewAI Cause Detection Crew with Ollama
    cause_engine = CauseInferenceEngine(client)
    severity_classifier = SeverityClassifier()
    state.cause_crew = CauseDetectionCrew(
        qdrant_client=client,
        cause_inference_engine=cause_engine,
        severity_classifier=severity_classifier,
        llm=llm  # Pass Ollama LLM for reasoning
    )
    
    # Initialize MSDS and SOP integrations
    msds_database_path = os.getenv("MSDS_DATABASE_PATH", "data/msds_database.json")
    sop_database_path = os.getenv("SOP_DATABASE_PATH", "data/sop_database.json")
    
    msds_integration = MSDSIntegration(msds_database_path)
    sop_integration = SOPIntegration(sop_database_path)
    
    # Initialize CrewAI Response Crews with Ollama
    response_engine = ResponseStrategyEngine(client, msds_integration, sop_integration)
    state.mild_crew = MildResponseCrew(client, response_engine, llm=llm)
    state.medium_crew = MediumResponseCrew(client, response_engine, llm=llm)
    state.high_crew = HighResponseCrew(client, response_engine, llm=llm)
    
    logger.info("✅ CrewAI monitoring system initialized")
    if llm:
        logger.info("🦙 All agents are using Ollama LLM for reasoning")
    else:
        logger.info("⚠️  Agents running without LLM (rule-based fallback)")
    
    yield
    
    # Cleanup
    if state.video_cap:
        state.video_cap.release()


app = FastAPI(lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint"""
    llm_status = "Ollama (llama3.2:1b)" if OLLAMA_AVAILABLE else "Disabled"
    return {
        "service": "Chemical Leak Monitoring API - CrewAI Version",
        "version": "2.0.0",
        "status": "running",
        "agent_type": "CrewAI",
        "llm": llm_status
    }


@app.post("/api/start")
async def start_monitoring(
    video_path: str = "anomalous_1.mp4",
    audio_path: str = "anomalous_audio.wav",
    sensor_path: str = "anomalous_sensor.csv"
):
    """Start monitoring with specified data files"""
    try:
        logger.info(f"Loading data files: video={video_path}, audio={audio_path}, sensor={sensor_path}")
        
        # Load video
        state.video_cap = cv2.VideoCapture(video_path)
        if not state.video_cap.isOpened():
            raise HTTPException(status_code=400, detail=f"Failed to open video: {video_path}")
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=32000)
        state.audio_data = (audio, sr)
        
        # Load sensor data
        state.sensor_df = pd.read_csv(sensor_path)
        
        # Reset state
        state.monitoring = True
        state.current_sample = 0
        state.samples_processed = 0
        state.anomalies_detected = 0
        state.start_time = time.time()
        
        logger.info("✅ Monitoring started with CrewAI agents")
        
        return {
            "status": "started",
            "video_frames": int(state.video_cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "audio_duration": len(audio) / sr,
            "sensor_samples": len(state.sensor_df),
            "agent_type": "CrewAI"
        }
        
    except Exception as e:
        logger.error(f"Failed to start monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/stop")
async def stop_monitoring():
    """Stop monitoring"""
    state.monitoring = False
    logger.info("Monitoring stopped")
    return {"status": "stopped"}


@app.get("/api/status")
async def get_status():
    """Get monitoring status"""
    uptime = int(time.time() - state.start_time) if state.start_time else 0
    
    llm_enabled = OLLAMA_AVAILABLE
    llm_model = os.getenv("OLLAMA_MODEL", "llama3.2:1b") if llm_enabled else "None"
    
    return {
        "monitoring": state.monitoring,
        "samples_processed": state.samples_processed,
        "anomalies_detected": state.anomalies_detected,
        "uptime_seconds": uptime,
        "agent_type": "CrewAI",
        "llm_enabled": llm_enabled,
        "llm_model": llm_model
    }


@app.get("/api/data")
async def get_data():
    """Get current monitoring data"""
    if not state.monitoring:
        return {
            "monitoring": False,
            "message": "Monitoring not started"
        }
    
    try:
        # Start timing
        start_time = time.time()
        
        # Get current sample (with frame skipping)
        sample_idx = state.current_sample * state.frame_skip
        
        # Read video frame
        state.video_cap.set(cv2.CAP_PROP_POS_FRAMES, sample_idx)
        ret, frame = state.video_cap.read()
        if not ret:
            # Loop back to start
            state.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            state.current_sample = 0
            sample_idx = 0
            ret, frame = state.video_cap.read()
        
        # Get audio window
        audio, sr = state.audio_data
        audio_start = (sample_idx * sr) % len(audio)
        audio_end = min(audio_start + sr, len(audio))
        audio_window = audio[audio_start:audio_end]
        
        # Get sensor reading
        sensor_idx = sample_idx % len(state.sensor_df)
        sensor_row = state.sensor_df.iloc[sensor_idx]
        
        # Convert to proper format with timestamp
        current_timestamp = datetime.utcnow()
        sensor_reading = {
            "timestamp": current_timestamp,
            "temperature_celsius": float(sensor_row["temperature_celsius"]),
            "pressure_bar": float(sensor_row["pressure_bar"]),
            "gas_concentration_ppm": float(sensor_row["gas_concentration_ppm"]),
            "vibration_mm_s": float(sensor_row["vibration_mm_s"]),
            "flow_rate_lpm": float(sensor_row["flow_rate_lpm"])
        }
        
        # Timing: Embedding generation
        embedding_start = time.time()
        
        # Process with CrewAI Input Collection Crew
        embedding = await state.input_crew.process_data_point(
            video_frame=frame,
            audio_data=(audio_window, sr),
            sensor_reading=sensor_reading,
            metadata={"plant_zone": "Zone_A", "shift": "morning", "camera_id": "CAM-01"}
        )
        
        embedding_time = (time.time() - embedding_start) * 1000  # Convert to ms
        
        if embedding:
            # Timing: Qdrant search
            qdrant_start = time.time()
            
            # Detect anomaly with CrewAI Anomaly Detection Crew
            anomaly_result = await state.anomaly_crew.detect_anomaly(
                embedding=embedding,
                shift="morning",
                plant_zone="Zone_A"
            )
            
            qdrant_time = (time.time() - qdrant_start) * 1000  # Convert to ms
            
            # Log per-modality decisions for debugging
            logger.info(f"Per-modality decisions: video={anomaly_result.per_modality_decisions.get('video', False)}, audio={anomaly_result.per_modality_decisions.get('audio', False)}, sensor={anomaly_result.per_modality_decisions.get('sensor', False)}")
            
            # If anomaly detected, analyze with CrewAI Cause Detection Crew
            if anomaly_result.is_anomaly:
                state.anomalies_detected += 1
                
                # Analyze cause and severity
                cause_result = await state.cause_crew.analyze_anomaly(anomaly_result)
                
                # Route to appropriate CrewAI Response Crew
                severity = cause_result.severity
                if severity == "mild":
                    response = await state.mild_crew.execute_response(cause_result)
                elif severity == "medium":
                    response = await state.medium_crew.execute_response(cause_result)
                else:  # high
                    response = await state.high_crew.execute_response(cause_result)
                
                # Prepare response data in frontend format
                total_time = (time.time() - start_time) * 1000  # Total time in ms
                
                result_data = {
                    "monitoring": True,
                    "video": {
                        "frame": None,  # Will be added below
                        "anomaly": anomaly_result.per_modality_decisions.get("video", False),  # Only video anomaly
                        "location": "Zone_A",
                        "camera_id": "CAM-01",
                        "timestamp": current_timestamp.isoformat()
                    },
                    "audio": {
                        "waveform": audio_window.tolist()[:100],  # First 100 samples
                        "anomaly": anomaly_result.per_modality_decisions.get("audio", False),
                        "metrics": {
                            "peak": float(np.max(np.abs(audio_window))),
                            "rms": float(np.sqrt(np.mean(audio_window**2))),
                            "zcr": 0.0
                        }
                    },
                    "sensors": {
                        "values": {
                            "temperature": sensor_reading["temperature_celsius"],
                            "pressure": sensor_reading["pressure_bar"],
                            "gas_concentration": sensor_reading["gas_concentration_ppm"],
                            "vibration": sensor_reading["vibration_mm_s"],
                            "flow_rate": sensor_reading["flow_rate_lpm"],
                            "valve_status": "UNSTABLE" if anomaly_result.per_modality_decisions.get("sensor", False) else "STABLE"
                        },
                        "trends": {
                            "temperature": "UP" if sensor_reading["temperature_celsius"] > 80 else "STABLE",
                            "pressure": "UP" if sensor_reading["pressure_bar"] > 10 else "STABLE",
                            "gas_concentration": "UP" if sensor_reading["gas_concentration_ppm"] > 400 else "STABLE",
                            "vibration": "UP" if sensor_reading["vibration_mm_s"] > 5 else "STABLE",
                            "flow_rate": "DOWN" if sensor_reading["flow_rate_lpm"] < 100 else "STABLE"
                        },
                        "anomaly": anomaly_result.per_modality_decisions.get("sensor", False)
                    },
                    "alert": {
                        "system_status": severity.upper(),  # MILD, MEDIUM, HIGH
                        "severity": severity.upper(),
                        "risk_level": severity.upper(),
                        "cause": cause_result.cause_analysis.primary_cause,
                        "confidence": cause_result.cause_analysis.confidence,
                        "explanation": cause_result.cause_analysis.explanation,
                        "affected_modalities": [
                            mod for mod, is_anom in anomaly_result.per_modality_decisions.items() if is_anom
                        ]
                    },
                    "actions": response.get("actions_executed", []),
                    "timestamp": datetime.utcnow().isoformat(),
                    "system_metrics": {
                        "samples_processed": state.samples_processed + 1,
                        "anomalies_detected": state.anomalies_detected,
                        "anomaly_rate": (state.anomalies_detected / (state.samples_processed + 1)) * 100,
                        "uptime_seconds": int(time.time() - state.start_time)
                    },
                    "latency_ms": {
                        "total": int(total_time),
                        "qdrant": int(qdrant_time),
                        "embedding": int(embedding_time)
                    }
                }
            else:
                total_time = (time.time() - start_time) * 1000
                
                result_data = {
                    "monitoring": True,
                    "video": {
                        "frame": None,
                        "anomaly": False,
                        "location": "Zone_A",
                        "camera_id": "CAM-01",
                        "timestamp": current_timestamp.isoformat()
                    },
                    "audio": {
                        "waveform": audio_window.tolist()[:100],
                        "anomaly": False,
                        "metrics": {
                            "peak": float(np.max(np.abs(audio_window))),
                            "rms": float(np.sqrt(np.mean(audio_window**2))),
                            "zcr": 0.0
                        }
                    },
                    "sensors": {
                        "values": {
                            "temperature": sensor_reading["temperature_celsius"],
                            "pressure": sensor_reading["pressure_bar"],
                            "gas_concentration": sensor_reading["gas_concentration_ppm"],
                            "vibration": sensor_reading["vibration_mm_s"],
                            "flow_rate": sensor_reading["flow_rate_lpm"],
                            "valve_status": "STABLE"
                        },
                        "trends": {
                            "temperature": "STABLE",
                            "pressure": "STABLE",
                            "gas_concentration": "STABLE",
                            "vibration": "STABLE",
                            "flow_rate": "STABLE"
                        },
                        "anomaly": False
                    },
                    "alert": {
                        "system_status": "NORMAL",
                        "severity": "NORMAL",
                        "risk_level": "NORMAL",
                        "cause": None,
                        "confidence": 0.0,
                        "explanation": "No anomaly detected",
                        "affected_modalities": []
                    },
                    "actions": [],
                    "timestamp": datetime.utcnow().isoformat(),
                    "system_metrics": {
                        "samples_processed": state.samples_processed + 1,
                        "anomalies_detected": state.anomalies_detected,
                        "anomaly_rate": (state.anomalies_detected / (state.samples_processed + 1)) * 100 if state.samples_processed > 0 else 0.0,
                        "uptime_seconds": int(time.time() - state.start_time)
                    },
                    "latency_ms": {
                        "total": int(total_time),
                        "qdrant": int(qdrant_time) if 'qdrant_time' in locals() else 0,
                        "embedding": int(embedding_time)
                    }
                }
        else:
            total_time = (time.time() - start_time) * 1000
            
            result_data = {
                "monitoring": True,
                "error": "Failed to generate embedding",
                "video": {
                    "frame": None, 
                    "anomaly": False, 
                    "location": "Zone_A",
                    "camera_id": "CAM-01",
                    "timestamp": current_timestamp.isoformat()
                },
                "audio": {"waveform": [], "anomaly": False, "metrics": {}},
                "sensors": {},
                "alert": {"severity": "ERROR", "cause": None, "confidence": 0.0},
                "actions": [],
                "timestamp": datetime.utcnow().isoformat(),
                "system_metrics": {
                    "samples_processed": state.samples_processed,
                    "anomalies_detected": state.anomalies_detected,
                    "anomaly_rate": 0.0,
                    "uptime_seconds": int(time.time() - state.start_time)
                },
                "latency_ms": {
                    "total": int(total_time),
                    "qdrant": 0,
                    "embedding": int(embedding_time) if 'embedding_time' in locals() else 0
                }
            }
        
        # Encode frame for display
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        result_data["video"]["frame"] = frame_base64
        
        # Update state
        state.current_sample += 1
        state.samples_processed += 1
        
        logger.info(f"✅ Detection #{state.samples_processed} complete (frame {sample_idx}/{int(state.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))})")
        
        return result_data
        
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        return {
            "monitoring": True,
            "error": str(e)
        }


if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*60)
    print("🤖 Starting CrewAI Chemical Leak Monitoring Backend")
    print("="*60)
    print("\nBackend will be available at:")
    print("  • API: http://localhost:8000")
    print("  • Docs: http://localhost:8000/docs")
    print("\n🤖 Using CrewAI agents with LLM reasoning")
    print("="*60 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
