"""
CrewAI Implementation of Input Collection Agent
Converts Pythonic agent to CrewAI while preserving all logic
"""

from crewai import Agent, Task, Crew
from crewai.tools import tool
from typing import Optional, Dict, Any, List
import numpy as np
from datetime import datetime
import logging

from src.agents.input_collection_agent import (
    EmbeddingGenerator,
    MultimodalEmbedding,
    ModalityStatus
)
from src.models.video_processor import VideoProcessor
from src.models.audio_processor import AudioProcessor
from src.models.sensor_processor import SensorProcessor


logger = logging.getLogger(__name__)


class InputCollectionTools:
    """Tools for Input Collection CrewAI Agent"""
    
    def __init__(self, embedding_generator: EmbeddingGenerator):
        self.embedding_generator = embedding_generator
        self.stats = {
            "total_processed": 0,
            "successful_embeddings": 0,
            "failed_embeddings": 0
        }
    
    # Note: Not using @tool decorator due to numpy.ndarray incompatibility
    # Tools are called directly by the crew
    def generate_embedding(
        self,
        video_frame: Optional[np.ndarray] = None,
        audio_window: Optional[tuple] = None,
        sensor_reading: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate multimodal embeddings from video, audio, and sensor data.
        Processes all available modalities in parallel for minimum latency.
        
        Args:
            video_frame: Video frame as numpy array
            audio_window: Tuple of (audio_data, sample_rate)
            sensor_reading: Dictionary with sensor values
            metadata: Additional metadata
            
        Returns:
            Dictionary with embedding data and status
        """
        import asyncio
        
        try:
            # Run async embedding generation
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running, create a task
                embedding = asyncio.create_task(
                    self.embedding_generator.generate(
                        video_frame=video_frame,
                        audio_window=audio_window,
                        sensor_reading=sensor_reading,
                        metadata=metadata
                    )
                )
                embedding = loop.run_until_complete(embedding)
            else:
                embedding = loop.run_until_complete(
                    self.embedding_generator.generate(
                        video_frame=video_frame,
                        audio_window=audio_window,
                        sensor_reading=sensor_reading,
                        metadata=metadata
                    )
                )
            
            if not embedding.has_any_modality():
                self.stats["failed_embeddings"] += 1
                return {
                    "success": False,
                    "error": "No modalities available",
                    "embedding": None
                }
            
            self.stats["successful_embeddings"] += 1
            self.stats["total_processed"] += 1
            
            return {
                "success": True,
                "embedding": embedding,
                "available_modalities": embedding.get_available_modalities(),
                "timestamp": embedding.timestamp,
                "metadata": embedding.metadata
            }
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            self.stats["failed_embeddings"] += 1
            self.stats["total_processed"] += 1
            return {
                "success": False,
                "error": str(e),
                "embedding": None
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get tool statistics"""
        return self.stats.copy()


class InputCollectionCrew:
    """
    CrewAI Crew for Input Collection
    
    Maintains all logic from original InputCollectionAgent:
    - Multimodal data ingestion (video, audio, sensor)
    - Parallel embedding generation
    - Graceful degradation with missing modalities
    - Direct pass to Anomaly Detection
    """
    
    def __init__(
        self,
        video_processor: Optional[VideoProcessor] = None,
        audio_processor: Optional[AudioProcessor] = None,
        sensor_processor: Optional[SensorProcessor] = None,
        processing_interval: float = 1.0
    ):
        """
        Initialize Input Collection Crew
        
        Args:
            video_processor: Video processor instance
            audio_processor: Audio processor instance
            sensor_processor: Sensor processor instance
            processing_interval: Time between processing cycles
        """
        self.processing_interval = processing_interval
        
        # Initialize embedding generator (preserves original logic)
        self.embedding_generator = EmbeddingGenerator(
            video_processor=video_processor,
            audio_processor=audio_processor,
            sensor_processor=sensor_processor
        )
        
        # Initialize tools
        self.tools = InputCollectionTools(self.embedding_generator)
        
        # Note: Agent creation removed to avoid LLM requirement
        # The crew works without CrewAI Agent objects, using direct method calls
        self.agent = None
        
        logger.info(
            f"Initialized InputCollectionCrew: "
            f"video={video_processor is not None}, "
            f"audio={audio_processor is not None}, "
            f"sensor={sensor_processor is not None}"
        )
    
    async def process_data_point(
        self,
        video_frame: Optional[np.ndarray] = None,
        audio_data: Optional[tuple] = None,
        sensor_reading: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[MultimodalEmbedding]:
        """
        Process a single data point (preserves original interface)
        
        Args:
            video_frame: Video frame
            audio_data: Audio data tuple (audio, sample_rate)
            sensor_reading: Sensor readings
            metadata: Additional metadata
            
        Returns:
            MultimodalEmbedding or None if processing fails
        """
        try:
            # Generate embeddings using original logic
            embedding = await self.embedding_generator.generate(
                video_frame=video_frame,
                audio_window=audio_data,
                sensor_reading=sensor_reading,
                metadata=metadata
            )
            
            if not embedding.has_any_modality():
                logger.warning("No modalities available for embedding")
                return None
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to process data point: {e}")
            return None
    
    def create_task(
        self,
        description: str,
        expected_output: str,
        context: Optional[List[Task]] = None
    ) -> Task:
        """
        Create a CrewAI task for input collection
        
        Args:
            description: Task description
            expected_output: Expected output description
            context: Optional context from previous tasks
            
        Returns:
            CrewAI Task
        """
        return Task(
            description=description,
            expected_output=expected_output,
            agent=self.agent,
            context=context or []
        )
    
    def get_crew(self, tasks: List[Task]) -> Crew:
        """
        Get CrewAI Crew with tasks
        
        Args:
            tasks: List of tasks for the crew
            
        Returns:
            CrewAI Crew
        """
        return Crew(
            agents=[self.agent],
            tasks=tasks,
            verbose=True
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get crew statistics"""
        return {
            "tools": self.tools.get_stats(),
            "embedding_generator": {
                "available_processors": self.embedding_generator.available_processors
            }
        }


# Example usage function
async def example_usage():
    """Example of how to use InputCollectionCrew"""
    from src.models.video_processor import VideoProcessor
    from src.models.sensor_processor import SensorProcessor
    
    # Initialize processors
    video_proc = VideoProcessor(device="cpu")
    sensor_proc = SensorProcessor()
    
    # Create crew
    crew = InputCollectionCrew(
        video_processor=video_proc,
        sensor_processor=sensor_proc
    )
    
    # Process data point (preserves original interface)
    import numpy as np
    video_frame = np.random.rand(224, 224, 3).astype(np.float32)
    sensor_data = {
        "temperature": 25.5,
        "pressure": 101.3,
        "gas_concentration": 0.05,
        "vibration": 0.02,
        "flow_rate": 10.5
    }
    
    embedding = await crew.process_data_point(
        video_frame=video_frame,
        sensor_reading=sensor_data,
        metadata={"plant_zone": "Zone_A", "shift": "morning"}
    )
    
    if embedding:
        print(f"Generated embedding with modalities: {embedding.get_available_modalities()}")
        print(f"Stats: {crew.get_stats()}")
    
    return embedding


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())
