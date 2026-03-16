from src.crewai_agents import InputCollectionCrew, AnomalyDetectionCrew

# Initialize (same parameters as before!)
input_crew = InputCollectionCrew(video_proc, audio_proc, sensor_proc)
anomaly_crew = AnomalyDetectionCrew(client, search_engine, threshold_mgr, storage_mgr)

# Use (same interface as before!)
embedding = await input_
