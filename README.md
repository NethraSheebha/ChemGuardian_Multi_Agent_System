# ChemGuardian
**AI-Powered Chemical Leak Detection System using Multi-Agent Intelligence and Vector Similarity Search**

ChemGuardian is a **production-grade industrial safety system** designed to detect chemical leaks in real time using **multimodal data streams, vector similarity search, and a coordinated multi-agent architecture**.

The system combines **computer vision, audio analysis, and sensor data embeddings** with **Qdrant vector search** to identify anomalies in industrial environments such as chemical plants and manufacturing facilities.

ChemGuardian enables **early leak detection, automated root-cause analysis, and intelligent safety response**, reducing the risk of catastrophic industrial accidents.

---

# Project Description

Chemical leaks in industrial facilities can escalate into severe safety hazards if not detected immediately. Traditional monitoring systems rely on threshold-based alarms that often miss early warning signals or produce false positives.

**ChemGuardian addresses this limitation by leveraging AI-driven anomaly detection.**

The platform continuously processes **video frames, audio signals, and sensor data**, converts them into **multimodal embeddings**, and stores them in a **vector database** for similarity-based anomaly detection.

When unusual patterns are detected, a team of specialized AI agents collaborates to:

1. Detect anomalies
2. Identify potential causes
3. Assess risk levels
4. Trigger appropriate response strategies

This **multi-agent workflow enables faster and more intelligent responses** compared to conventional monitoring systems.

---

# Key Features

## Multimodal Anomaly Detection

Processes multiple data modalities simultaneously:

- **Video Streams** – Leak visualization, vapor clouds, equipment changes  
- **Audio Signals** – Gas leak sounds, pressure releases  
- **Sensor Data** – Chemical concentration, temperature, pressure  

---

## Vector Similarity Detection

Uses **Qdrant Vector Database** to compare incoming embeddings against historical baseline patterns to identify anomalies.

---

## Multi-Agent AI Architecture

Autonomous agents collaborate to analyze data, determine causes, and generate safety responses.

---

## Adaptive Detection Thresholds

Thresholds automatically adjust based on feedback to reduce false positives.

---

## Continual Learning

The system improves over time by incorporating **newly labeled anomaly data**.

---

## Real-Time Processing

Designed for **industrial deployment with low latency detection pipelines.**

---

# System Architecture

```
Data Sources
   │
   ▼
Input Collection Agent
   │
   ▼
Anomaly Detection Agent
   │
   ▼
Cause Detection Agent
   │
   ▼
Risk Response Agents
   │
   ▼
Qdrant Vector Database
(baselines, embeddings, anomalies)
```

---

# Agents Overview

ChemGuardian uses **CrewAI-based autonomous agents** to orchestrate the detection workflow.

---

## Input Collection Agent

Responsible for ingesting and preprocessing multimodal data streams.

### Responsibilities

- Collect video frames (1 FPS)
- Process audio segments (1 second windows)
- Read sensor data (1 second intervals)
- Generate embeddings for each modality

---

## Anomaly Detection Agent

Identifies unusual patterns by comparing incoming embeddings against historical baselines stored in Qdrant.

### Responsibilities

- Query vector database
- Compute similarity scores
- Detect abnormal deviations
- Trigger anomaly alerts

---

## Cause Detection Agent

Analyzes detected anomalies to determine the likely cause of the leak.

### Responsibilities

- Cross-analyze modalities
- Identify chemical signatures
- Correlate sensor and environmental data
- Predict leak source

---

## Risk Response Agents

Initiates safety actions once a leak is confirmed.

### Responsibilities

- Classify risk severity
- Generate response strategies
- Trigger alerts and safety protocols
- Log incident reports

---

# Multimodal Embedding Models

ChemGuardian generates embeddings using specialized neural networks for each data type.

| Data Type | Model |
|----------|-------|
| Video | MobileNetV3 |
| Audio | PANNs CNN14 |
| Sensor Data | Learned embedding adapter |

These embeddings are stored in **Qdrant** for efficient similarity search.

---

# Target Chemicals

ChemGuardian is designed to detect leaks involving:

- Chlorine
- Ammonia
- Methyl Isocyanate (MIC)
- Industrial acidic gases
- Toxic industrial chemicals

---

# Performance Targets

| Metric | Target |
|------|------|
| Processing Interval | 1 second |
| Detection Latency | < 2 seconds |
| Throughput | ≥ 1 data point/sec per modality |

---

# Project Structure

```
.
├── src/
│   ├── agents/          # CrewAI agent implementations
│   ├── models/          # Multimodal embedding models
│   ├── database/        # Qdrant client and schemas
│   ├── config/          # Configuration management
│   └── utils/           # Helper utilities
│
├── tests/               # Unit and integration tests
├── logs/                # Application logs
├── .env.example         # Environment variable template
└── requirements.txt     # Python dependencies
```

---

# Technology Stack

## AI / ML

- PyTorch
- MobileNetV3
- PANNs CNN14
- Multimodal Embeddings

## AI Agents

- CrewAI

## Vector Database

- Qdrant

## Infrastructure

- Python
- Docker (optional)
- REST APIs

---

# Installation

Clone the repository:

```bash
git clone https://github.com/NethraSheebha/CHemGuardian_Multi_Agent_System.git
cd chemguardian
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Configure environment variables:

```bash
cp .env.example .env
```

Run the system:

```bash
python main.py
```

---

# Use Cases

ChemGuardian can be deployed in:

- Chemical manufacturing plants
- Fertilizer factories
- Oil and gas facilities
- Hazardous material storage sites
- Industrial safety monitoring systems

---

# Future Improvements

- Edge deployment on industrial IoT devices
- Real-time drone-based monitoring
- Predictive maintenance integration
- Automated containment protocols
- Advanced multimodal transformer models

---

# License

Proprietary – Industrial Safety System
