# Chemical Leak Monitoring System

A production-grade multi-agent system for real-time chemical leak detection and response using multimodal embeddings and vector similarity search.

## Overview

This system processes video (1 FPS), audio (1-second windows), and sensor data (1-second intervals) to detect chemical leaks in industrial facilities. It uses:

- **Multimodal Embeddings**: Video (MobileNetV3), Audio (PANNs CNN14), Sensor (learned adapter)
- **Vector Database**: Qdrant for similarity-based anomaly detection
- **Multi-Agent Architecture**: Input Collection, Anomaly Detection, Cause Detection, Risk Response
- **Adaptive Thresholds**: Dynamic threshold adjustment based on feedback
- **Continual Learning**: Model updates from labeled anomalies

## Architecture

```
Data Sources → Input Collection Agent → Anomaly Detection Agent → Cause Detection Agent → Risk Response Agents
                                              ↓
                                        Qdrant Vector DB
                                    (baselines, data, anomalies)
```

## Project Structure

```
.
├── src/
│   ├── agents/          # Agent implementations
│   ├── models/          # Embedding models
│   ├── database/        # Qdrant client and schemas
│   ├── config/          # Configuration management
│   └── utils/           # Utilities and helpers
├── tests/               # Test suite
├── logs/                # Application logs
├── .env.example         # Environment variable template
└── requirements.txt     # Python dependencies
```

## Setup

### 1. Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Edit `.env` with your settings:

- `QDRANT_HOST`: Qdrant server host (default: localhost)
- `QDRANT_PORT`: Qdrant server port (default: 6333)
- `DEVICE`: cpu or cuda for model inference

### 3. Start Qdrant

Using Docker:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

### 4. Run Tests

```bash
pytest
```

## Configuration

All configuration is managed through environment variables (see `.env.example`):

- **Qdrant**: Connection parameters
- **Models**: Model selection and device
- **Thresholds**: Initial adaptive thresholds
- **Agents**: Processing intervals and queue depths
- **Logging**: Log level, format, and directory

## Development

### Running Tests

```bash
# All tests
pytest

# Unit tests only
pytest -m unit

# With coverage
pytest --cov=src --cov-report=html
```

### Code Quality

```bash
# Type checking
mypy src/

# Linting
flake8 src/

# Formatting
black src/
```

## Target Chemicals

- Chlorine
- Ammonia
- Methyl Isocyanate (MIC)
- Acidic and toxic industrial gases

## Performance Targets

- **Processing Interval**: 1 second per data point
- **End-to-End Latency**: < 2 seconds from ingestion to detection
- **Throughput**: ≥ 1 data point/second per modality

## License

Proprietary - Industrial Safety System
