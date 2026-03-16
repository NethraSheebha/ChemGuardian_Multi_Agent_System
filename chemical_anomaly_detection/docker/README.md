# Docker Deployment Guide

This directory contains Docker configurations for the Chemical Leak Monitoring System.

## Architecture

The system consists of the following containerized services:

1. **Qdrant** - Vector database for storing embeddings and baselines
2. **Input Collection Agent** - Ingests and embeds multimodal data (video, audio, sensor)
3. **Anomaly Detection Agent** - Detects anomalies via similarity search
4. **Cause Detection Agent** - Infers causes and classifies severity
5. **Mild Response Agent** - Handles mild severity incidents
6. **Medium Response Agent** - Handles medium severity incidents
7. **High Response Agent** - Handles high severity incidents and emergency shutdown

## Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+
- At least 8GB RAM available for Docker
- 20GB disk space for images and volumes

## Quick Start

### 1. Configure Environment Variables

Copy the environment template and customize:

```bash
cp .env.docker.example .env.docker
```

Edit `.env.docker` to set your configuration values.

### 2. Build Images

Build all service images:

```bash
docker-compose build
```

Or build specific services:

```bash
docker-compose build input-collection
docker-compose build anomaly-detection
```

### 3. Start Services

Start all services:

```bash
docker-compose up -d
```

Start specific services:

```bash
docker-compose up -d qdrant input-collection
```

### 4. Verify Health

Check service status:

```bash
docker-compose ps
```

View logs:

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f input-collection
```

Check Qdrant health:

```bash
curl http://localhost:6333/healthz
```

## Service Configuration

### Qdrant

- **Ports**: 6333 (HTTP), 6334 (gRPC)
- **Volume**: `qdrant_storage` for persistent data
- **Health Check**: HTTP endpoint at `/healthz`

### Input Collection Agent

- **Dependencies**: Qdrant
- **Environment Variables**:
  - `CHEM_MONITOR_QDRANT_HOST` - Qdrant hostname (default: qdrant)
  - `CHEM_MONITOR_VIDEO_MODEL_PATH` - Path to video model
  - `CHEM_MONITOR_AUDIO_MODEL_PATH` - Path to audio model
  - `CHEM_MONITOR_SENSOR_MODEL_PATH` - Path to sensor model
- **Volumes**:
  - `./data:/app/data:ro` - Read-only data directory
  - `./models:/app/models:ro` - Read-only model directory
  - `input_logs:/app/logs` - Log volume

### Anomaly Detection Agent

- **Dependencies**: Qdrant, Input Collection
- **Environment Variables**:
  - `CHEM_MONITOR_INITIAL_THRESHOLD_VIDEO` - Initial video threshold (default: 0.7)
  - `CHEM_MONITOR_INITIAL_THRESHOLD_AUDIO` - Initial audio threshold (default: 0.65)
  - `CHEM_MONITOR_INITIAL_THRESHOLD_SENSOR` - Initial sensor threshold (default: 2.5)
  - `CHEM_MONITOR_THRESHOLD_LEARNING_RATE` - Adaptive learning rate (default: 0.05)

### Cause Detection Agent

- **Dependencies**: Qdrant, Anomaly Detection
- **Environment Variables**:
  - `CHEM_MONITOR_MSDS_DATABASE_PATH` - Path to MSDS database
  - `CHEM_MONITOR_SOP_DATABASE_PATH` - Path to SOP database

### Response Agents (Mild, Medium, High)

- **Dependencies**: Qdrant, Cause Detection
- **Environment Variables**:
  - `CHEM_MONITOR_MSDS_DATABASE_PATH` - Path to MSDS database
  - `CHEM_MONITOR_SOP_DATABASE_PATH` - Path to SOP database
  - `CHEM_MONITOR_EMERGENCY_CONTACT` - Emergency contact (High only)

## Graceful Shutdown

All agents implement graceful shutdown handlers with a 30-second grace period:

```bash
# Stop all services gracefully
docker-compose down

# Stop with timeout override
docker-compose down -t 60
```

During shutdown:

1. Agents stop accepting new data
2. In-flight processing completes
3. Queued embeddings are flushed to Qdrant
4. Connections are closed cleanly

## Networking

All services communicate via the `chem-monitor-network` bridge network:

- Internal DNS resolution (e.g., `qdrant:6333`)
- Isolated from host network by default
- Only Qdrant ports exposed to host

## Volumes

Persistent volumes for data storage:

- `qdrant_storage` - Qdrant vector database
- `input_logs` - Input Collection Agent logs
- `anomaly_logs` - Anomaly Detection Agent logs
- `cause_logs` - Cause Detection Agent logs
- `response_*_logs` - Response Agent logs

View volume details:

```bash
docker volume ls | grep chem-monitor
docker volume inspect chem-monitor-qdrant-storage
```

## Monitoring

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service with tail
docker-compose logs -f --tail=100 input-collection

# Since timestamp
docker-compose logs --since 2024-01-01T00:00:00
```

### Resource Usage

```bash
# Container stats
docker stats

# Specific service
docker stats chem-monitor-input-collection
```

### Health Checks

```bash
# Check all health statuses
docker-compose ps

# Inspect specific service health
docker inspect --format='{{.State.Health.Status}}' chem-monitor-qdrant
```

## Troubleshooting

### Service Won't Start

1. Check logs: `docker-compose logs <service-name>`
2. Verify dependencies are healthy: `docker-compose ps`
3. Check environment variables: `docker-compose config`
4. Verify volumes are accessible: `docker volume ls`

### High Memory Usage

1. Check container stats: `docker stats`
2. Adjust Docker memory limits in `docker-compose.yml`:
   ```yaml
   deploy:
     resources:
       limits:
         memory: 2G
   ```

### Qdrant Connection Issues

1. Verify Qdrant is healthy: `curl http://localhost:6333/healthz`
2. Check network connectivity: `docker-compose exec input-collection ping qdrant`
3. Review Qdrant logs: `docker-compose logs qdrant`

### Agent Crashes

1. Check logs for stack traces: `docker-compose logs <agent-name>`
2. Verify model files exist: `docker-compose exec <agent-name> ls -la /app/models`
3. Check data files: `docker-compose exec <agent-name> ls -la /app/data`

## Scaling

Scale specific services:

```bash
# Scale anomaly detection to 3 instances
docker-compose up -d --scale anomaly-detection=3

# Scale response agents
docker-compose up -d --scale response-mild=2 --scale response-medium=2
```

**Note**: Input Collection Agent should not be scaled (single instance processes data stream).

## Production Deployment

### Security Hardening

1. **Use secrets for sensitive data**:

   ```yaml
   secrets:
     qdrant_api_key:
       external: true
   ```

2. **Enable TLS for Qdrant**:

   ```yaml
   environment:
     - QDRANT__SERVICE__ENABLE_TLS=true
   ```

3. **Run with read-only root filesystem**:

   ```yaml
   security_opt:
     - no-new-privileges:true
   read_only: true
   ```

4. **Limit container capabilities**:
   ```yaml
   cap_drop:
     - ALL
   cap_add:
     - NET_BIND_SERVICE
   ```

### Resource Limits

Add resource constraints:

```yaml
deploy:
  resources:
    limits:
      cpus: "2.0"
      memory: 4G
    reservations:
      cpus: "1.0"
      memory: 2G
```

### Logging

Configure centralized logging:

```yaml
logging:
  driver: "json-file"
  options:
    max-size: "100m"
    max-file: "10"
```

Or use external logging driver:

```yaml
logging:
  driver: "syslog"
  options:
    syslog-address: "tcp://logs.example.com:514"
```

## Backup and Recovery

### Backup Qdrant Data

```bash
# Stop services
docker-compose down

# Backup volume
docker run --rm -v chem-monitor-qdrant-storage:/data -v $(pwd)/backups:/backup \
  alpine tar czf /backup/qdrant-backup-$(date +%Y%m%d).tar.gz -C /data .

# Restart services
docker-compose up -d
```

### Restore Qdrant Data

```bash
# Stop services
docker-compose down

# Restore volume
docker run --rm -v chem-monitor-qdrant-storage:/data -v $(pwd)/backups:/backup \
  alpine tar xzf /backup/qdrant-backup-20240101.tar.gz -C /data

# Restart services
docker-compose up -d
```

## Maintenance

### Update Images

```bash
# Pull latest base images
docker-compose pull

# Rebuild with no cache
docker-compose build --no-cache

# Restart with new images
docker-compose up -d
```

### Clean Up

```bash
# Remove stopped containers
docker-compose down

# Remove volumes (WARNING: deletes data)
docker-compose down -v

# Remove unused images
docker image prune -a

# Remove all unused resources
docker system prune -a --volumes
```

## Support

For issues and questions:

- Check logs: `docker-compose logs`
- Review documentation: `README.md`
- Inspect configuration: `docker-compose config`
