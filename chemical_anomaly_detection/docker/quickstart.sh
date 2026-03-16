#!/bin/bash
# Quick start script for Chemical Leak Monitoring System

set -e

echo "=========================================="
echo "Chemical Leak Monitoring System"
echo "Quick Start Script"
echo "=========================================="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed"
    echo "Please install Docker from https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "Error: Docker Compose is not installed"
    echo "Please install Docker Compose from https://docs.docker.com/compose/install/"
    exit 1
fi

# Check if .env.docker exists
if [ ! -f .env.docker ]; then
    echo "Creating .env.docker from template..."
    cp .env.docker.example .env.docker
    echo "✓ Created .env.docker"
    echo ""
    echo "Please review and customize .env.docker before proceeding"
    echo "Press Enter to continue or Ctrl+C to exit..."
    read
fi

# Create required directories
echo "Creating required directories..."
mkdir -p data models logs backups
echo "✓ Directories created"
echo ""

# Check if MSDS and SOP databases exist
if [ ! -f data/msds_database.json ]; then
    echo "Warning: data/msds_database.json not found"
    echo "Creating empty MSDS database..."
    echo '{"chemicals": []}' > data/msds_database.json
fi

if [ ! -f data/sop_database.json ]; then
    echo "Warning: data/sop_database.json not found"
    echo "Creating empty SOP database..."
    echo '{"procedures": []}' > data/sop_database.json
fi

# Build images
echo "Building Docker images..."
echo "This may take several minutes on first run..."
docker-compose build

if [ $? -ne 0 ]; then
    echo "Error: Failed to build Docker images"
    exit 1
fi

echo "✓ Images built successfully"
echo ""

# Start services
echo "Starting services..."
docker-compose up -d

if [ $? -ne 0 ]; then
    echo "Error: Failed to start services"
    exit 1
fi

echo "✓ Services started"
echo ""

# Wait for Qdrant to be healthy
echo "Waiting for Qdrant to be ready..."
timeout=60
elapsed=0
while [ $elapsed -lt $timeout ]; do
    if curl -s http://localhost:6333/healthz > /dev/null 2>&1; then
        echo "✓ Qdrant is ready"
        break
    fi
    sleep 2
    elapsed=$((elapsed + 2))
    echo -n "."
done

if [ $elapsed -ge $timeout ]; then
    echo ""
    echo "Warning: Qdrant did not become ready within ${timeout}s"
    echo "Check logs with: docker-compose logs qdrant"
fi

echo ""
echo "=========================================="
echo "System Status"
echo "=========================================="
docker-compose ps
echo ""

echo "=========================================="
echo "Quick Start Complete!"
echo "=========================================="
echo ""
echo "Useful commands:"
echo "  View logs:        docker-compose logs -f"
echo "  Stop services:    docker-compose down"
echo "  Restart services: docker-compose restart"
echo "  Check status:     docker-compose ps"
echo "  View Qdrant UI:   http://localhost:6333/dashboard"
echo ""
echo "For more information, see docker/README.md"
