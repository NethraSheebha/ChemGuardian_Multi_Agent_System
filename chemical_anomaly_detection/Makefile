# Makefile for Chemical Leak Monitoring System Docker Operations

.PHONY: help build up down restart logs ps clean test health backup restore

# Default target
help:
	@echo "Chemical Leak Monitoring System - Docker Operations"
	@echo ""
	@echo "Available targets:"
	@echo "  build          - Build all Docker images"
	@echo "  up             - Start all services"
	@echo "  down           - Stop all services"
	@echo "  restart        - Restart all services"
	@echo "  logs           - View logs from all services"
	@echo "  ps             - Show service status"
	@echo "  health         - Check health of all services"
	@echo "  clean          - Remove containers, networks, and volumes"
	@echo "  backup         - Backup Qdrant data"
	@echo "  restore        - Restore Qdrant data from backup"
	@echo "  test           - Run integration tests"
	@echo ""
	@echo "Service-specific targets:"
	@echo "  build-<service>    - Build specific service"
	@echo "  logs-<service>     - View logs for specific service"
	@echo "  restart-<service>  - Restart specific service"
	@echo ""
	@echo "Services: qdrant, input-collection, anomaly-detection,"
	@echo "          cause-detection, response-mild, response-medium, response-high"

# Build all images
build:
	docker-compose build

# Build specific service
build-%:
	docker-compose build $*

# Start all services
up:
	docker-compose up -d

# Start specific service
up-%:
	docker-compose up -d $*

# Stop all services
down:
	docker-compose down

# Stop specific service
down-%:
	docker-compose stop $*

# Restart all services
restart: down up

# Restart specific service
restart-%:
	docker-compose restart $*

# View logs from all services
logs:
	docker-compose logs -f

# View logs from specific service
logs-%:
	docker-compose logs -f $*

# Show service status
ps:
	docker-compose ps

# Check health of all services
health:
	@echo "Checking service health..."
	@docker-compose ps
	@echo ""
	@echo "Qdrant health:"
	@curl -s http://localhost:6333/healthz || echo "Qdrant not responding"

# Clean up containers, networks, and volumes
clean:
	@echo "WARNING: This will remove all containers, networks, and volumes!"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		docker-compose down -v; \
		docker system prune -f; \
	fi

# Backup Qdrant data
backup:
	@mkdir -p backups
	@echo "Stopping services..."
	docker-compose down
	@echo "Backing up Qdrant data..."
	docker run --rm \
		-v chem-monitor-qdrant-storage:/data \
		-v $$(pwd)/backups:/backup \
		alpine tar czf /backup/qdrant-backup-$$(date +%Y%m%d-%H%M%S).tar.gz -C /data .
	@echo "Backup complete: backups/qdrant-backup-$$(date +%Y%m%d-%H%M%S).tar.gz"
	@echo "Restarting services..."
	docker-compose up -d

# Restore Qdrant data from backup
restore:
	@echo "Available backups:"
	@ls -lh backups/qdrant-backup-*.tar.gz 2>/dev/null || echo "No backups found"
	@read -p "Enter backup filename (e.g., qdrant-backup-20240101-120000.tar.gz): " backup_file; \
	if [ -f "backups/$$backup_file" ]; then \
		echo "Stopping services..."; \
		docker-compose down; \
		echo "Restoring from $$backup_file..."; \
		docker run --rm \
			-v chem-monitor-qdrant-storage:/data \
			-v $$(pwd)/backups:/backup \
			alpine tar xzf /backup/$$backup_file -C /data; \
		echo "Restore complete"; \
		echo "Restarting services..."; \
		docker-compose up -d; \
	else \
		echo "Backup file not found: backups/$$backup_file"; \
		exit 1; \
	fi

# Run integration tests
test:
	@echo "Running integration tests..."
	docker-compose up -d
	@sleep 10
	@echo "Waiting for services to be healthy..."
	@timeout 60 bash -c 'until curl -s http://localhost:6333/healthz > /dev/null; do sleep 2; done' || (echo "Services failed to start" && exit 1)
	@echo "Services are healthy"
	@echo "Running tests..."
	python -m pytest tests/integration/ -v

# Show resource usage
stats:
	docker stats --no-stream

# Tail logs with line count
tail-%:
	docker-compose logs --tail=100 -f $*

# Execute shell in container
shell-%:
	docker-compose exec $* /bin/bash

# Show configuration
config:
	docker-compose config

# Pull latest images
pull:
	docker-compose pull

# Rebuild without cache
rebuild:
	docker-compose build --no-cache

# Scale specific service
scale-%:
	@read -p "Enter number of replicas: " replicas; \
	docker-compose up -d --scale $*=$$replicas
