#!/usr/bin/env python3
"""
Health check script for Chemical Leak Monitoring System agents.

This script is used by Docker HEALTHCHECK to verify agent health.
Returns exit code 0 if healthy, 1 if unhealthy.
"""

import sys
import os
import asyncio
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.insert(0, '/app')

try:
    from src.config.settings import SystemConfig
    from qdrant_client import QdrantClient
except ImportError as e:
    print(f"Import error: {e}", file=sys.stderr)
    sys.exit(1)


async def check_qdrant_connection() -> bool:
    """
    Check if Qdrant is accessible.
    
    Returns:
        True if Qdrant is healthy, False otherwise
    """
    try:
        config = SystemConfig()
        client = QdrantClient(
            host=config.qdrant_host,
            port=config.qdrant_port,
            timeout=5
        )
        
        # Try to get collections
        collections = client.get_collections()
        return True
        
    except Exception as e:
        print(f"Qdrant connection failed: {e}", file=sys.stderr)
        return False


def check_heartbeat_file() -> bool:
    """
    Check if agent heartbeat file is recent.
    
    Agents should update a heartbeat file every 60 seconds.
    If the file is older than 120 seconds, consider unhealthy.
    
    Returns:
        True if heartbeat is recent, False otherwise
    """
    heartbeat_path = Path("/tmp/agent_heartbeat")
    
    if not heartbeat_path.exists():
        print("Heartbeat file does not exist", file=sys.stderr)
        return False
        
    try:
        mtime = datetime.fromtimestamp(heartbeat_path.stat().st_mtime)
        age = datetime.now() - mtime
        
        if age > timedelta(seconds=120):
            print(f"Heartbeat file is stale: {age.total_seconds()}s old", file=sys.stderr)
            return False
            
        return True
        
    except Exception as e:
        print(f"Error checking heartbeat: {e}", file=sys.stderr)
        return False


def check_error_log() -> bool:
    """
    Check if there are recent critical errors in logs.
    
    Returns:
        True if no critical errors, False if critical errors found
    """
    log_path = Path("/app/logs/agent.log")
    
    if not log_path.exists():
        # No log file yet, assume healthy
        return True
        
    try:
        # Check last 100 lines for CRITICAL errors
        with open(log_path, 'r') as f:
            lines = f.readlines()[-100:]
            
        critical_count = sum(1 for line in lines if 'CRITICAL' in line)
        
        if critical_count > 5:
            print(f"Too many critical errors: {critical_count}", file=sys.stderr)
            return False
            
        return True
        
    except Exception as e:
        print(f"Error checking logs: {e}", file=sys.stderr)
        # Don't fail health check if we can't read logs
        return True


async def main():
    """
    Run all health checks.
    
    Exit with code 0 if healthy, 1 if unhealthy.
    """
    checks = {
        "qdrant_connection": await check_qdrant_connection(),
        "heartbeat": check_heartbeat_file(),
        "error_log": check_error_log()
    }
    
    # Print check results
    for check_name, result in checks.items():
        status = "✓" if result else "✗"
        print(f"{status} {check_name}: {'healthy' if result else 'unhealthy'}")
    
    # All checks must pass
    if all(checks.values()):
        print("Agent is healthy")
        sys.exit(0)
    else:
        print("Agent is unhealthy")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
