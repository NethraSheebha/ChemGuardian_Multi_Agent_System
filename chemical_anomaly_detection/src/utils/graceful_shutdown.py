"""
Graceful shutdown handler for Chemical Leak Monitoring System agents.

This module provides utilities for handling SIGTERM and SIGINT signals
to ensure agents shut down cleanly, flushing queues and closing connections.
"""

import asyncio
import signal
import sys
from typing import Callable, List, Optional
import logging

logger = logging.getLogger(__name__)


class GracefulShutdownHandler:
    """
    Handles graceful shutdown for agents.
    
    Registers signal handlers for SIGTERM and SIGINT, and executes
    cleanup callbacks in order when shutdown is triggered.
    """
    
    def __init__(self, timeout: int = 30):
        """
        Initialize graceful shutdown handler.
        
        Args:
            timeout: Maximum time in seconds to wait for cleanup (default: 30)
        """
        self.timeout = timeout
        self.cleanup_callbacks: List[Callable] = []
        self.shutdown_event = asyncio.Event()
        self._shutdown_initiated = False
        
    def register_cleanup(self, callback: Callable) -> None:
        """
        Register a cleanup callback to be executed during shutdown.
        
        Callbacks are executed in the order they are registered.
        
        Args:
            callback: Async or sync function to call during shutdown
        """
        self.cleanup_callbacks.append(callback)
        logger.debug(f"Registered cleanup callback: {callback.__name__}")
        
    def setup_signal_handlers(self) -> None:
        """
        Set up signal handlers for SIGTERM and SIGINT.
        
        This should be called from the main thread.
        """
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        logger.info("Signal handlers registered for SIGTERM and SIGINT")
        
    def _signal_handler(self, signum: int, frame) -> None:
        """
        Handle shutdown signals.
        
        Args:
            signum: Signal number
            frame: Current stack frame
        """
        if self._shutdown_initiated:
            logger.warning("Shutdown already initiated, ignoring signal")
            return
            
        signal_name = signal.Signals(signum).name
        logger.info(f"Received {signal_name}, initiating graceful shutdown...")
        self._shutdown_initiated = True
        self.shutdown_event.set()
        
    async def wait_for_shutdown(self) -> None:
        """
        Wait for shutdown signal.
        
        This is an async method that blocks until a shutdown signal is received.
        """
        await self.shutdown_event.wait()
        
    async def cleanup(self) -> None:
        """
        Execute all registered cleanup callbacks.
        
        Callbacks are executed in order with a timeout. If the timeout
        is exceeded, remaining callbacks are skipped.
        """
        if not self._shutdown_initiated:
            logger.warning("Cleanup called without shutdown signal")
            return
            
        logger.info(f"Starting cleanup with {len(self.cleanup_callbacks)} callbacks")
        
        try:
            async with asyncio.timeout(self.timeout):
                for i, callback in enumerate(self.cleanup_callbacks, 1):
                    try:
                        logger.info(f"Executing cleanup callback {i}/{len(self.cleanup_callbacks)}: {callback.__name__}")
                        
                        if asyncio.iscoroutinefunction(callback):
                            await callback()
                        else:
                            callback()
                            
                        logger.info(f"Cleanup callback {i} completed successfully")
                        
                    except Exception as e:
                        logger.error(f"Error in cleanup callback {callback.__name__}: {e}", exc_info=True)
                        # Continue with remaining callbacks
                        
        except asyncio.TimeoutError:
            logger.error(f"Cleanup timeout ({self.timeout}s) exceeded, forcing shutdown")
            
        logger.info("Cleanup completed")
        
    async def shutdown(self) -> None:
        """
        Execute full shutdown sequence: wait for signal, then cleanup.
        
        This is a convenience method that combines wait_for_shutdown and cleanup.
        """
        await self.wait_for_shutdown()
        await self.cleanup()
        
    def is_shutting_down(self) -> bool:
        """
        Check if shutdown has been initiated.
        
        Returns:
            True if shutdown is in progress, False otherwise
        """
        return self._shutdown_initiated


class ShutdownManager:
    """
    Context manager for graceful shutdown handling.
    
    Usage:
        async with ShutdownManager(timeout=30) as shutdown:
            shutdown.register_cleanup(cleanup_func1)
            shutdown.register_cleanup(cleanup_func2)
            
            # Run agent main loop
            await agent.run()
    """
    
    def __init__(self, timeout: int = 30):
        """
        Initialize shutdown manager.
        
        Args:
            timeout: Maximum time in seconds to wait for cleanup
        """
        self.handler = GracefulShutdownHandler(timeout=timeout)
        
    async def __aenter__(self):
        """Enter context manager."""
        self.handler.setup_signal_handlers()
        return self.handler
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and perform cleanup."""
        if exc_type is not None:
            logger.error(f"Exception during agent execution: {exc_val}", exc_info=True)
            
        # If shutdown was initiated, cleanup has already been done
        if not self.handler.is_shutting_down():
            logger.info("Performing cleanup on context exit")
            await self.handler.cleanup()
            
        return False  # Don't suppress exceptions


# Example usage
async def example_agent_main():
    """
    Example of how to use graceful shutdown in an agent.
    """
    async with ShutdownManager(timeout=30) as shutdown:
        # Register cleanup callbacks
        shutdown.register_cleanup(close_database_connections)
        shutdown.register_cleanup(flush_message_queue)
        shutdown.register_cleanup(save_state)
        
        # Main agent loop
        while not shutdown.is_shutting_down():
            try:
                # Process data
                await process_data()
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                
        logger.info("Agent main loop exited")


async def close_database_connections():
    """Example cleanup callback."""
    logger.info("Closing database connections...")
    await asyncio.sleep(0.5)  # Simulate cleanup
    logger.info("Database connections closed")


async def flush_message_queue():
    """Example cleanup callback."""
    logger.info("Flushing message queue...")
    await asyncio.sleep(0.5)  # Simulate cleanup
    logger.info("Message queue flushed")


def save_state():
    """Example cleanup callback (sync)."""
    logger.info("Saving agent state...")
    # Simulate cleanup
    logger.info("Agent state saved")


async def process_data():
    """Example data processing."""
    pass


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run example
    asyncio.run(example_agent_main())
