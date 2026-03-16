"""
Simple script to run the CrewAI backend with Ollama LLM
"""

import uvicorn
from backend_api_crewai import app

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🤖 Starting CrewAI Chemical Leak Monitoring Backend")
    print("="*60)
    print("\nBackend will be available at:")
    print("  • API: http://localhost:8000")
    print("  • Docs: http://localhost:8000/docs")
    print("\n🦙 Using Ollama LLM for agent reasoning")
    print("   Model: llama3.2:1b")
    print("   URL: http://localhost:11434")
    print("\n⚠️  Make sure Ollama is running:")
    print("   Check with: ollama list")
    print("\nWaiting for startup (this may take 45-75 seconds)...")
    print("="*60 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
