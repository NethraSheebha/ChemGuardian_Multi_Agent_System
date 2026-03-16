"""
Regenerate sensor baselines with corrected normalization parameters
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from scripts.seed_baselines import main

if __name__ == "__main__":
    print("\n" + "="*80)
    print("REGENERATING SENSOR BASELINES")
    print("Using corrected normalization parameters from normal_sensor_data.csv")
    print("="*80 + "\n")
    
    asyncio.run(main())
    
    print("\n" + "="*80)
    print("✅ Sensor baselines regenerated successfully!")
    print("="*80)
