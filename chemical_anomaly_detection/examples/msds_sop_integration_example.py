"""Example demonstrating MSDS and SOP integration usage."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.integrations import MSDSIntegration, SOPIntegration


def main():
    """Demonstrate MSDS and SOP integration."""
    
    # Initialize integrations
    print("Initializing MSDS and SOP integrations...")
    msds = MSDSIntegration('data/msds_database.json')
    sop = SOPIntegration('data/sop_database.json')
    
    print(f"\nLoaded {len(msds.get_all_chemicals())} chemicals from MSDS database")
    print(f"Loaded SOPs for {len(sop.get_all_zones())} plant zones")
    
    # Example 1: Get chemical information
    print("\n" + "="*80)
    print("Example 1: Retrieving Chemical Information")
    print("="*80)
    
    chemical_name = "chlorine"
    chlorine_info = msds.get_chemical_info(chemical_name)
    
    if chlorine_info:
        print(f"\nChemical: {chlorine_info.name}")
        print(f"CAS Number: {chlorine_info.cas_number}")
        print(f"\nExposure Limits (ppm):")
        for limit_type, value in chlorine_info.exposure_limits.items():
            print(f"  {limit_type}: {value}")
        
        print(f"\nEmergency Procedures ({len(chlorine_info.emergency_procedures)}):")
        for i, procedure in enumerate(chlorine_info.emergency_procedures[:3], 1):
            print(f"  {i}. {procedure}")
        if len(chlorine_info.emergency_procedures) > 3:
            print(f"  ... and {len(chlorine_info.emergency_procedures) - 3} more")
        
        print(f"\nPPE Requirements ({len(chlorine_info.ppe_requirements)}):")
        for i, ppe in enumerate(chlorine_info.ppe_requirements[:3], 1):
            print(f"  {i}. {ppe}")
        if len(chlorine_info.ppe_requirements) > 3:
            print(f"  ... and {len(chlorine_info.ppe_requirements) - 3} more")
    
    # Example 2: Get SOPs for a specific zone and severity
    print("\n" + "="*80)
    print("Example 2: Retrieving Standard Operating Procedures")
    print("="*80)
    
    plant_zone = "zone_a"
    severity = "high"
    
    procedures = sop.get_procedures(plant_zone, severity)
    
    print(f"\nSOPs for {plant_zone.upper()} - {severity.upper()} severity:")
    print(f"Total procedures: {len(procedures)}")
    print("\nProcedures:")
    for i, procedure in enumerate(procedures[:5], 1):
        print(f"  {i}. {procedure}")
    if len(procedures) > 5:
        print(f"  ... and {len(procedures) - 5} more")
    
    # Example 3: Simulated emergency response
    print("\n" + "="*80)
    print("Example 3: Simulated Emergency Response")
    print("="*80)
    
    detected_chemical = "ammonia"
    affected_zone = "zone_b"
    incident_severity = "medium"
    
    print(f"\nIncident Details:")
    print(f"  Chemical Detected: {detected_chemical}")
    print(f"  Plant Zone: {affected_zone}")
    print(f"  Severity: {incident_severity}")
    
    # Get MSDS information
    chemical_info = msds.get_chemical_info(detected_chemical)
    if chemical_info:
        print(f"\n[MSDS] Chemical: {chemical_info.name}")
        print(f"[MSDS] IDLH Level: {chemical_info.exposure_limits.get('IDLH', 'N/A')} ppm")
        print(f"[MSDS] Required PPE: {', '.join(chemical_info.ppe_requirements[:2])}")
    
    # Get SOP procedures
    sop_procedures = sop.get_procedures(affected_zone, incident_severity)
    print(f"\n[SOP] Emergency Procedures for {affected_zone.upper()} ({incident_severity}):")
    for i, procedure in enumerate(sop_procedures[:4], 1):
        print(f"  {i}. {procedure}")
    
    # Example 4: List all available zones and severities
    print("\n" + "="*80)
    print("Example 4: Available Zones and Severities")
    print("="*80)
    
    print("\nAvailable Plant Zones:")
    for zone in sop.get_all_zones():
        severities = sop.get_severities_for_zone(zone)
        print(f"  {zone.upper()}: {', '.join(severities)}")
    
    print("\nAvailable Chemicals:")
    for chemical in msds.get_all_chemicals()[:5]:
        print(f"  - {chemical}")
    if len(msds.get_all_chemicals()) > 5:
        print(f"  ... and {len(msds.get_all_chemicals()) - 5} more")


if __name__ == "__main__":
    main()
