"""SOP (Standard Operating Procedure) integration for emergency response procedures."""

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class SOPIntegration:
    """Integration class for loading and querying SOP database."""
    
    def __init__(self, sop_database_path: str):
        """
        Initialize SOP integration.
        
        Args:
            sop_database_path: Path to SOP database (JSON or SQLite)
        """
        self.database_path = Path(sop_database_path)
        self.sop_db: Dict[str, Dict[str, List[str]]] = {}
        self._load_sop_database()
    
    def _load_sop_database(self) -> None:
        """Load SOP database from JSON or SQLite file."""
        if not self.database_path.exists():
            logger.warning(f"SOP database not found at {self.database_path}, using empty database")
            return
        
        if self.database_path.suffix == '.json':
            self._load_from_json()
        elif self.database_path.suffix in ['.db', '.sqlite', '.sqlite3']:
            self._load_from_sqlite()
        else:
            raise ValueError(f"Unsupported database format: {self.database_path.suffix}")
        
        logger.info(f"Loaded SOPs for {len(self.sop_db)} plant zones from database")
    
    def _load_from_json(self) -> None:
        """Load SOP data from JSON file."""
        try:
            with open(self.database_path, 'r') as f:
                data = json.load(f)
            
            # Expected structure: {plant_zone: {severity: [procedures]}}
            for plant_zone, severity_dict in data.items():
                self.sop_db[plant_zone.lower()] = {}
                for severity, procedures in severity_dict.items():
                    self.sop_db[plant_zone.lower()][severity.lower()] = procedures
        except Exception as e:
            logger.error(f"Error loading SOP database from JSON: {e}")
            raise
    
    def _load_from_sqlite(self) -> None:
        """Load SOP data from SQLite database."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Query SOPs table
            cursor.execute("""
                SELECT plant_zone, severity, procedures
                FROM sops
            """)
            
            for row in cursor.fetchall():
                plant_zone, severity, procedures = row
                
                # Parse JSON procedures field
                procedures_list = json.loads(procedures) if procedures else []
                
                # Initialize plant_zone dict if not exists
                if plant_zone.lower() not in self.sop_db:
                    self.sop_db[plant_zone.lower()] = {}
                
                self.sop_db[plant_zone.lower()][severity.lower()] = procedures_list
            
            conn.close()
        except Exception as e:
            logger.error(f"Error loading SOP database from SQLite: {e}")
            raise
    
    def get_procedures(self, plant_zone: str, severity: str) -> List[str]:
        """
        Retrieve zone-specific SOPs for given severity level.
        
        Args:
            plant_zone: Plant zone identifier (case-insensitive)
            severity: Severity level - 'mild', 'medium', or 'high' (case-insensitive)
        
        Returns:
            List of procedure strings, empty list if not found
        """
        normalized_zone = plant_zone.lower().strip()
        normalized_severity = severity.lower().strip()
        
        # Validate severity
        if normalized_severity not in ['mild', 'medium', 'high']:
            logger.warning(f"Invalid severity level: {severity}")
            return []
        
        # Check if zone exists
        if normalized_zone not in self.sop_db:
            logger.warning(f"Plant zone '{plant_zone}' not found in SOP database")
            return []
        
        # Check if severity exists for this zone
        if normalized_severity not in self.sop_db[normalized_zone]:
            logger.warning(f"No SOPs found for zone '{plant_zone}' with severity '{severity}'")
            return []
        
        return self.sop_db[normalized_zone][normalized_severity]
    
    def get_all_zones(self) -> List[str]:
        """
        Get list of all plant zones in the database.
        
        Returns:
            List of plant zone identifiers
        """
        return list(self.sop_db.keys())
    
    def get_severities_for_zone(self, plant_zone: str) -> List[str]:
        """
        Get list of severity levels available for a specific plant zone.
        
        Args:
            plant_zone: Plant zone identifier (case-insensitive)
        
        Returns:
            List of severity levels, empty list if zone not found
        """
        normalized_zone = plant_zone.lower().strip()
        
        if normalized_zone not in self.sop_db:
            return []
        
        return list(self.sop_db[normalized_zone].keys())
