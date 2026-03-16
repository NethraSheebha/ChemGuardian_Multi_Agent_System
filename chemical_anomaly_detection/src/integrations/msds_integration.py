"""MSDS (Material Safety Data Sheet) integration for chemical information."""

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class ChemicalInfo:
    """Chemical information from MSDS database."""
    
    name: str
    cas_number: str
    exposure_limits: Dict[str, float]  # TWA, STEL, IDLH in ppm
    emergency_procedures: List[str]
    ppe_requirements: List[str]


class MSDSIntegration:
    """Integration class for loading and querying MSDS database."""
    
    def __init__(self, msds_database_path: str):
        """
        Initialize MSDS integration.
        
        Args:
            msds_database_path: Path to MSDS database (JSON or SQLite)
        """
        self.database_path = Path(msds_database_path)
        self.msds_db: Dict[str, ChemicalInfo] = {}
        self._load_msds_database()
    
    def _load_msds_database(self) -> None:
        """Load MSDS database from JSON or SQLite file."""
        if not self.database_path.exists():
            logger.warning(f"MSDS database not found at {self.database_path}, using empty database")
            return
        
        if self.database_path.suffix == '.json':
            self._load_from_json()
        elif self.database_path.suffix in ['.db', '.sqlite', '.sqlite3']:
            self._load_from_sqlite()
        else:
            raise ValueError(f"Unsupported database format: {self.database_path.suffix}")
        
        logger.info(f"Loaded {len(self.msds_db)} chemicals from MSDS database")
    
    def _load_from_json(self) -> None:
        """Load MSDS data from JSON file."""
        try:
            with open(self.database_path, 'r') as f:
                data = json.load(f)
            
            for chemical_name, chemical_data in data.items():
                self.msds_db[chemical_name.lower()] = ChemicalInfo(
                    name=chemical_data.get('name', chemical_name),
                    cas_number=chemical_data.get('cas_number', ''),
                    exposure_limits=chemical_data.get('exposure_limits', {}),
                    emergency_procedures=chemical_data.get('emergency_procedures', []),
                    ppe_requirements=chemical_data.get('ppe_requirements', [])
                )
        except Exception as e:
            logger.error(f"Error loading MSDS database from JSON: {e}")
            raise
    
    def _load_from_sqlite(self) -> None:
        """Load MSDS data from SQLite database."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Query chemicals table
            cursor.execute("""
                SELECT name, cas_number, exposure_limits, 
                       emergency_procedures, ppe_requirements
                FROM chemicals
            """)
            
            for row in cursor.fetchall():
                name, cas_number, exposure_limits, emergency_procedures, ppe_requirements = row
                
                # Parse JSON fields
                exposure_limits_dict = json.loads(exposure_limits) if exposure_limits else {}
                emergency_procedures_list = json.loads(emergency_procedures) if emergency_procedures else []
                ppe_requirements_list = json.loads(ppe_requirements) if ppe_requirements else []
                
                self.msds_db[name.lower()] = ChemicalInfo(
                    name=name,
                    cas_number=cas_number,
                    exposure_limits=exposure_limits_dict,
                    emergency_procedures=emergency_procedures_list,
                    ppe_requirements=ppe_requirements_list
                )
            
            conn.close()
        except Exception as e:
            logger.error(f"Error loading MSDS database from SQLite: {e}")
            raise
    
    def get_chemical_info(self, chemical_name: str) -> Optional[ChemicalInfo]:
        """
        Retrieve MSDS information for a specific chemical.
        
        Args:
            chemical_name: Name of the chemical (case-insensitive)
        
        Returns:
            ChemicalInfo object if found, None otherwise
        """
        normalized_name = chemical_name.lower().strip()
        
        # Direct lookup
        if normalized_name in self.msds_db:
            return self.msds_db[normalized_name]
        
        # Try common aliases
        aliases = {
            'cl2': 'chlorine',
            'nh3': 'ammonia',
            'methyl isocyanate': 'mic',
            'methylisocyanate': 'mic',
        }
        
        if normalized_name in aliases:
            alias = aliases[normalized_name]
            if alias in self.msds_db:
                return self.msds_db[alias]
        
        logger.warning(f"Chemical '{chemical_name}' not found in MSDS database")
        return None
    
    def get_all_chemicals(self) -> List[str]:
        """
        Get list of all chemicals in the database.
        
        Returns:
            List of chemical names
        """
        return [info.name for info in self.msds_db.values()]
