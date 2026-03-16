# Integration Modules

This directory contains integration modules for external data sources used by the Chemical Leak Monitoring System.

## MSDS Integration

The `MSDSIntegration` class provides access to Material Safety Data Sheet (MSDS) information for hazardous chemicals.

### Features

- Load MSDS data from JSON or SQLite databases
- Retrieve chemical information including:
  - CAS numbers
  - Exposure limits (TWA, STEL, IDLH)
  - Emergency procedures
  - PPE requirements
- Case-insensitive chemical name lookup
- Support for common chemical aliases

### Usage

```python
from src.integrations import MSDSIntegration

# Initialize with database path
msds = MSDSIntegration('data/msds_database.json')

# Get chemical information
chlorine_info = msds.get_chemical_info('chlorine')

if chlorine_info:
    print(f"Chemical: {chlorine_info.name}")
    print(f"CAS: {chlorine_info.cas_number}")
    print(f"IDLH: {chlorine_info.exposure_limits['IDLH']} ppm")
    print(f"Emergency Procedures: {chlorine_info.emergency_procedures}")
    print(f"PPE Required: {chlorine_info.ppe_requirements}")

# List all available chemicals
chemicals = msds.get_all_chemicals()
```

### Supported Chemicals

The default database includes:

- Chlorine (Cl2)
- Ammonia (NH3)
- Methyl Isocyanate (MIC)
- Hydrogen Sulfide (H2S)
- Sulfur Dioxide (SO2)
- Hydrochloric Acid (HCl)

### Database Format

#### JSON Format

```json
{
  "chlorine": {
    "name": "Chlorine",
    "cas_number": "7782-50-5",
    "exposure_limits": {
      "TWA": 0.5,
      "STEL": 1.0,
      "IDLH": 10.0
    },
    "emergency_procedures": [
      "Evacuate area immediately",
      "Activate emergency ventilation systems"
    ],
    "ppe_requirements": [
      "Self-contained breathing apparatus (SCBA)",
      "Chemical-resistant suit"
    ]
  }
}
```

#### SQLite Format

Table: `chemicals`

| Column               | Type | Description                      |
| -------------------- | ---- | -------------------------------- |
| name                 | TEXT | Chemical name                    |
| cas_number           | TEXT | CAS registry number              |
| exposure_limits      | TEXT | JSON object with TWA, STEL, IDLH |
| emergency_procedures | TEXT | JSON array of procedures         |
| ppe_requirements     | TEXT | JSON array of PPE items          |

## SOP Integration

The `SOPIntegration` class provides access to Standard Operating Procedures (SOPs) for emergency response.

### Features

- Load SOP data from JSON or SQLite databases
- Retrieve zone-specific and severity-specific procedures
- Support for three severity levels: mild, medium, high
- Case-insensitive zone and severity lookup
- List available zones and severities

### Usage

```python
from src.integrations import SOPIntegration

# Initialize with database path
sop = SOPIntegration('data/sop_database.json')

# Get procedures for specific zone and severity
procedures = sop.get_procedures('zone_a', 'high')

for i, procedure in enumerate(procedures, 1):
    print(f"{i}. {procedure}")

# List all available zones
zones = sop.get_all_zones()

# Get severities available for a zone
severities = sop.get_severities_for_zone('zone_a')
```

### Supported Zones

The default database includes:

- Zone A: General processing area
- Zone B: Reactor area (contains MIC)
- Zone C: Storage area
- Zone D: Utilities and equipment
- Control Room: Central monitoring

### Severity Levels

- **Mild**: Minor incidents requiring monitoring and documentation
- **Medium**: Significant incidents requiring evacuation and containment
- **High**: Critical incidents requiring emergency shutdown and external notification

### Database Format

#### JSON Format

```json
{
  "zone_a": {
    "mild": [
      "Notify shift supervisor immediately",
      "Increase monitoring frequency"
    ],
    "medium": [
      "Activate emergency ventilation",
      "Evacuate non-essential personnel"
    ],
    "high": ["ACTIVATE EMERGENCY ALARM", "Evacuate all personnel immediately"]
  }
}
```

#### SQLite Format

Table: `sops`

| Column     | Type | Description                       |
| ---------- | ---- | --------------------------------- |
| plant_zone | TEXT | Zone identifier                   |
| severity   | TEXT | Severity level (mild/medium/high) |
| procedures | TEXT | JSON array of procedure strings   |

## Integration with Response Agents

These integration modules are used by the Risk Response Agents to provide context-specific emergency response information:

```python
from src.integrations import MSDSIntegration, SOPIntegration

# Initialize integrations
msds = MSDSIntegration('data/msds_database.json')
sop = SOPIntegration('data/sop_database.json')

# In response agent
def execute_response(chemical_detected, plant_zone, severity):
    # Get MSDS information
    chemical_info = msds.get_chemical_info(chemical_detected)

    # Get SOPs
    procedures = sop.get_procedures(plant_zone, severity)

    # Execute response with integrated information
    print(f"Chemical: {chemical_info.name}")
    print(f"IDLH: {chemical_info.exposure_limits['IDLH']} ppm")
    print(f"PPE: {chemical_info.ppe_requirements}")
    print(f"Procedures: {procedures}")
```

## Testing

Unit tests are available in the `tests/` directory:

```bash
# Test MSDS integration
pytest tests/test_msds_integration.py -v

# Test SOP integration
pytest tests/test_sop_integration.py -v

# Test both
pytest tests/test_msds_integration.py tests/test_sop_integration.py -v
```

## Example

See `examples/msds_sop_integration_example.py` for a complete demonstration:

```bash
python examples/msds_sop_integration_example.py
```

## Requirements

- Python 3.8+
- No external dependencies (uses standard library only)

## Error Handling

Both integration classes handle missing files gracefully:

- If the database file doesn't exist, an empty database is initialized with a warning
- Invalid file formats raise `ValueError` with descriptive messages
- Missing chemicals or zones return `None` or empty lists with warnings logged

## Logging

Both classes use Python's standard logging module:

```python
import logging

# Configure logging to see integration messages
logging.basicConfig(level=logging.INFO)
```

Log messages include:

- Database loading success/failure
- Number of items loaded
- Lookup failures (chemical not found, zone not found)
- Invalid severity levels
