# Bank Erosion Debugger

The Bank Erosion Debugger module provides utilities for debugging and outputting bank erosion calculations in the D-FAST Bank Erosion software.

## Overview

The Bank Erosion Debugger module contains a class and utility functions for writing debug information about bank erosion calculations to shapefiles and CSV files. This information can be used to analyze and troubleshoot bank erosion calculations.

## Components

The Bank Erosion Debugger module consists of the following components:

### Debugger Class

:::: dfastbe.bank_erosion.debugger

The Debugger class provides methods for writing debug information about bank erosion calculations, such as:

- **last_discharge_level**: Writes debug information about the last discharge level to a shapefile and CSV file
- **middle_levels**: Writes debug information about the middle discharge levels to a shapefile and CSV file
- **_write_data**: Writes data to a shapefile and CSV file

### Utility Functions

The Bank Erosion Debugger module includes utility functions for writing data to files:

- **_write_shp**: Writes data to a shapefile
- **_write_csv**: Writes data to a CSV file

## Usage Example

```python
from dfastbe.bank_erosion.debugger import Debugger
from dfastbe.io.config import ConfigFile
from dfastbe.bank_erosion.bank_erosion import Erosion

# Load configuration file
config_file = ConfigFile.read("config.cfg")

# Initialize Erosion object
erosion = Erosion(config_file)

# Access debugger
debugger = erosion.debugger

# Write debug information for the last discharge level
debugger.last_discharge_level(
    bank_index=0,
    single_bank=bank_data.left,
    fairway_data=fairway_data,
    erosion_inputs=erosion_inputs.left,
    discharge_level_pars=level_parameters.left,
    water_depth_fairway=water_depth_fairway,
    dn_eq1=dn_eq1,
    dv_eq1=dv_eq1,
    bank_height=bank_height
)
```

For more details on the specific classes and functions, refer to the API reference below.