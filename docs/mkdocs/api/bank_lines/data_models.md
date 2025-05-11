# Bank Lines Data Models

The Bank Lines Data Models module provides data structures for representing bank lines and related data in the D-FAST Bank Erosion software.

## Overview

The Bank Lines Data Models module contains classes that represent various aspects of bank lines, such as river data, simulation data, and bank line geometry. These data models are used by the Bank Lines module to process and analyze bank lines.

```mermaid
classDiagram
    %% Main Classes

    %% Data Models - Bank Lines
    class BankLinesRiverData {
        -ConfigFile config_file
        -LineString river_center_line
        -Tuple station_bounds
        +search_lines()
        -_get_bank_lines_simulation_data()
        +simulation_data()
    }

    class SearchLines {
        +List lines
        +LineGeometry mask
        +__init__(List, LineGeometry)
        +mask(List, LineString, float)
        -_select_closest_part(MultiLineString, LineString, float)
        +to_polygons()
    }

    %% Data Models - IO
    class LineGeometry {
        +LineString line
        +dict data
        +__init__(LineString, Tuple, str)
        +as_array()
        +add_data(Dict)
        +to_file(str, Dict)
        +mask(LineString, Tuple)
        -_find_mask_index(float, array)
        -_handle_bound(int, float, bool, array)
        -_interpolate_point(int, float, array)
        +intersect_with_line(array)
    }

    class BaseSimulationData {
        +array x_node
        +array y_node
        +array n_nodes
        +array face_node
        +array bed_elevation_location
        +array bed_elevation_values
        +array water_level_face
        +array water_depth_face
        +array velocity_x_face
        +array velocity_y_face
        +array chezy_face
        +float dry_wet_threshold
        +__init__(array, array, array, array, array, array, array, array, array, array, array, float)
        +read(str, str)
        +clip(LineString, float)
    }

    class BaseRiverData {
        -ConfigFile config_file
        -LineString river_center_line
        -Tuple station_bounds
        +__init__(ConfigFile)
        +get_bbox(array, float)
        +get_erosion_sim_data(int)
    }

    SearchLines --> LineGeometry : uses

    BankLinesRiverData --|> BaseRiverData : inherits
    BankLinesRiverData --> ConfigFile : uses
    BankLinesRiverData --> SearchLines : uses
    BankLinesRiverData --> BaseSimulationData : uses

    BaseRiverData --> ConfigFile : uses
    BaseRiverData --> LineGeometry : uses
    
```

## Components

The Bank Lines Data Models module consists of the following components:

### Data Models

::: dfastbe.bank_lines.data_models

The data models component provides classes for representing various types of data related to bank lines, such as:

- **BankLinesRiverData**: Represents river data for bank line detection
- **BankLineGeometry**: Represents the geometry of a bank line
- **BankLineProperties**: Represents properties of a bank line

## Usage Example

```python
from dfastbe.bank_lines.data_models import BankLinesRiverData
from dfastbe.io.config import ConfigFile

# Load configuration file
config_file = ConfigFile.read("config.cfg")

# Create river data object
river_data = BankLinesRiverData(config_file)

# Access river data properties
print(f"River name: {river_data.name}")
print(f"River length: {river_data.length} km")
```

For more details on the specific classes and their properties, refer to the API reference below.
