# Bank Lines Module

The Bank Lines module is responsible for detecting bank lines from hydrodynamic simulation results. It is one of the core components of the D-FAST Bank Erosion software.

## Overview

The Bank Lines module processes hydrodynamic simulation results to detect bank lines, which are the boundaries between wet and dry areas in the river. These bank lines are then used as input for bank erosion calculations. The module can detect bank lines for multiple simulations and combine them into a single set of bank lines.

```mermaid
classDiagram
    %% Main Classes

    class BankLines {
        -ConfigFile config_file
        -bool gui
        -Path output_dir
        -bool debug
        -dict plot_flags
        -float max_river_width
        +__init__(ConfigFile, bool)
        +detect()
        +mask(GeoSeries, Polygon)
        +plot(array, int, List, Tuple, List, ConfigFile)
        +save(List, GeoSeries, List, List, ConfigFile)
        +detect_bank_lines(BaseSimulationData, float, ConfigFile)
        -_calculate_water_depth(BaseSimulationData)
        -_generate_bank_lines(BaseSimulationData, array, array, array, float)
        -_progress_bar(int, int)
    }

    class ConfigFile {
        -ConfigParser config
        -str path
        +__init__(ConfigParser, Path)
        +read(Path)
        +write(str)
        +make_paths_absolute()
        +get_str(str, str, str)
        +get_bool(str, str, bool)
        +get_float(str, str, float, bool)
        +get_int(str, str, int, bool)
        +get_sim_file(str, str)
        +get_start_end_stations()
        +get_search_lines()
        +read_bank_lines(str)
        +get_parameter(str, str, List, Any, str, bool, List, bool)
        +get_bank_search_distances(int)
        +get_range(str, str)
        +get_river_center_line()
        +resolve(str)
        +relative_to(str)
        +get_plotting_flags(Path)
        +get_output_dir(str)
    }
    
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


    BankLines --> ConfigFile : uses
    BankLines --> BankLinesRiverData : uses
    BankLines --> SearchLines : uses
    BankLines --> LineGeometry : uses
    BankLines --> BaseSimulationData : uses

    SearchLines --> LineGeometry : uses

    BankLinesRiverData --|> BaseRiverData : inherits
    BankLinesRiverData --> ConfigFile : uses
    BankLinesRiverData --> SearchLines : uses
    BankLinesRiverData --> BaseSimulationData : uses

    BaseRiverData --> ConfigFile : uses
    BaseRiverData --> LineGeometry : uses
    
```

## Components

The Bank Lines module consists of the following components:

### Main Classes

::: dfastbe.bank_lines.bank_lines

### Data Models

::: dfastbe.bank_lines.data_models

### Utility Functions

The Bank Lines module includes several utility functions for processing bank lines:

- **sort_connect_bank_lines**: Sorts and connects bank line fragments
- **poly_to_line**: Converts polygons to lines
- **tri_to_line**: Converts triangles to lines

## Workflow

The typical workflow for bank line detection is:

1. Initialize the BankLines object with a configuration file
2. Call the `detect` method to start the bank line detection process
3. The `detect` method orchestrates the entire process:
   - Loads hydrodynamic simulation data
   - Calculates water depth
   - Generates bank lines
   - Masks bank lines with bank areas
   - Saves bank lines to output files
   - Generates plots

## Usage Example

```python
from dfastbe.io.config import ConfigFile
from dfastbe.bank_lines.bank_lines import BankLines

# Load configuration file
config_file = ConfigFile.read("config.cfg")

# Initialize BankLines object
bank_lines = BankLines(config_file)

# Run bank line detection
bank_lines.detect()
```

For more details on the specific methods and classes, refer to the API reference below.
