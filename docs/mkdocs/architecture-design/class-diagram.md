# Class Diagram for Existing D-FAST Bank Erosion Code

## Overview

This document provides a class diagram and detailed description of the current architecture of the D-FAST Bank Erosion software. The diagram illustrates the relationships between classes, key methods, and data flow across all modules.

## Class Diagram

```mermaid
classDiagram
    %% Main Classes

    %% Generic Base Class
    class BaseBank~T~ {
        +T left
        +T right
        +Optional[int] id
        +get_bank(int)
        +from_column_arrays(Dict, Type, Tuple)
        +__iter__()
    }
    class Erosion {
        -ConfigFile config_file
        -bool gui
        -Path bank_dir
        -Path output_dir
        -bool debug
        -dict plot_flags
        +__init__(ConfigFile, bool)
        +run()
        -_process_river_axis_by_center_line()
        -_get_fairway_data(LineGeometry, MeshData)
        +calculate_fairway_bank_line_distance(BankData, FairwayData, ErosionSimulationData)
        -_prepare_initial_conditions(ConfigFile, List, FairwayData)
        -_process_discharge_levels(array, tuple, ConfigFile, ErosionInputs, BankData, FairwayData)
        -_postprocess_erosion_results(tuple, array, BankData, ErosionResults)
        +compute_erosion_per_level(int, BankData, ErosionSimulationData, FairwayData, LevelParameters, ErosionInputs, tuple, int, array)
        -_write_bankline_shapefiles(list, list, ConfigFile)
        -_write_volume_outputs(ErosionResults, array)
        -_generate_plots(array, ErosionSimulationData, list, array, float, ErosionInputs, WaterLevelData, MeshData, BankData, ErosionResults)
    }

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

    %% Data Models - Bank Erosion
    class ErosionRiverData {
        -ConfigFile config_file
        -LineString river_center_line
        -Tuple station_bounds
        +__init__(ConfigFile)
        +simulation_data()
        -_get_bank_output_dir()
        -_get_bank_line_dir()
        -_read_river_axis()
    }

    class ErosionSimulationData {
        +compute_mesh_topology()
        +apply_masked_indexing(array, array)
        +calculate_bank_velocity(SingleBank, array)
        +calculate_bank_height(SingleBank, array)
    }

    class BankData {
        +List banks
        +from_column_arrays(dict, Type, GeoDataFrame, int, Tuple)
        +bank_line_coords()
        +is_right_bank()
        +bank_chainage_midpoints()
        +num_stations_per_bank()
    }

    class SingleBank {
        +LineString bank_line
        +array face_indices
        +array chainage
        +bool is_right
        +__post_init__()
        -_segment_length()
        -_dx()
        -_dy()
        +get_mid_points(bool, str)
    }

    class FairwayData {
        +LineString fairway_axis
        +Polygon fairway_polygon
        +array fairway_initial_water_levels
        +array fairway_velocities
        +array fairway_chezy_coefficients
    }

    class ErosionInputs {
        +List banks
        +dict shipping_data
        +from_column_arrays(dict, Type, Dict, array, Tuple)
        +bank_protection_level()
        +tauc()
    }

    class SingleErosion {
        +array wave_fairway_distance_0
        +array wave_fairway_distance_1
        +array bank_protection_level
        +array tauc
        +array bank_type
    }

    class ErosionResults {
        +int erosion_time
        +List velocity
        +List bank_height
        +List water_level
        +List chezy
        +List vol_per_discharge
        +List ship_wave_max
        +List ship_wave_min
        +List line_size
        +List flow_erosion_dist
        +List ship_erosion_dist
        +List total_erosion_dist
        +List total_eroded_vol
        +List eq_erosion_dist
        +List eq_eroded_vol
        +array avg_erosion_rate
        +array eq_eroded_vol_per_km
        +array total_eroded_vol_per_km
    }

    class WaterLevelData {
        +List water_levels
        +array hfw_max
    }

    class MeshData {
        +array x_node
        +array y_node
        +array n_nodes
        +array face_node
        +array face_x
        +array face_y
        +array face_area
        +array face_nodes_count
        +array face_nodes_indices
    }

    class DischargeLevels {
        +List levels
        +__init__(List)
        +__getitem__(int)
        +__len__()
        +append(CalculationLevel)
        +get_max_hfw_level()
        +total_erosion_volume()
        +__iter__()
        +accumulate(str, str)
        -_accumulate_attribute_side(str, str)
        -_get_attr_both_sides_level(str, object)
        +get_attr_level(str)
        +get_water_level_data(array)
    }

    class CalculationLevel {
        +List banks
        +from_column_arrays(dict, Type, float, Tuple)
    }

    class SingleCalculation {
        +array water_level
        +array velocity
        +array chezy
        +array flow_erosion_dist
        +array ship_erosion_dist
        +array total_erosion_dist
        +array total_eroded_vol
        +array eq_erosion_dist
        +array eq_eroded_vol
    }

    class LevelParameters {
        +List banks
    }

    class SingleParameters {
        +float discharge
        +float probability
        +dict ship_parameters
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

    %% Relationships
    Erosion --> ConfigFile : uses
    Erosion --> ErosionRiverData : uses
    Erosion --> ErosionSimulationData : uses
    Erosion --> BankData : uses
    Erosion --> FairwayData : uses
    Erosion --> ErosionInputs : uses
    Erosion --> ErosionResults : uses
    Erosion --> DischargeLevels : uses
    Erosion --> WaterLevelData : uses
    Erosion --> MeshData : uses
    Erosion --> SingleCalculation : uses
    Erosion --> LevelParameters : uses
    Erosion --> CalculationLevel : uses
    Erosion --> SingleParameters : uses
    Erosion --> SingleErosion : uses
    Erosion --> Debugger : uses
    Erosion --> BankLinesProcessor : uses
    Erosion --> LineGeometry : uses

    BankLines --> ConfigFile : uses
    BankLines --> BankLinesRiverData : uses
    BankLines --> SearchLines : uses
    BankLines --> LineGeometry : uses
    BankLines --> BaseSimulationData : uses
    
    SearchLines --> LineGeometry : uses

    ErosionRiverData --|> BaseRiverData : inherits
    ErosionRiverData --> ConfigFile : uses

    BankLinesRiverData --|> BaseRiverData : inherits
    BankLinesRiverData --> ConfigFile : uses
    BankLinesRiverData --> SearchLines : uses
    BankLinesRiverData --> BaseSimulationData : uses

    ErosionSimulationData --|> BaseSimulationData : inherits
    ErosionSimulationData --> MeshData : uses
    ErosionSimulationData --> SingleBank : uses

    %% Inheritance relationships
    BankData --|> BaseBank : inherits
    BankData --|> BaseBank : inherits
    ErosionInputs --|> BaseBank : inherits
    CalculationLevel --|> BaseBank : inherits
    LevelParameters --|> BaseBank : inherits

    %% Containment relationships
    BankData --> SingleBank : contains
    ErosionInputs --> SingleErosion : contains
    CalculationLevel --> SingleCalculation : contains
    LevelParameters --> SingleParameters : contains
    DischargeLevels --> CalculationLevel : contains

    BaseRiverData --> ConfigFile : uses
    BaseRiverData --> LineGeometry : uses
```

## Class Descriptions

### Main Classes

#### Erosion
- **Responsibility**: Handles the calculation of bank erosion based on hydrodynamic data and detected bank lines
- **Key Methods**:
  - `run()`: Executes the complete erosion analysis workflow
  - `compute_erosion_per_level()`: Computes erosion for a specific discharge level
  - `calculate_fairway_bank_line_distance()`: Calculates the distance between bank lines and fairway
- **Dependencies**: ConfigFile, ErosionRiverData, BankData, FairwayData, ErosionInputs, ErosionResults, DischargeLevels, WaterLevelData, MeshData

#### BankLines
- **Responsibility**: Handles the detection of bank lines from hydrodynamic simulation results
- **Key Methods**:
  - `detect()`: Executes the bank line detection workflow
  - `detect_bank_lines()`: Detects bank lines from simulation data
  - `mask()`: Masks bank lines with bank areas
  - `save()`: Saves bank lines to output files
- **Dependencies**: ConfigFile, BankLinesRiverData, SearchLines

#### ConfigFile
- **Responsibility**: Handles configuration file parsing and management
- **Key Methods**:
  - `read()`: Reads a configuration file
  - `write()`: Writes a configuration file
  - `get_parameter()`: Gets a parameter from the configuration
  - `get_river_center_line()`: Gets the river center line from the configuration
- **Dependencies**: None

### Data Models - Bank Erosion

#### BaseBank
- **Responsibility**: Generic base class for representing paired bank data (left and right banks)
- **Key Properties**:
  - `left`: Left bank data (generic type)
  - `right`: Right bank data (generic type)
  - `id`: Optional identifier
- **Key Methods**:
  - `get_bank()`: Gets bank data for a specific bank index (0 for left, 1 for right)
  - `from_column_arrays()`: Creates a BaseBank instance from column arrays
  - `__iter__()`: Allows iteration over banks
- **Dependencies**: None

#### ErosionRiverData
- **Responsibility**: Represents river data for erosion calculations
- **Key Methods**:
  - `simulation_data()`: Gets simulation data for erosion calculations
- **Dependencies**: BaseRiverData, ConfigFile

#### ErosionSimulationData
- **Responsibility**: Represents simulation data for erosion calculations
- **Key Methods**:
  - `compute_mesh_topology()`: Computes mesh topology
  - `calculate_bank_velocity()`: Calculates velocity at bank
  - `calculate_bank_height()`: Calculates height at bank
- **Dependencies**: BaseSimulationData, SingleBank

#### BankData
- **Responsibility**: Represents bank data for erosion calculations
- **Key Properties**:
  - `left`: Left bank (SingleBank)
  - `right`: Right bank (SingleBank)
  - `id`: Optional identifier
- **Inheritance**: Inherits from BaseBank[SingleBank]
- **Dependencies**: BaseBank, SingleBank

#### SingleBank
- **Responsibility**: Represents a single bank for erosion calculations
- **Key Properties**:
  - `bank_line`: Bank line geometry
  - `face_indices`: Indices of faces adjacent to the bank
  - `chainage`: Chainage along the bank
  - `is_right`: Whether the bank is a right bank
- **Dependencies**: None

#### FairwayData
- **Responsibility**: Represents fairway data for erosion calculations
- **Key Properties**:
  - `fairway_axis`: Fairway axis geometry
  - `fairway_polygon`: Fairway polygon geometry
  - `fairway_initial_water_levels`: Initial water levels in the fairway
  - `fairway_velocities`: Velocities in the fairway
  - `fairway_chezy_coefficients`: Chezy coefficients in the fairway
- **Dependencies**: None

#### ErosionInputs
- **Responsibility**: Represents inputs for erosion calculations
- **Key Properties**:
  - `left`: Left bank erosion inputs (SingleErosion)
  - `right`: Right bank erosion inputs (SingleErosion)
  - `id`: Optional identifier
  - `shipping_data`: Shipping data for erosion calculations
- **Inheritance**: Inherits from BaseBank[SingleErosion]
- **Dependencies**: BaseBank, SingleErosion

#### SingleErosion
- **Responsibility**: Represents erosion inputs for a single bank
- **Key Properties**:
  - `wave_fairway_distance_0`: Distance from fairway for wave calculations
  - `wave_fairway_distance_1`: Distance from fairway for wave calculations
  - `bank_protection_level`: Bank protection level
  - `tauc`: Critical shear stress
  - `bank_type`: Bank type
- **Dependencies**: None

#### ErosionResults
- **Responsibility**: Represents results of erosion calculations
- **Key Properties**:
  - Various lists and arrays storing erosion-related data
- **Dependencies**: None

#### Debugger
- **Responsibility**: Handles debugging and output of bank erosion calculations
- **Key Methods**:
  - `last_discharge_level()`: Writes the last discharge level to a shapefile and CSV file
  - `middle_levels()`: Writes the middle levels to a shapefile and CSV file
  - `_write_data()`: Writes data to a shapefile and CSV file
- **Dependencies**: None

#### BankLinesProcessor
- **Responsibility**: Processes bank lines and intersects them with a mesh
- **Key Methods**:
  - `intersect_with_mesh()`: Intersects bank lines with a mesh and returns bank data
- **Dependencies**: ErosionRiverData, LineGeometry

#### WaterLevelData
- **Responsibility**: Represents water level data for erosion calculations
- **Key Properties**:
  - `water_levels`: List of water levels
  - `hfw_max`: Maximum water level
- **Dependencies**: None

#### MeshData
- **Responsibility**: Represents mesh data for erosion calculations
- **Key Properties**:
  - Various arrays storing mesh-related data
- **Dependencies**: None

#### DischargeLevels
- **Responsibility**: Represents discharge levels for erosion calculations
- **Key Properties**:
  - `levels`: List of CalculationLevel objects
- **Dependencies**: CalculationLevel

#### CalculationLevel
- **Responsibility**: Represents a calculation level for erosion calculations
- **Key Properties**:
  - `left`: Left bank calculation parameters (SingleCalculation)
  - `right`: Right bank calculation parameters (SingleCalculation)
  - `id`: Optional identifier
- **Inheritance**: Inherits from BaseBank[SingleCalculation]
- **Dependencies**: BaseBank, SingleCalculation

#### SingleCalculation
- **Responsibility**: Represents parameters for discharge calculations
- **Key Properties**:
  - Various arrays storing discharge-related data
- **Dependencies**: None

#### LevelParameters
- **Responsibility**: Represents parameters for discharge levels
- **Key Properties**:
  - `left`: Left bank parameters (SingleParameters)
  - `right`: Right bank parameters (SingleParameters)
  - `id`: Optional identifier
- **Inheritance**: Inherits from BaseBank[SingleParameters]
- **Dependencies**: BaseBank, SingleParameters

#### SingleParameters
- **Responsibility**: Represents parameters for each bank
- **Key Properties**:
  - `discharge`: Discharge value
  - `probability`: Probability of discharge
  - `ship_parameters`: Ship parameters
- **Dependencies**: None

### Data Models - Bank Lines

#### BankLinesRiverData
- **Responsibility**: Represents river data for bank line detection
- **Key Methods**:
  - `search_lines()`: Gets search lines for bank line detection
  - `simulation_data()`: Gets simulation data for bank line detection
- **Dependencies**: BaseRiverData, ConfigFile, SearchLines

#### SearchLines
- **Responsibility**: Represents search lines for bank line detection
- **Key Methods**:
  - `mask()`: Masks search lines with river center line
  - `to_polygons()`: Converts search lines to polygons
- **Dependencies**: LineGeometry

### Data Models - IO

#### LineGeometry
- **Responsibility**: Represents line geometry
- **Key Methods**:
  - `mask()`: Masks line geometry
  - `intersect_with_line()`: Intersects line geometry with another line
- **Dependencies**: None

#### BaseSimulationData
- **Responsibility**: Base class for simulation data
- **Key Methods**:
  - `read()`: Reads simulation data from a file
  - `clip()`: Clips simulation data to a region
- **Dependencies**: None

#### BaseRiverData
- **Responsibility**: Base class for river data
- **Key Methods**:
  - `get_erosion_sim_data()`: Gets simulation data for erosion calculations
- **Dependencies**: ConfigFile, LineGeometry

## Data Flow

1. The user initializes the `Erosion` or `BankLines` class with a `ConfigFile`
2. The `ConfigFile` is used to read configuration parameters
3. For bank line detection:
   - `BankLines` uses `BankLinesRiverData` to get river data
   - `BankLinesRiverData` uses `SearchLines` to get search lines
   - `BankLines` detects bank lines and saves them to output files
4. For erosion calculation:
   - `Erosion` uses `ErosionRiverData` to get river data
   - `Erosion` processes the river axis and gets fairway data
   - `Erosion` prepares initial conditions and processes discharge levels
   - `Erosion` computes erosion per level and post-processes results
   - `Erosion` writes output files and generates plots

This architecture provides a clear separation of concerns, with each class having a specific responsibility. The data flows through the system in a logical manner, with each step building on the previous one.
