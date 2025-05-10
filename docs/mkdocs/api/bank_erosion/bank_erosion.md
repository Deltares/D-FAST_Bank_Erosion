# Bank Erosion Module

The Bank Erosion module is responsible for calculating bank erosion based on hydrodynamic data and detected bank lines. It is one of the core components of the D-FAST Bank Erosion software.

## Overview

The Bank Erosion module calculates the amount of bank material that will be eroded during the first year and until equilibrium, based on hydrodynamic simulation results and detected bank lines. It takes into account various factors such as river geometry, discharge levels, and shipping parameters.

## Components

The Bank Erosion module consists of the following components:

### Main Classes

::: dfastbe.bank_erosion.bank_erosion

### Mesh Processing

::: dfastbe.bank_erosion.mesh_processor

### Debugging Utilities

::: dfastbe.bank_erosion.debugger

### Data Models

The Bank Erosion module uses several data models to represent inputs, calculation parameters, and results:

- **ErosionRiverData**: Represents river data for erosion calculations
- **ErosionSimulationData**: Represents simulation data for erosion calculations
- **BankData**: Represents bank data for erosion calculations
- **SingleCalculation**: Represents parameters for discharge calculations
- **LevelParameters**: Represents parameters for discharge levels
- **DischargeLevels**: Represents discharge levels for erosion calculations
- **ErosionInputs**: Represents inputs for erosion calculations
- **ErosionResults**: Represents results of erosion calculations
- **FairwayData**: Represents fairway data for erosion calculations
- **CalculationLevel**: Represents a calculation level for erosion calculations
- **MeshData**: Represents mesh data for erosion calculations
- **SingleParameters**: Represents parameters for each bank
- **SingleErosion**: Represents a single erosion calculation
- **WaterLevelData**: Represents water level data for erosion calculations

## Workflow

The typical workflow for bank erosion calculation is:

1. Initialize the Erosion object with a configuration file
2. Call the `run` method to start the erosion calculation process
3. The `run` method orchestrates the entire process:
   - Processes the river axis
   - Gets fairway data
   - Calculates bank-fairway distance
   - Prepares initial conditions
   - Processes discharge levels
   - Computes erosion per level
   - Post-processes results
   - Writes output files
   - Generates plots

## Usage Example

```python
from dfastbe.io.config import ConfigFile
from dfastbe.bank_erosion.bank_erosion import Erosion

# Load configuration file
config_file = ConfigFile.read("config.cfg")

# Initialize Erosion object
erosion = Erosion(config_file)

# Run erosion calculation
erosion.run()
```

For more details on the specific methods and classes, refer to the API reference below.
