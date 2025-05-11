# Bank Erosion Calculation Data Models

The Bank Erosion Calculation Data Models module provides data structures for representing calculation parameters, inputs, and results in the D-FAST Bank Erosion software.

## Overview

The Bank Erosion Calculation Data Models module contains classes that represent various aspects of bank erosion calculations, such as bank data, erosion inputs, calculation parameters, and results. These data models are used by the Bank Erosion module to process and analyze bank erosion.

## Components

The Bank Erosion Calculation Data Models module consists of the following components:

### Data Models

::: dfastbe.bank_erosion.data_models.calculation

The data models component provides classes for representing various types of data related to bank erosion calculations, such as:

- **BaseBank**: Generic base class for representing paired bank data (left and right banks)
- **SingleErosion**: Represents erosion inputs for a single bank
- **ErosionInputs**: Represents inputs for erosion calculations
- **WaterLevelData**: Represents water level data for erosion calculations
- **MeshData**: Represents mesh data for erosion calculations
- **SingleBank**: Represents a single bank for erosion calculations
- **BankData**: Represents bank data for erosion calculations
- **FairwayData**: Represents fairway data for erosion calculations
- **ErosionResults**: Represents results of erosion calculations
- **SingleParameters**: Represents parameters for each bank
- **SingleLevelParameters**: Represents parameters for discharge levels
- **SingleCalculation**: Represents parameters for discharge calculations
- **SingleDischargeLevel**: Represents a calculation level for erosion calculations
- **DischargeLevels**: Represents discharge levels for erosion calculations

## Usage Example

```python
from dfastbe.bank_erosion.data_models.calculation import BankData, ErosionInputs, ErosionResults
from dfastbe.io.config import ConfigFile
from dfastbe.bank_erosion.bank_erosion import Erosion

# Load configuration file
config_file = ConfigFile.read("config.cfg")

# Initialize Erosion object
erosion = Erosion(config_file)

# Access bank data
bank_data = erosion.bl_processor.intersect_with_mesh(erosion.simulation_data.mesh_data)

# Print bank data properties
print(f"Number of bank lines: {bank_data.n_bank_lines}")
print(f"Left bank is right bank: {bank_data.left.is_right}")
print(f"Right bank is right bank: {bank_data.right.is_right}")
```

For more details on the specific classes and their properties, refer to the API reference below.