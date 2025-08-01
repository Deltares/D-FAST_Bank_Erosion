# Bank Erosion Mesh Processor

The Bank Erosion Mesh Processor module provides utilities for processing mesh data in the D-FAST Bank Erosion software.

## Overview

The Bank Erosion Mesh Processor module contains functions for intersecting bank lines with a mesh and processing mesh data. These functions are used by the Bank Erosion module to prepare data for erosion calculations.

## Components

The Bank Erosion Mesh Processor module consists of the following components:

### Mesh Processing Functions

::: dfastbe.bank_erosion.mesh.processor

The mesh processing component provides functions for processing mesh data, such as:

- **intersect_line_mesh**: Intersects a line with a mesh and returns the intersection points and face indices
- **enlarge**: Enlarges an array to a new shape
- **get_slices_ab**: Gets slices between two points
- **_get_slices_core**: Helper function for _get_slices
- **_get_slices**: Helper function for intersect_line_mesh

## Usage Example

```python
from dfastbe.bank_erosion.mesh.processor import MeshWrapper
from dfastbe.bank_erosion.mesh.data_models import MeshData
from dfastbe.io.config import ConfigFile
from dfastbe.bank_erosion.bank_erosion import Erosion


# Load configuration file
config_file = ConfigFile.read("config.cfg")

# Initialize Erosion object
erosion = Erosion(config_file)

# Get mesh data
mesh_data = erosion.simulation_data.compute_mesh_topology()

# Intersect a bank line with the mesh
bank_line_coords = erosion.river_data.bank_lines.geometry[0].coords
coords_along_bank, face_indices = MeshWrapper(mesh_data).intersect_with_coords(bank_line_coords)

# Print results
print(f"Number of intersection points: {len(coords_along_bank)}")
print(f"Number of face indices: {len(face_indices)}")
```

For more details on the specific functions, refer to the API reference below.