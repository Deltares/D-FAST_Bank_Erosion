# Plotting Module

The Plotting module provides functions for generating visualizations of bank lines, erosion results, and other data in the D-FAST Bank Erosion software.

## Overview

The Plotting module contains functions for creating various types of plots and visualizations that help users understand the results of bank line detection and erosion calculations. These visualizations include maps of bank lines, erosion profiles, and time series of erosion volumes.

## Components

The Plotting module consists of the following components:

### Plotting Functions

::: dfastbe.plotting

The plotting functions component provides functions for creating various types of visualizations, such as:

- Maps of bank lines and erosion results
- Profiles of bank erosion
- Time series of erosion volumes
- Visualizations of hydrodynamic data

## Workflow

The typical workflow for using the Plotting module is:

1. Perform bank line detection or erosion calculation
2. Call the appropriate plotting functions to visualize the results
3. Display the plots or save them to files

## Usage Example

```python
import matplotlib.pyplot as plt
from dfastbe import plotting as df_plt
from dfastbe.io.config import ConfigFile
from dfastbe.bank_erosion.bank_erosion import Erosion

# Load configuration file
config_file = ConfigFile.read("config.cfg")

# Initialize Erosion object
erosion = Erosion(config_file)

# Run erosion calculation
erosion.run()

# Create a plot of the results
fig, ax = plt.subplots(figsize=(10, 8))
df_plt.plot_bank_lines(ax, bank_lines, color='blue', linewidth=1.5)
df_plt.plot_erosion_results(ax, erosion_results, cmap='viridis')
plt.savefig("erosion_results.png")
plt.show()
```

For more details on the specific functions, refer to the API reference below.
