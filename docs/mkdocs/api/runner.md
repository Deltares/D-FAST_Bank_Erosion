# Command Module

The Command module serves as the main entry point for the D-FAST Bank Erosion software, handling command-line arguments and orchestrating the execution of the different operational modes.

## Overview

The Command module provides the interface between the user (via command-line or GUI) and the core functionality of the D-FAST Bank Erosion software. It parses command-line arguments, initializes the appropriate language settings, and launches the requested operational mode (BANKLINES, BANKEROSION, or GUI).

## Components

The Command module consists of the following components:

### Run Function

::: dfastbe.runner

The `run` function is the main entry point for the D-FAST Bank Erosion software. It initializes the language settings and launches the requested operational mode.

## Operational Modes

The D-FAST Bank Erosion software supports three operational modes:

1. **BANKLINES**: Detects bank lines from hydrodynamic simulation results
2. **BANKEROSION**: Calculates bank erosion based on detected bank lines and hydrodynamic data
3. **GUI**: Provides a graphical user interface for configuring and running the above processes

## Workflow

The typical workflow for using the Command module is:

1. Call the `run` function with the desired language, run mode, and configuration file
2. The `run` function initializes the language settings
3. Depending on the run mode, the `run` function:
   - Launches the GUI
   - Runs bank line detection
   - Runs bank erosion calculation

## Usage Example

```python
from dfastbe.runner import Runner

# Run in GUI mode with English language
runner = Runner(language="UK", run_mode="GUI", configfile="config.cfg")
runner.run()

# Run bank line detection with Dutch language
runner = Runner(language="NL", run_mode="BANKLINES", configfile="config.cfg")
runner.run()
# Run bank erosion calculation with English language
runner = Runner(language="UK", run_mode="BANKEROSION", configfile="config.cfg")
runner.run()
```

For more details on the specific functions, refer to the API reference below.
