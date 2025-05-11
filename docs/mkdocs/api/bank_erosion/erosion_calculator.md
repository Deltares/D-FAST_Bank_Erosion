# Bank Erosion Calculator

The Bank Erosion Calculator module provides functionality for calculating bank erosion in the D-FAST Bank Erosion software.

## Overview

The Bank Erosion Calculator module contains a class that encapsulates the core erosion calculation algorithms. It is responsible for computing equilibrium bank erosion, bank erosion dynamics during specific discharge levels, and wave heights at banks due to passing ships.

## Components

The Bank Erosion Calculator module consists of the following components:

### ErosionCalculator Class

::: dfastbe.bank_erosion.erosion_calculator

The ErosionCalculator class provides methods for calculating bank erosion, such as:

- **comp_erosion_eq**: Computes the equilibrium bank erosion distance and volume
- **compute_bank_erosion_dynamics**: Computes the bank erosion during a specific discharge level
- **comp_hw_ship_at_bank**: Computes wave heights at bank due to passing ships

## Usage Example

```python
from dfastbe.bank_erosion.erosion_calculator import ErosionCalculator
from dfastbe.bank_erosion.data_models.calculation import SingleCalculation, SingleParameters, SingleErosion
import numpy as np

# Initialize ErosionCalculator
calculator = ErosionCalculator()

# Compute equilibrium bank erosion
erosion_distance_eq, erosion_volume_eq = calculator.comp_erosion_eq(
    bank_height=np.array([5.0, 5.5, 6.0]),
    segment_length=np.array([10.0, 10.0, 10.0]),
    water_level_fairway_ref=np.array([2.0, 2.0, 2.0]),
    discharge_level_pars=discharge_level_pars,
    bank_fairway_dist=np.array([20.0, 25.0, 30.0]),
    water_depth_fairway=np.array([3.0, 3.5, 4.0]),
    erosion_inputs=erosion_inputs
)

# Compute bank erosion dynamics
parameter = SingleCalculation()
parameter.bank_velocity = np.array([1.0, 1.2, 1.5])
parameter.water_level = np.array([2.0, 2.0, 2.0])
parameter.chezy = np.array([50.0, 50.0, 50.0])

parameter = calculator.compute_bank_erosion_dynamics(
    parameter,
    bank_height=np.array([5.0, 5.5, 6.0]),
    segment_length=np.array([10.0, 10.0, 10.0]),
    bank_fairway_dist=np.array([20.0, 25.0, 30.0]),
    water_level_fairway_ref=np.array([2.0, 2.0, 2.0]),
    discharge_level_pars=discharge_level_pars,
    time_erosion=1.0,
    water_depth_fairway=np.array([3.0, 3.5, 4.0]),
    erosion_inputs=erosion_inputs
)
```

For more details on the specific methods and their parameters, refer to the API reference below.