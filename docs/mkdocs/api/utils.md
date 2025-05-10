# Utilities Module

The Utilities module provides general utility functions used across the D-FAST Bank Erosion software. These functions support various operations in the Bank Lines and Bank Erosion modules.

## Overview

The Utilities module contains a collection of helper functions that are used by multiple components of the D-FAST Bank Erosion software. These functions handle common tasks such as geometric operations, data processing, and visualization support.

## Components

The Utilities module consists of the following components:

### General Utilities

::: dfastbe.utils

The general utilities component provides functions for various common tasks, such as:

- Geometric operations (e.g., checking if a point is on the right side of a line)
- Visualization support (e.g., getting zoom extents for plots)
- Data processing (e.g., interpolation, filtering)

## Usage Example

```python
from dfastbe.utils import get_zoom_extends, on_right_side

# Check if a point is on the right side of a line
is_on_right = on_right_side(line_start, line_end, point)

# Get zoom extents for plotting
x_min, x_max, y_min, y_max = get_zoom_extends(x, y, margin=0.1)
```

For more details on the specific functions, refer to the API reference below.
