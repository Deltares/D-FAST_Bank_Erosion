## 3.1.0 (2026-06-11)

### Feat

- migrate from PyQt5 to PySide6 and expand Python version support (#177)
- migrate from PyQt5 to PySide6 and expand Python version support
- **gui**: add and test GUI utilities, improve input validation and documentation (#175)
- **gui**: add and test GUI utilities, improve input validation and documentation

### Refactor

- **configs**: extract get_configuration into ConfigurationExporter (#193)
- **configs**: replace load_configuration with ConfigurationLoader (#190)
- **gui**: Refactor GUI architecture (#180)
- **gui**: Refactor GUI architecture (#180)
- **logging**: migrate logging functions to LogData singleton class (#166)
- **logging**: migrate logging functions to LogData singleton class

## 3.0.1 (2026-01-05)

## 3.0.0 (2025-09-04)

### BREAKING CHANGE

- Refactored structure and method signatures across the `bank_erosion` module.
- Refactored structure and APIs of river and bank erosion modules, including class renaming (`CenterLine` -> `LineGeometry`) and method signature changes.
- Public APIs now expect/return `BankData`, `FairwayData`, and
`ErosionResults`. Several parameter names have changed (see above), so downstream
code must be updated accordingly.
- The package no longer supports Python 3.10.

### Feat

- **ci**: restructure and move TeamCity configurations and templates to the github repo (#87)
- **ci**: restructure and move TeamCity configurations and templates to the github repo

### Fix

- **gui**: PDF manual opening security issue (#139)
- **gui**: improve cross-platform support and robustness of PDF manual opening
- nuitka build fails on certifi dependency
- binaries tests and teamcity build steps (#71) (dfast-285)
- binary tests failing in team city
- readme, and mkdocs (#56) (dfast-272)
- restructure main directory (#10)

### Refactor

- **erosion**: introduce `Parameters` namedtuple and simplify erosion input handling (#149)
- **erosion**: introduce `Parameters` namedtuple and simplify erosion input handling
- **mesh**: refine MeshWrapper API and improve intersection handling structure (#148)
- **mesh**: refine MeshWrapper API and improve intersection handling structure
- **mesh**: modularize and restructure mesh and bank line processing logic (#147)
- **mesh**: modularize and restructure mesh and bank line processing logic
- **mesh**: restructure mesh module and improve submodule organization (#146)
- **mesh**: restructure mesh module and improve submodule organization
- **mesh**: modularize and restructure mesh (intersect line) and slicing logic (#143)
- **mesh**: modularize and restructure mesh traversal and slicing logic
- extract ship parameter functionality into ShipsParameters dataclass
- **plotting**: modularize and abstract plotting system for bank erosion and bank lines (#102)
- **plotting**: modularize and abstract plotting system for bank erosion and bank lines
- **erosion**: restructure erosion modules and improve documentation (#115)
- **erosion**: restructure erosion modules and improve documentation
- reorganize project structure into sub-modules and improve file organization (#112)
- reorganize project structure into sub-modules and improve file organization
- **bank_erosion**: restructure discharge level processing and debugging utilities (#106)
- **bank_erosion**: restructure discharge level processing and debugging utilities
- **bank_erosion**: restructure module and improve clarity of core classes and methods `_prepare_initial_conditions` and `_process_discharge_levels`
- **bank_erosion**: restructure module and improve clarity of core classes and methods `_prepare_initial_conditions` and `_process_discharge_levels`
- **bank_erosion**: modularize and improve river and bank erosion data handling (#93)
- **bank_erosion**: modularize and improve river and bank erosion data handling
- **geometry**: `RiverData`, `GeometryLine` and `SearchLine` classes (#95)
- **geometry**: rename CenterLine to GeometryLine and improve test coverage
- **core**: restructure river analysis with new CenterLine and SearchLine classes (#92)
- **core**: restructure river analysis with new CenterLine and SearchLine classes (#92)
- clean up utilities by removing unused functions and protecting locals (#90)
- clean up utilities by removing unused functions and protecting locals (#90)
- **simulation**: restructure and rename SimulationData class and methods (#78)
- **structures**: replace scattered variables with BankData, FairwayData & ErosionResults dataclasses (#77) (dfast-276)
- Upgrade shapely to v2.1.0 (#73) (dfast-280)
- Upgrade shapely to v2.1.0 (#73) 
- Use a dataclass for MeshData
- Use dataclass for erosioninputs and waterleveldata
- Combine variables into a WaterLevelData class
- Refactor CLI  (#48) (dfast-270)
- move io functionalities to separate repo `dfastio`  (#43) (dfast-261)
- Refactor bank erosion (#36) (dfast-265)
- Bank Line module Refactoring (#35)
- refactor config file functionalities (#23) (dfast-248 )
- split the batch module into bank_lines and bank_erosion modules (#25) (dfast-249 )

## v2.0.0 (2020-12-15)
