# Technical Description

This is a draft version of the technical documentation for D-FAST Bank Erosion.
It's constantly updated as we convert new parts of the code.

## Design Considerations

1. This new program replaces WAQBANK which was developed for the SIMONA system.
1. The program must read D-Flow FM map-files instead of SIMONA SDS-files.
1. The program must allow users to it in batch mode from the command line.
1. The program must use the same input files as WAQBANK (except for keywords that obviously need upgrading such as references to old SIMONA files).
1. The program must give the same numerical results for the same input (although in new format) as WAQBANK (within the numerical accuracy of the algorithm given the switch in programming languages).
1. The program doesn't have to generate the same figures as the WAQBANK program, but it should allow for easy creation of equivalent figures.
1. The output files must comply with international standard such that results can be easily visualized (e.g. using QUICKPLOT or QGIS).
1. It would be nice to include a simple graphical user interface to support users with the input specification.

## Software Design

D-FAST Bank Erosion consists of three separate programs; all written in Python.

* **dfastbe1**: The preprocessor to identify bank lines from the mesh
* **dfastbe2**: The actual data analysis tool
* **dfastbegui**: The graphical user interface to create and edit the input files for the other two programs.

These programs are discussed individually below.

### dfastbe1

### dfastbe2

### dfastbegui

## Coding Conventions

This program has been programmed following PEP 8 style guide.

## Testing

1. The first step in the validation testing of the new code is to convert some SIMONA output files into NetCDF files that have the same effective data structure as the D-Flow FM output files.
It is verified that the new version of WAQBANK produces the same analysis results using those new files as the existing WAQBANK version gives for the original files.
1. In the second step we rerun SIMONA simulations using D-Flow FM, i.e.~on the same curvilinear SIMONA mesh.
This will result in differences due to the numerical scheme of D-Flow FM compared to SIMONA, but not due to mesh differences.
1. In the third step we rerun the case using an optimal D-Flow FM unstructured mesh.
This will result in differences due to both numerical scheme and mesh, but this is what will be done in the end ... so this is representative for the results that we will get using the new tool in the new setting.
This change should not cause major differences if the quantitative results of the tool are of any use.

## Command Line Arguments

| short | long | description |
|-------|:-----|:------------|
| -i    | --input_file | name of configuration file |
| -v    | --verbosity  | set verbosity level of run-time diagnostics: DEBUG, {INFO}, WARNING, ERROR or CRITICAL) |
