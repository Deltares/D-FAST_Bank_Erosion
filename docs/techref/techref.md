# Technical Description

This is the technical reference manual.

## Design Considerations

1. This new program is to replace WAQBank which was developed for the SIMONA system.
1. This program should give the same results for the same input (although in new format) as WAQBank.
1. The new program reads D-Flow FM map-files instead of SIMONA SDS_files.
1. Users must be able to run this program in batch mode from the command line.
1. A simple graphical user interface will be added to support users with the input specification.

## Code Design

## Coding Conventions

This program has been programmed following PEP 8 style guide.

## Testing

## Command Line Arguments

| short | long | description |
|-------|:-----|:------------|
| -i    | --input_file | name of configuration file |
| -v    | --verbosity  | set verbosity level of run-time diagnostics: DEBUG, {INFO}, WARNING, ERROR or CRITICAL) |
