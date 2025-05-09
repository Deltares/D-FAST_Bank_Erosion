# D-FAST Bank Erosion

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=Deltares_D-FAST_Bank_Erosion&metric=alert_status)](https://sonarcloud.io/dashboard?id=Deltares_D-FAST_Bank_Erosion)

[![ci](https://github.com/Deltares/D-FAST_Bank_Erosion/actions/workflows/ci.yml/badge.svg)](https://github.com/Deltares/D-FAST_Bank_Erosion/actions/workflows/ci.yml)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=Deltares_D-FAST_Bank_Erosion&metric=alert_status)](https://sonarcloud.io/dashboard?id=Deltares_D-FAST_Bank_Erosion)
[![codecov](https://img.shields.io/codecov/c/github/deltares/D-FAST_Bank_Erosion.svg?style=flat-square)](https://app.codecov.io/gh/deltares/D-FAST_Bank_Erosion?displayType=list)


This is one of the [Deltares](https://www.deltares.nl) Fluvial Assessment Tools to be used in conjunction with D-Flow FM.
The purpose of this particular tool is
* to compute local bank erosion sensitivity, and
* to give an estimate of the amount of bank material that will be eroded
    * during the first year, and
    * until equilibrium.

The user should carry out a number of steady state hydrodynamic simulations for different discharges using [D-Flow FM](https://www.deltares.nl/en/software/module/d-flow-flexible-mesh/).
The results of these simulations will be combined with some basic morphological characteristics to estimate the bank erosion.
For more details see the documentation section.

## Documentation

The documentation consists of
* a [Technical Reference Manual](docs/end-user-docs/techref.md) in Markdown, and
* a LaTeX Technical Reference Manual.
* a LaTeX User Manual including scientific description.
The sources of all documents can be found in the `docs` folder.

## Installation
- For full instruction on how to install the package, please follow the instruction in documentation here [installation]
  (https://deltares.github.io/D-FAST_Bank_Erosion/latest/guides/poetry.html)
- For full developer documentations visit [documentation](https://deltares.github.io/D-FAST_Bank_Erosion/latest/index.html)

## License

This software is distributed under the terms of the GNU Lesser General Public License Version 2.1.
See the [license file](LICENSE.md) for details.