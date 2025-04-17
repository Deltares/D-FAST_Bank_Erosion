## Description of Files in the TeamCity pipelines

The `DFastBETests` folder contains Kotlin DSL configuration files for defining and managing TeamCity build configurations, templates, and dependencies for the D-FAST Bank Erosion project. Below is a description of each file:

### 1. `settings.kts`
**Purpose:**  
The main entry point for the TeamCity Kotlin DSL configuration. Defines the project structure, parameters, and references to build configurations and templates.

**Key Features:**  
- Sets the project description and version.  
- Includes global parameters like `CONDA_ENV_NAME` and `python.version`.  
- References build configurations such as `UnitTestsSonarCloud`, `BuildWithCommandWindow`, and others.

---

### 2. `unittestssonarcloud.kt`
**Purpose:**  
Defines the build configuration for running unit tests and performing SonarCloud analysis.

**Key Features:**  
- Runs the unit tests
- Integrates with SonarCloud for code quality analysis.  
- Publishes commit statuses to GitHub using a personal access token.  

---

### 3. `dfastcleanconfiguration.kt`
**Purpose:**  
Provides a reusable template for cleaning up the build environment.

**Key Features:**  
- Defines the steps for setting up the conda environment and clean up of the environment. 
- Can be used as a base template for other build configurations to ensure a clean workspace.

---

### 4. `buildwithcommandwindow.kt`
**Purpose:**  
Defines the build configuration for compiling the D-FAST Bank Erosion project providing a command window version as final result.

**Key Features:**  
- Executes a batch script (`BuildDfastbe.bat`) to build the project.  
- Relies on the template (`dfastcleanconfiguration.kt`) for conda environment setup.  
- Includes failure conditions:  
    - Fails the build if an `AssertionError` is detected in the logs.  
    - Fails the build if the total artifact size is below a specified threshold (e.g., 100MB).  
- Defines dependencies:  
    - Depends on `LatexManualGeneration` for PDF artifacts.  
    - Depends on `UnitTestsSonarCloud` for unit tests and code coverage.  

---

### 5. `buildwithoutcommandwindow.kt`
**Purpose:**  
Defines the build configuration for compiling the D-FAST Bank Erosion project where the final result suppresses the command window.

**Key Features:**  
- Executes a batch script (`BuildDfastbe_no_command_window.bat`) to build the project.  
- Relies on the template (`dfastcleanconfiguration.kt`) for conda environment setup.   
- Includes failure conditions:  
    - Fails the build if an `AssertionError` is detected in the logs.  
    - Fails the build if the total artifact size is below a specified threshold (e.g., 100MB).  
- Defines dependencies:  
    - Depends on `LatexManualGeneration` for PDF artifacts.  
    - Depends on `UnitTestsSonarCloud` for unit tests and code coverage. 

---

### 6. `latexmanualgeneration.kt`
**Purpose:**  
Defines a build configuration for generating LaTeX-based documentation for the project.

**Key Features:**  
- Runs a script to compile LaTeX files into PDF documentation.
- Produces PDF artifacts that are used as dependencies to prevent other build configurations from running if the documentation generation fails (e.g., `BuildWithCommandWindow`).
- Ensures that the documentation is up-to-date with the latest changes in the project.

---

### 7. `signedrelease.kt`
**Purpose:**  
Defines a build configuration for creating signed releases of the D-FAST Bank Erosion project. This configuration combines artifacts from multiple sources, including the command window version and the suppressed command window version, and ensures that the release is properly signed and ready for distribution.

**Key Features:**  
- Produces a signed release artifact.
- Moves the CLI version of dfastbe.exe to the root directory and cleans up unnecessary folders.

---

### 8. `signedreleasecommand.kt`
**Purpose:**
Defines a build configuration for executing the signing process of release artifacts using a command-based approach. This configuration ensures that the release artifacts are signed with the appropriate certificates and are ready for secure distribution.

**Key Features:**
- Produces a signed release zip archive artifact.
- Collects the artifacts from the signing configuration.

---

### 9. `distributiontests.kt`
**Purpose:**  
Defines a build configuration for running distribution tests on the built binaries.

**Key Features:**  
- Ensures that the distributed artifacts are functional and meet quality standards.
- Runs automated tests on the final release binaries.

---

### 10. `pom.xml`
**Purpose:**  
Defines the Maven project configuration for the D-FAST Bank Erosion project. This file is used to manage project dependencies, build lifecycle, and plugins.

**Key Features:**  
- Specifies project metadata such as group ID, artifact ID, and version.  
- Manages dependencies required for building and testing the project.  
- Configures Maven plugins for tasks like compiling, packaging, and testing.  
- Supports integration with CI/CD pipelines by automating build and deployment processes.  
- Ensures compatibility with Java-based tools and frameworks.

---

### Folder Overview
The `DFastBETests` folder is structured to modularize the TeamCity build configurations and templates for the D-FAST Bank Erosion project. Each file serves a specific purpose, such as running tests, building the project, or managing dependencies, while the `settings.kts` file ties everything together into a cohesive project.
