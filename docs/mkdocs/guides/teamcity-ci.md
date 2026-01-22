## Description of Files in the TeamCity pipelines

The `.teamcity/DFastBETests` folder contains Kotlin DSL configuration files for defining and managing TeamCity build configurations, templates, and dependencies for the D-FAST Bank Erosion project. Below is a description of each file as currently implemented:

### 1. `settings.kts`
**Purpose:**  
The main entry point for the TeamCity Kotlin DSL configuration. Defines the project structure, parameters, and references to build configurations and templates.

**Key Features:**  
- Sets the project description and version.  
- Includes global parameters like `python.version`, and `poetry.path`.  
- References build configurations such as `UnitTests`, `BuildMain`, `BuildTerminal`, `LatexManual`, `SignedRelease`, `SignedReleaseTerminal`, and `TestBinaries`.

---

### 2. `poetry_template.kt`
**Purpose:**  
Provides a reusable template for all builds using Poetry for dependency management and environment setup.

**Key Features:**  
- Installs Poetry standalone in a temporary directory.  
- Creates and manages a Poetry environment using the specified Python version.  
- Installs dependencies via Poetry.  
- Cleans up the Poetry environment after the build.  
- Ensures builds run only on Windows agents with the required Python path.

---

### 3. `unit_tests.kt`
**Purpose:**  
Defines the build configuration for running unit tests and (optionally) SonarCloud analysis.

**Key Features:**  
- Uses the Poetry template for environment setup.  
- Runs unit tests and generates coverage reports.  
- (Optionally) runs SonarCloud analysis (step is present but disabled by default).  
- Publishes commit statuses to GitHub.

---

### 4. `build_main.kt`
**Purpose:**  
Defines the build configuration for compiling the D-FAST Bank Erosion project as the main distribution (without a command window).

**Key Features:**  
- Uses the Poetry template for environment setup.  
- Executes the `BuildDfastbe_no_command_window.bat` script via Poetry.  
- Produces zipped distribution artifacts.  
- Fails the build on `AssertionError` in logs.

---

### 5. `build_terminal.kt`
**Purpose:**  
Defines the build configuration for compiling the D-FAST Bank Erosion project with a command window (for debugging).

**Key Features:**  
- Uses the Poetry template for environment setup.  
- Executes the `BuildDfastbe.bat` script via Poetry.  
- Produces zipped distribution artifacts.  
- Fails the build on `AssertionError` in logs.  
- Has a build timeout and artifact size checks.

---

### 6. `latex_manual.kt`
**Purpose:**  
Defines a build configuration for generating LaTeX-based documentation for the project.

**Key Features:**  
- Runs scripts to generate user manual, technical reference, and release notes using LaTeX and BibTeX.  
- Produces PDF and log artifacts.  
- Fails the build if documentation generation fails.  
- Publishes commit statuses to GitHub.

---

### 7. `signed_release.kt`
**Purpose:**  
Defines a build configuration for collecting and packaging the signed release of the D-FAST Bank Erosion project.

**Key Features:**  
- Collects artifacts from the main and terminal builds, as well as from signing dependencies.  
- Produces a signed release zip artifact.  
- Runs only on Windows agents.

---

### 8. `signed_release_terminal.kt`
**Purpose:**  
Defines a build configuration for collecting the signed release with a command window for debugging.

**Key Features:**  
- Collects artifacts from the terminal build and signing dependencies.  
- Produces a signed release zip artifact.  
- Runs only on Windows agents.

---

### 9. `test_binaries.kt`
**Purpose:**  
Defines a build configuration for running distribution tests on the built binaries.

**Key Features:**  
- Uses the Poetry template for environment setup.  
- Runs tests on the distributed binaries.  
- Publishes commit statuses to GitHub.  
- Triggers on VCS changes in production environment.  
- Depends on the signed release with command window.

---

### 10. `pom.xml`
**Purpose:**  
Defines the Maven project configuration for the D-FAST Bank Erosion project (legacy or for Java-based tools).

**Key Features:**  
- Specifies project metadata and dependencies.  
- May be used for integration with Java-based tools or legacy build steps.

---

### Additional Notes
- All builds are now based on Poetry for Python dependency and environment management.
- The build pipeline is modular, with clear separation between testing, building, documentation, signing, and distribution testing.
- Commit status publishing and clean workspace features are enabled for quality and traceability.
- The configuration is designed for Windows build agents and expects a valid Python path and Poetry installation.
