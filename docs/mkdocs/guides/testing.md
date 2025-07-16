# Naming conventions
## Test Files
- all test files should start with `test_`, and use snake case naming convention (i.e. `test_my_functionality.py`).

# Test Functions
- all test functions should start with `test_`, and use snake case naming convention.
```python
def test_my_functionality():
    # Test code here
    pass
```
## Test Classes
- all test classes should start with `Test`, and use CamelCase naming convention.
```python
class TestMyFunctionality:
    def test_my_functionality(self):
        # Test code here
        pass
```

# Docstring Convention for tests

All pytest test functions and classes in D-FAST Bank Erosion should include a clear and structured docstring describing the purpose and behavior of the test. The recommended format includes the following sections:

## Example Docstring Structure

```python
def test_my_functionality():
    """
    Brief description of what the test covers.

    Args:
        param1 (Type): Description of the parameter.
        param2 (Type): Description of the parameter.

    Mocks:
        - Description of any objects, functions, or modules that are mocked in the test.
        - Example: Mocked database connection to avoid real I/O.

    Asserts:
        - Description of what is being asserted or validated in the test.
        - Example: Asserts that the output matches the expected result.
    """
    # Test code here
    pass
```

## Guidelines

- **Description:** Start with a concise summary of what the test is verifying.
- **Args:** List and describe any parameters used in the test (optional for simple tests).
- **Mocks:** Clearly state what is mocked, why, and how it affects the test.
- **Asserts:** Explicitly describe what the test is asserting, including expected outcomes or behaviors.

This convention helps maintain clarity, consistency, and traceability in the test suite, making it easier for contributors to understand the intent and coverage of each test.

# Binaries

## Creating tests for the binaries
- all tests that are using the binaries are located in `tests/test_binaries`.
- The binaries are being built in teamcity in the following directory
```python
from pathlib import Path
repo_root = Path(__file__).resolve().parent.parent.parent
exe_path = repo_root / "dfastbe.dist/dfastbe.exe"
```
- Use the above path to trigger the binaries in any test.
- any test in the `tests/test_binaries` directory have to be marked with the following pytest marker.
```python
import pytest
@pytest.mark.binaries
def test_my_functionality():
    # Test code here
    pass
```

# Teamcity Pipelines
The team city pipelines have hard coded steps that locate the dfast built binaries, and also hard coded steps to
trigger the tests. This is a work in progress.
- The build step that triggers testing the binaries uses the following command
```shell
pytest -v tests/test_binaries/ --no-cov
```
- While the unit tests pipeline uses the `binaries` pytest marker to not run the tests that uses the binaries.
- So not marking any binary test with the `binaries` marker will break the unit tests pipeline.
```shell
pytest --junitxml="report.xml" --cov=%COVERAGE_LOC% --cov-report=xml tests/ -m "not binaries"
```
