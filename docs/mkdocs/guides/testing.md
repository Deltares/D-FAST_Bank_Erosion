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