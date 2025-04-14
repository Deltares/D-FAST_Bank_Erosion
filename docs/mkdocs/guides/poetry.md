# Installation using Poetry

You can use a Poetry-based installation if you are using
D-FAST_Bank_Erosion from a local clone of the Github repository,
for example if you intend to contribute to the code.

## Clone the GitHub repo
Use your own preferred way of cloning the GitHub repository of D-FAST_Bank_Erosion.
In the examples below it is placed in `C:\checkouts\D-FAST_Bank_Erosion_git`.

## Use Poetry to install D-FAST_Bank_Erosion
We use `poetry` to manage our package and its dependencies.

!!! note
    If you use `conda`, do not combine conda virtual environments with the poetry virtual environment.
    In other words, run the `poetry install` command from the `base` conda environment.

1. Download + installation instructions for Poetry are [here](https://python-poetry.org/).
2. After installation of Poetry itself, now use it to install your local clone of the D-FAST_Bank_Erosion package, as follows.
   Make sure Poetry is available on your `PATH` and run `poetry install` in the D-FAST_Bank_Erosion directory in your shell of choice.
   This will create a virtual environment in which D-FAST_Bank_Erosion is installed and made available for use in your own scripts.
   For example in an Anaconda PowerShell:
```
(base) PS C:\checkouts\D-FAST_Bank_Erosion_git> poetry install
Creating virtualenv D-FAST_Bank_Erosion-kHkQBdtS-py3.8 in C:\Users\dam_ar\AppData\Local\pypoetry\Cache\virtualenvs
Installing dependencies from lock file

Package operations: 67 installs, 0 updates, 0 removals

  * Installing six (1.16.0)
[..]
Installing the current project: D-FAST_Bank_Erosion (0.1.5)
(base) PS C:\checkouts\D-FAST_Bank_Erosion_git>
```  
   If you need to use an already existing Python installation, you can activate it and run `poetry env use system` before `poetry install`.

3. Test your installation, by running the D-FAST_Bank_Erosion pytest suite via poetry:
```
(base) PS C:\checkouts\D-FAST_Bank_Erosion_git> poetry run pytest
===================================== test session starts ======================================
platform win32 -- Python 3.8.8, pytest-6.2.5, py-1.10.0, pluggy-1.0.0
rootdir: C:\checkouts\D-FAST_Bank_Erosion_git, configfile: pyproject.toml
plugins: cov-2.12.1
collected 473 items / 2 deselected / 471 selected

tests\io\dflowfm\ini\test_ini.py ........................................................ [  3%]
tests\io\dflowfm\test_bc.py ....                                                          [  4%]
tests\io\dflowfm\test_ext.py ........................................................     [  5%]
tests\io\dflowfm\test_fnm.py ..................                                           [ 11%]
tests\io\dflowfm\test_net.py ............                                                 [ 11%]
tests\io\dflowfm\test_parser.py .                                                         [ 12%]
tests\io\dflowfm\test_polyfile.py ........................................................[ 23%]
....................................                                                      [ 27%]
tests\io\dflowfm\test_structure.py .......................................................[ 42%]
.........................................................                                 [ 54%]
tests\io\dimr\test_dimr.py ...                                                            [ 56%]
tests\io\rr\meteo\test_bui.py ...........................                                 [ 57%]
tests\io\test_docker.py .                                                                 [ 70%]
tests\test_model.py ...............                                                       [ 78%]
tests\test_utils.py .......                                                               [ 91%]
.........................................                                                 [100%]

============================== 471 passed, 2 deselected in 3.50s ===============================
(base) PS C:\checkouts\D-FAST_Bank_Erosion_git>
```  
4. Start using D-FAST_Bank_Erosion. You can launch your favourite editor (for example VS Code)
by first starting a poetry shell with the virtual D-FAST_Bank_Erosion environment:
```
(base) PS C:\checkouts\D-FAST_Bank_Erosion_git> poetry shell
(base) PS C:\checkouts\D-FAST_Bank_Erosion_git> code
```

## Switching Between Python Versions
If you need to switch between Python versions (e.g., from Python 3.9 to Python 3.10), you can configure Poetry to use a specific Python version. Follow these steps:

1. **Verify python version in `pyproject.toml`**:
The `pyproject.toml` specifies which python version is supported. Please verify that the python version you wish to use is supported. For example, the following configuration:
```toml
[tool.poetry.dependencies]
python = "~3.10"
```
It Indicates that the project supports Python 3.10 with any minor version (e.g., 3.10.1, 3.10.2, etc.).

2. **Check Installed Python Versions**:
Ensure that the desired Python version (e.g., Python 3.10) is installed on your system. 
You can check the available versions by running:
on windows:
```shell
python --version
``` 
or on Linux:
```shell
python3 --version 
```

3. **Set the Python Version for Poetry**:
Use the `poetry env use` command to specify the Python version for your project. For example: `poetry env use python3.10`. This will create a new virtual environment using Python 3.10.

4. **Verify the Python Version**:
After setting the Python version, verify that Poetry is using the correct version: `poetry run python --version`. The output should show Python 3.10.

5. **Reinstall Dependencies**:
If you switch Python versions, you may need to reinstall your dependencies. Run: `poetry install`.

6. **Test the Installation**: Run the test suite to ensure everything works correctly with the new Python version: 
`poetry run pytest`.

