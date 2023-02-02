rem Load Proj files, referenced by Nuitka during its build and used by the tool during usage.
rem Resolves the following error: pyproj.exceptions.DataDirError: Valid PROJ data directory not found. Either set the path using the environmental variable PROJ_LIB or with `pyproj.datadir.set_data_dir`.
cd ..
mkdir dfastbe.dist\proj
copy .venv\Lib\site-packages\pyproj\proj_dir\share\proj\* dfastbe.dist\proj
cd %~dp0