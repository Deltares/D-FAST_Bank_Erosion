rem File "d:\checkouts\D-FAST\D-FAST_Bank_Erosion\dfastbe.dist\pyproj\datadir.py", line 109, in get_data_dir
rem  pyproj.exceptions.DataDirError: Valid PROJ data directory not found. Either set the path using the environmental variable PROJ_LIB or with `pyproj.datadir.set_data_dir`.
cd ..
mkdir dfastbe.dist\proj
copy .venv\Lib\site-packages\pyproj\proj_dir\share\proj\* dfastbe.dist\proj
cd %~dp0