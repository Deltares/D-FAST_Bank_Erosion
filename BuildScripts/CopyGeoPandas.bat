rem Load geopandas data sets
rem Resolves the following error: __init__.py", line 6, in <module geopandas.datasets>, StopIteration
cd ..
mkdir dfastbe.dist\geopandas\datasets
copy .venv\Lib\site-packages\geopandas\datasets\natural* dfastbe.dist\geopandas\datasets
cd %~dp0
