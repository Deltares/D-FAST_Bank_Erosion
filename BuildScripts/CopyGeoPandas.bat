rem Load geopandas data sets
cd ..
mkdir dfastbe.dist\geopandas\datasets
copy .venv\Lib\site-packages\geopandas\datasets\natural* dfastbe.dist\geopandas\datasets
cd %~dp0
