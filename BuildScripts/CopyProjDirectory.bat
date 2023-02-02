rem Load Proj files, referenced by Nuitka during its build and used by the tool during usage.
cd ..
mkdir dfastbe.dist\proj
copy .venv\Lib\site-packages\pyproj\proj_dir\share\proj\* dfastbe.dist\proj
cd %~dp0