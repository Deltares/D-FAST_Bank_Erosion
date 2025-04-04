@echo off

echo Moving libraries to different folder to resolve dependencies ...

echo pyproj.libs ...
move dfastbe.dist\pyproj.libs\* dfastbe.dist\pyproj
del /Q dfastbe.dist\pyproj.libs

echo matplotlib.libs ...
move dfastbe.dist\matplotlib.libs\* dfastbe.dist\matplotlib
del /Q dfastbe.dist\matplotlib.libs

echo fiona.libs ...
move dfastbe.dist\fiona.libs\* dfastbe.dist\fiona
del /Q dfastbe.dist\fiona.libs
