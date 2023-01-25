@echo off

cd %~dp0

call CopyLanguageFile.bat
call CopyDutchRiversFile.bat
call CopyUserManualFile.bat
call CopyGeoPandas.bat

rem end of post build