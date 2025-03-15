@echo off

cd %~dp0
cd..

START /B /WAIT poetry run nuitka --standalone --assume-yes-for-downloads --python-flag=no_site --show-progress --plugin-enable=pyqt5 --plugin-enable=tk-inter --file-reference-choice=runtime dfastbe

cd %~dp0
call PostBuild.bat

rem end of build
