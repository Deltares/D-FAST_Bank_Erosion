@echo off

cd %~dp0

START /B /WAIT poetry run nuitka --standalone --python-flag=no_site --show-progress --plugin-enable=pyqt5 --plugin-enable=tk-inter --file-reference-choice=runtime dfastbe

call PostBuild.bat

rem end of build