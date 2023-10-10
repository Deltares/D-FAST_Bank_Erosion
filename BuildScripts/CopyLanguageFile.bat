rem Load language file 'messages.UK.ini'
rem Resolves the following error: Unable to load language file 'messages.UK.ini'
cd ..
mkdir dfastbe.dist\dfastbe
copy dfastbe\messages.*.ini dfastbe.dist\dfastbe
copy dfastbe\*.png dfastbe.dist\dfastbe
cd %~dp0