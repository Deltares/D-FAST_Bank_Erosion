rem Load file 'dfastbe_usermanual.pdf'
rem Resolves the following error: ...\dfastbe_usermanual.pdf' is not recognized as an internal or external command, operable program or batch file.
cd ..
copy docs\dfastbe_usermanual.pdf dfastbe.dist\dfastbe
cd %~dp0
