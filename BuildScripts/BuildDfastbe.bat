@echo off

# redirect output and error logs to files when --no-console is specified
if "%1" == "--no-console" (
    set cmd_box_args=--windows-force-stderr-spec=%PROGRAM%logs.txt ^
     --windows-force-stdout-spec=%PROGRAM%output.txt ^
     --windows-disable-console ^
     src/dfastbe
) else (
    set cmd_box_args=src/dfastbe
)

# get version number
for /f "tokens=*" %%i in ('poetry version -s') do set VERSION=%%i
echo %VERSION%

cd %~dp0
cd..
START /B /WAIT python -m nuitka ^
 --standalone ^
 --mingw64 ^
 --assume-yes-for-downloads ^
 --python-flag=no_site ^
 --python-flag=no_asserts ^
 --python-flag=no_docstrings ^
 --nofollow-import-to=*.tests ^
 --nofollow-import-to=*unittest* ^
 --report=compilation-report.xml ^
 --show-progress ^
 --enable-plugin=pyqt5 ^
 --file-reference-choice=runtime ^
 --include-package=numpy ^
 --include-package=pyproj ^
 --include-module=shapely ^
 --include-package=matplotlib ^
 --include-package=netCDF4 ^
 --include-package=cftime ^
 --include-module=geopandas ^
 --include-package-data=geopandas.datasets ^
 --include-module=fiona ^
 --company-name=Deltares ^
 --file-version=%VERSION% ^
 --product-version=2025.01 ^
 --product-name="D-FAST Bank Erosion" ^
 --file-description="A Python tool to perform a bank erosion analysis based on a number of D-Flow FM simulations." ^
 --trademarks="All indications and logos of, and references to, \"D-FAST\", \"D-FAST Bank Erosion\" and \"D-FAST BE\" are registered trademarks of Stichting Deltares, and remain the property of Stichting Deltares. All rights reserved." ^
 --copyright="Copyright (C) 2025 Stichting Deltares." ^
 --windows-icon-from-ico=src/dfastbe/gui/icons/D-FASTBE.png ^
 --include-data-files=src/dfastbe/io/log_data/messages.NL.ini=dfastbe/io/log_data/messages.NL.ini ^
 --include-data-files=src/dfastbe/io/log_data/messages.UK.ini=dfastbe/io/log_data/messages.UK.ini ^
 --include-data-files=src/dfastbe/gui/icons/D-FASTBE.png=dfastbe/gui/icons/D-FASTBE.png ^
 --include-data-files=src/dfastbe/gui/icons/open.png=dfastbe/gui/icons/open.png ^
 --include-data-files=src/dfastbe/gui/icons/add.png=dfastbe/gui/icons/add.png ^
 --include-data-files=src/dfastbe/gui/icons/edit.png=dfastbe/gui/icons/edit.png ^
 --include-data-files=src/dfastbe/gui/icons/remove.png=dfastbe/gui/icons/remove.png ^
 --include-data-files=LICENSE.md=LICENSE.md ^
 --include-data-files=docs/dfastbe_usermanual.pdf=dfastbe/dfastbe_usermanual.pdf ^
 --include-data-files=docs/dfastbe_techref.pdf=dfastbe/dfastbe_techref.pdf ^
 --include-data-files=docs/dfastbe_release_notes.pdf=dfastbe/dfastbe_release_notes.pdf ^
 %cmd_box_args%

rem move some libraries to resolve dependencies ...
call BuildScripts\Move_Libs.bat

rem include example files into the distribution
call BuildScripts\Collect_Examples.bat

rem end of build