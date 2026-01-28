@echo off

rem Define common paths
set ICONS_SRC=src/dfastbe/gui/icons
set ICONS_DEST=dfastbe/gui/icons
set LOG_DATA_SRC=src/dfastbe/io/log_data
set LOG_DATA_DEST=dfastbe/io/log_data
set DOCS_SRC=docs
set DOCS_DEST=dfastbe

rem redirect output and error logs to files when --no-console is specified
if "%1" == "--no-console" (
    set cmd_box_args=--windows-force-stderr-spec=%PROGRAM%logs.txt --windows-force-stdout-spec=%PROGRAM%output.txt --windows-disable-console
) else (
    set cmd_box_args=
)

rem get version number
for /f "tokens=*" %%i in ('poetry version -s') do set VERSION=%%i

echo.
echo Version: %VERSION%
echo.

cd %~dp0
cd..

echo Starting Nuitka build...
echo.

START /B /WAIT python -m nuitka ^
 --standalone ^
 --mingw64 ^
 --assume-yes-for-downloads ^
 --python-flag=no_site ^
 --python-flag=no_asserts ^
 --python-flag=no_docstrings ^
 --nofollow-import-to=*.tests ^
 --enable-plugin=anti-bloat ^
 --report=compilation-report.xml ^
 --show-progress ^
 --enable-plugin=pyside6 ^
 --file-reference-choice=runtime ^
 --include-package=unittest ^
 --include-package=numpy ^
 --include-package=pyproj ^
 --include-module=shapely ^
 --include-package=matplotlib ^
 --include-package=netCDF4 ^
 --include-package=cftime ^
 --include-module=geopandas ^
 --include-package=pandas ^
 --include-package=pytz ^
 --include-distribution-metadata=pytz ^
 --include-package-data=geopandas.datasets ^
 --include-module=fiona ^
 --company-name=Deltares ^
 --file-version=%VERSION% ^
 --product-version=2025.01 ^
 --product-name="D-FAST Bank Erosion" ^
 --file-description="A Python tool to perform a bank erosion analysis based on a number of D-Flow FM simulations." ^
 --trademarks="All indications and logos of, and references to, \"D-FAST\", \"D-FAST Bank Erosion\" and \"D-FAST BE\" are registered trademarks of Stichting Deltares, and remain the property of Stichting Deltares. All rights reserved." ^
 --copyright="Copyright (C) 2025 Stichting Deltares." ^
 --windows-icon-from-ico=%ICONS_SRC%/D-FASTBE.png ^
 --include-data-files=%LOG_DATA_SRC%/messages.NL.ini=%LOG_DATA_DEST%/messages.NL.ini ^
 --include-data-files=%LOG_DATA_SRC%/messages.UK.ini=%LOG_DATA_DEST%/messages.UK.ini ^
 --include-data-files=%ICONS_SRC%/D-FASTBE.png=%ICONS_DEST%/D-FASTBE.png ^
 --include-data-files=%ICONS_SRC%/open.png=%ICONS_DEST%/open.png ^
 --include-data-files=%ICONS_SRC%/add.png=%ICONS_DEST%/add.png ^
 --include-data-files=%ICONS_SRC%/edit.png=%ICONS_DEST%/edit.png ^
 --include-data-files=%ICONS_SRC%/remove.png=%ICONS_DEST%/remove.png ^
 --include-data-files=LICENSE.md=LICENSE.md ^
 --include-data-files=%DOCS_SRC%/dfastbe_usermanual.pdf=%DOCS_DEST%/dfastbe_usermanual.pdf ^
 --include-data-files=%DOCS_SRC%/dfastbe_techref.pdf=%DOCS_DEST%/dfastbe_techref.pdf ^
 --include-data-files=%DOCS_SRC%/dfastbe_release_notes.pdf=%DOCS_DEST%/dfastbe_release_notes.pdf ^
 %cmd_box_args% ^
 src/dfastbe

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Nuitka build failed with error code %ERRORLEVEL%
    echo.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo Nuitka build completed successfully
echo.

rem move some libraries to resolve dependencies ...
echo Moving libraries to resolve dependencies...
call BuildScripts\Move_Libs.bat

rem include example files into the distribution
echo Collecting example files...
call BuildScripts\Collect_Examples.bat

echo.
echo Build completed successfully!
echo.

rem end of build
