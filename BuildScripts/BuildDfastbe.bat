@echo off

if "%1" == "--no-console" (

set cmd_box_args=--windows-force-stderr-spec=%PROGRAM%logs.txt ^
 --windows-force-stdout-spec=%PROGRAM%output.txt ^
 --windows-disable-console ^
 dfastbe
 
) else (

set cmd_box_args=dfastbe

)

cd %~dp0
cd..

START /B /WAIT poetry run nuitka ^
 --standalone ^
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
 --include-package=pyproj ^
 --include-module=shapely ^
 --include-package=matplotlib ^
 --include-package=netCDF4 ^
 --include-module=geopandas ^
 --include-package-data=geopandas.datasets ^
 --include-module=fiona ^
 --company-name=Deltares ^
 --file-version=2.3.2 ^
 --product-version=2024.00 ^
 --product-name="D-FAST Bank Erosion" ^
 --file-description="A Python tool to perform a bank erosion analysis based on a number of D-Flow FM simulations." ^
 --trademarks="All indications and logos of, and references to, \"D-FAST\", \"D-FAST Bank Erosion\" and \"D-FAST BE\" are registered trademarks of Stichting Deltares, and remain the property of Stichting Deltares. All rights reserved." ^
 --copyright="Copyright (C) 2024 Stichting Deltares." ^
 --windows-icon-from-ico=dfastbe/D-FASTBE.png ^
 --include-data-files=dfastbe/messages.NL.ini=dfastbe/messages.NL.ini ^
 --include-data-files=dfastbe/messages.UK.ini=dfastbe/messages.UK.ini ^
 --include-data-files=dfastbe/D-FASTBE.png=dfastbe/D-FASTBE.png ^
 --include-data-files=dfastbe/add.png=dfastbe/add.png ^
 --include-data-files=dfastbe/edit.png=dfastbe/edit.png ^
 --include-data-files=dfastbe/open.png=dfastbe/open.png ^
 --include-data-files=dfastbe/remove.png=dfastbe/remove.png ^
 --include-data-files=LICENSE.md=LICENSE.md ^
 --include-data-files=docs/dfastbe_usermanual.pdf=dfastbe/dfastbe_usermanual.pdf ^
 --include-data-files=docs/dfastbe_techref.pdf=dfastbe/dfastbe_techref.pdf ^
 %cmd_box_args%

rem end of build