nuitka --standalone --python-flag=no_site --show-progress --plugin-enable=numpy --plugin-enable=qt-plugins --plugin-enable=tk-inter --file-reference-choice=runtime dfastbe > dfastbe.build.log 2>&1
rem pause

rem The code execution connot proceed because python38.dll was not found. Reinstalling the program may fix this problem.
copy ..\envs\dfastbe\python38.dll dfastbe.dist


rem File "D:\checkouts\D-FAST\D-FAST_Bank_Erosion\dfastbe.dist\importlib\resources.py", line 97, in open_binary
rem FileNotFoundError: [Errno 2] No such file or directory: '...\\dfastbe.dist\\certifi\\cacert.pem'
mkdir dfastbe.dist\certifi
copy ..\envs\dfastbe\Lib\site-packages\certifi\cacert.pem dfastbe.dist\certifi


rem File "d:\checkouts\D-FAST\D-FAST_Bank_Erosion\dfastbe.dist\pyproj\datadir.py", line 109, in get_data_dir
rem  pyproj.exceptions.DataDirError: Valid PROJ data directory not found. Either set the path using the environmental variable PROJ_LIB or with `pyproj.datadir.set_data_dir`.
mkdir dfastbe.dist\proj
copy ..\envs\dfastbe\Library\share\proj\* dfastbe.dist\proj


rem File "d:\checkouts\D-FAST\D-FAST_Bank_Erosion\dfastbe.dist\shapely\geos.py", line 60, in load_dll
rem  OSError: Could not find lib geos_c.dll or load any of its variants [].
mkdir dfastbe.dist\shapely\DLLs
copy ..\envs\dfastbe\Lib\site-packages\shapely\DLLs\* dfastbe.dist\shapely\DLLs


rem   File "d:\checkouts\D-FAST\D-FAST_Bank_Erosion\dfastbe.dist\geopandas\datasets\__init__.py", line 6, in <module geopandas.datasets>
rem StopIteration
mkdir dfastbe.dist\geopandas\datasets
copy ..\envs\dfastbe\Lib\site-packages\geopandas\datasets\natural* dfastbe.dist\geopandas\datasets


rem File "d:\checkouts\D-FAST\D-FAST_Bank_Erosion\dfastbe.dist\PyQt5\__init__.py", line 33, in find_qt
rem  ImportError: unable to find Qt5Core.dll on PATH
rem mkdir dfastbe.dist\PyQt5\Qt\bin
rem copy ..\envs\dfastbe\Library\bin\Qt5Core* dfastbe.dist\PyQt5\Qt\bin


rem File "d:\checkouts\D-FAST\D-FAST_Bank_Erosion\dfastbe.dist\fiona\collection.py", line 9, in <module fiona.collection>
rem  ImportError: LoadLibraryExW 'd:\checkouts\D-FAST\D-FAST_Bank_Erosion\dfastbe.dist\fiona\ogrext.pyd' failed: The specified module could not be found.
copy ..\envs\dfastbe\Library\bin\*.dll dfastbe.dist
rem don't need the boost libraries and maybe some other ...
del /y dfastbe.dist/boost*.dll


rem ImportError: LoadLibraryExW 'd:\checkouts\D-FAST\D-FAST_Bank_Erosion\dfastbe.dist\_ctypes.pyd' failed: The specified module could not be found.
copy ..\envs\dfastbe\DLLs\libffi-7.dll dfastbe.dist


rem File "d:\checkouts\D-FAST\D-FAST_Bank_Erosion\dfastbe.dist\matplotlib\__init__.py", line 772, in _get_data_path
rem RuntimeError: Could not find the matplotlib data files
del /s /y dfastbe.dist\mpl-data
mkdir dfastbe.dist\matplotlib\mpl-data
xcopy /s /y ..\envs\dfastbe\Lib\site-packages\matplotlib\mpl-data\* dfastbe.dist\matplotlib\mpl-data


rem File "d:\checkouts\D-FAST\D-FAST_Bank_Erosion\dfastbe.dist\rtree\core.py", line 126, in <module rtree.core>
rem  OSError: could not find or load spatialindex_c-64.dll
mkdir dfastbe.dist\Library\bin
copy ..\envs\dfastbe\Library\bin\spatialindex* dfastbe.dist\Library\bin


rem File "d:\checkouts\D-FAST\D-FAST_Bank_Erosion\dfastbe.dist\dfastbe\gui.py", line 32, in <module dfastbe.gui>
rem  ImportError: LoadLibraryExW 'd:\checkouts\D-FAST\D-FAST_Bank_Erosion\dfastbe.dist\PyQt5\QtWidgets.pyd' failed: The specified module could not be found.
rem copy ..\envs\dfastbe\python3.dll dfastbe.dist\PyQt5
rem copy dfastbe.dist\*.dll dfastbe.dist\PyQt5


rem Unable to load language file 'messages.UK.ini'
mkdir dfastbe.dist\dfastbe
copy dfastbe\messages.*.ini dfastbe.dist\dfastbe
copy dfastbe\*.png dfastbe.dist\dfastbe


rem qt.qpa.plugin: Could not find the Qt platform plugin "windows" in ""
rem  This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.
mkdir dfastbe.dist\platforms
xcopy /s /y ..\envs\dfastbe\Library\plugins\platforms\* dfastbe.dist\platforms


rem File "d:\checkouts\D-FAST\D-FAST_Bank_Erosion\dfastbe.dist\tkinter\__init__.py", line 2261, in __init__
rem  _tkinter.TclError: Can't find a usable init.tcl in the following directories:
rem   d:/checkouts/D-FAST/D-FAST_Bank_Erosion/lib/tcl8.6 ...
rem This probably means that Tcl wasn't installed properly.
del /s /y dfastbe.dist\tcl
mkdir dfastbe.dist\lib\tcl8.6
xcopy /s /y ..\envs\dfastbe\Library\lib\tcl8.6\* dfastbe.dist\lib\tcl8.6


rem File "d:\checkouts\D-FAST\D-FAST_Bank_Erosion\dfastbe.dist\tkinter\__init__.py", line 2261, in __init__
rem  _tkinter.TclError: Can't find a usable tk.tcl in the following directories:
rem  d:/checkouts/D-FAST/D-FAST_Bank_Erosion/lib/tcl8.6/tk8.6 d:/checkouts/D-FAST/D-FAST_Bank_Erosion/lib/tk8.6 ...
rem This probably means that tk wasn't installed properly.
del /s /y dfastbe.dist\tk
mkdir dfastbe.dist\lib\tk8.6
xcopy /s /y ..\envs\dfastbe\Library\lib\tk8.6\* dfastbe.dist\lib\tk8.6

rem '...\dfastbe_usermanual.pdf' is not recognized as an internal or external command, operable program or batch file.
copy docs\dfastbe_usermanual.pdf dfastbe.dist\dfastbe