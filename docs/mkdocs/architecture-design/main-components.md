# D-FAST Bank Erosion Architecture and Design

## repository structure
```
D-FAST_Bank_Erosion/
├───.github
│   ├───ISSUE_TEMPLATE
│   └───workflows
├───docs
│   ├───end-user-docs
│   │   ├───chapters
│   │   ├───cover
│   │   └───figures
│   └───mkdocs
│       ├───api
│       ├───gui
│       └───guides
├───examples
│   └───data
├───src
│	└───dfastbe
│	    │   add.png
│	    │   bank_erosion.py
│	    │   bank_lines.py
│	    │   cmd.py
│	    │   D-FASTBE.png
│	    │   edit.png
│	    │   gui.py
│	    │   io.py
│	    │   io.py~
│	    │   kernel.py
│	    │   messages.NL.ini
│	    │   messages.UK.ini
│	    │   open.png
│	    │   plotting.py
│	    │   remove.png
│	    │   src.pyproj
│	    │   support.py
│	    │   utils.py
│	    │   __init__.py
│	    └───__main__.py
├───tests
├───tests-dist
├── README.md
├── pyproject.toml
└── license.md
```

## Workflow and how modules interact: 

```mermaid
flowchart TD
    subgraph CLI
        main["__main__.py"]
    end

    subgraph GUI
        gui["gui.py"]
    end

    subgraph Core
        cmd["cmd.py"]
        bank_erosion["bank_erosion.py"]
        bank_lines["bank_lines.py"]
        kernel["kernel.py"]
    end

    subgraph Data
        io["io.py"]
        support["support.py"]
        utils["utils.py"]
    end

    subgraph Plotting
        plotting["plotting.py"]
    end

    main --> cmd
    cmd -->|parses config| io
    cmd -->|calculate erosion| bank_erosion
    cmd -->|starts analysis/Detect banks| bank_lines
    gui --> cmd
    gui --> bank_erosion
    gui --> io
    bank_erosion --> kernel
    bank_erosion --> io
    bank_erosion --> support
    bank_erosion --> plotting
    bank_erosion --> utils

    kernel --> support
    kernel --> utils
    
    bank_lines --> support
    bank_lines --> utils
    bank_lines --> io
    bank_lines --> plotting
    bank_lines --> kernel
    bank_lines --> utils
    
    plotting --> io
    plotting --> support
    plotting --> utils

```


```mermaid
graph TD
    subgraph Presentation_Layer
        GUI[gui.py]
        CLI[cmd.py & __main__.py]
    end

    subgraph Application_Layer
        AppLogic[bank_erosion.py]
    end

    subgraph Domain_Logic_Layer
        Kernel[kernel.py]
        BankDetection[bank_lines.py]
    end

    subgraph Data_Access_Layer
        IO[io.py]
        Support[support.py]
        Utils[utils.py]
    end

    subgraph Config_Layer
        ConfigManager["ConfigFile / RiverData"]
    end

    CLI --> AppLogic
    GUI --> AppLogic

    AppLogic --> Kernel
    AppLogic --> BankDetection
    AppLogic --> IO
    AppLogic --> Utils
    AppLogic --> Support
    AppLogic --> ConfigManager

    Kernel --> Support
    BankDetection --> Support
    BankDetection --> Utils
    IO --> ConfigManager
```