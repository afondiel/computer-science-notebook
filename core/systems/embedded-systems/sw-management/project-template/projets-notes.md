# Projects Notes

**Table of Contents (ToC)**
- [Overview](#overview)
- [Project Management](#project-management)
- [Project Struture - High Level/Customer Level (V Cycle and Agile Compliant)](#project-struture---high-levelcustomer-level-v-cycle-and-agile-compliant)
- [Project Structure - Product Level =\> generated using cookiecutter](#project-structure---product-level--generated-using-cookiecutter)
- [References](#references)

## Overview

Guidelines, notes and resources for project management.

## Project Management

- [Self-Driving Cars — Managing a Project @thinkautonomous.ai](https://www.thinkautonomous.ai/blog/self-driving-cars-managing-a-project/)
## Project Struture - High Level/Customer Level (V Cycle and Agile Compliant)

- Valeo

![](https://www.tpsgc-pwgsc.gc.ca/biens-property/sngp-npms/ti-it/images/nannnms-img1-eng.png)
```
- 00-Customer requirements (CDC)
- 01-Requirements Analysis 
- 02-Specifications
- 03-Architecture Design
- 04-Detailed Design
- 05-Implementation & Prototyping & Dev
- 06-Unit Testing
- 07-Integration & Test
- 08-Validation Test
- 09-Release (receips)
- 10-Customer Deliveries (shall match/satisfy custumer needs)
- 11-Planning
```
For more details, check out the document [Cycle.en-VBec](#)

- E2CAD

```
- 00-Customer requirements (CDC)
- 01-Specs
- 02-Design & architecture
- 03-Implementation & Prototyping
- 04-testing
- 05-Validation
- 06-Release
- 07-Planning
- 08-Resources*
```

- AGCO :uses multiple discs from svn turtoise (github) (L:/engineering: docs, S:/Dev...)

```
- Bootloader
- Build
- DEBUGG.txt
- Miscellaneous
- Tools
```

## Project Structure - AI Product Level => generated using [cookiecutter](https://github.com/cookiecutter/cookiecutter)

- AI/ML/DL/Data Science Project Template

```
├── LICENSE
├── Makefile           <- Makefile with commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default Sphinx project; see sphinx-doc.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.py           <- Make this project pip installable with `pip install -e`
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module
│   │
│   ├── data           <- Scripts to download or generate data
│   │   └── make_dataset.py
│   │
│   ├── features       <- Scripts to turn raw data into features for modeling
│   │   └── build_features.py
│   │
│   ├── models         <- Scripts to train models and then use trained models to make
│   │   │                 predictions
│   │   ├── predict_model.py
│   │   └── train_model.py
│   │
│   └── visualization  <- Scripts to create exploratory and results oriented visualizations
│       └── visualize.py
│
└── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io
```
src: [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/)

- Faurecia - AI project - Postural Monitoring

```
E:.
├───data_real
│   └───settings
├───data_simulations
│   └───files
├───modules
│   ├───ai_modules
│   │   ├───models
│   │   │   ├───empty_seat
│   │   │   ├───height_weight
│   │   │   └───postural_classifier
│   │   │       ├───data_loader
│   │   │       │   └───__pycache__
│   │   │       ├───encoding
│   │   │       │   ├───shapelet_transform
│   │   │       │   └───__pycache__
│   │   │       ├───models
│   │   │       │   └───__pycache__
│   │   │       ├───preprocessor
│   │   │       │   └───__pycache__
│   │   │       └───time_series
│   │   │           └───__pycache__
│   │   └───__pycache__
│   ├───common
│   │   └───__pycache__
│   ├───generic
│   │   └───__pycache__
│   ├───simple_modules
│   │   └───__pycache__
│   └───__pycache__
└───tools
    ├───docs
    └───scripts
        ├───hotspot_settings
        └───torch

```

- C Embedded - AUTOSAR - eRAD - BMOL

```
D:.
├───BUILD
└───SRC
    ├───HARD
    ├───MAIN
    ├───SWC_BSW
    │   ├───BSW
    │   ├───FAIL
    │   ├───MEMMAP
    │   ├───STD
    │   ├───STUB
    │   ├───SWC_BSW_CDD
    │   │   ├───APC
    │   │   ├───MCD
    │   │   ├───MCS
    │   │   ├───RDC
    │   │   ├───RES
    │   │   ├───SAFEDGM
    │   │   ├───SAFEMECH
    │   │   │   ├───BPU
    │   │   │   ├───CLK
    │   │   │   ├───ECCFLASH
    │   │   │   ├───MPU
    │   │   │   └───SMU
    │   │   ├───SAFETLIB
    │   │   │   ├───Common
    │   │   │   ├───SAFETLIB_CFG
    │   │   │   ├───SafeWdg
    │   │   │   ├───SafeWdgExtTlf
    │   │   │   ├───SafeWdgIf
    │   │   │   ├───SafeWdgQspi
    │   │   │   └───SMU
    │   │   ├───SAFEWDGM
    │   │   └───VSI
    │   ├───SWC_BSW_HAL
    │   │   └───IOHAL
    │   ├───SWC_BSW_LIB
    │   │   ├───F32SRV
    │   │   ├───MATH
    │   │   └───MATHSRV
    │   ├───SWC_BSW_MCAL
    │   │   ├───DET
    │   │   ├───DRV
    │   │   ├───EcuM
    │   │   ├───EMEM
    │   │   ├───IFX
    │   │   ├───ISR
    │   │   ├───MCAL
    │   │   ├───MCAL_CFG
    │   │   │   └───plugins
    │   │   ├───MCAL_TC26X_CAN
    │   │   ├───MCAL_TC26X_DIO
    │   │   ├───MCAL_TC26X_FLSLOADER
    │   │   ├───MCAL_TC26X_FLS_PMU
    │   │   ├───MCAL_TC26X_FR
    │   │   ├───MCAL_TC26X_GPT
    │   │   ├───MCAL_TC26X_ICU
    │   │   ├───MCAL_TC26X_MCU
    │   │   ├───MCAL_TC26X_PORT
    │   │   ├───MCAL_TC26X_PWM
    │   │   ├───MCAL_TC26X_SPI
    │   │   ├───MCAL_TC26X_WDG
    │   │   ├───OVC
    │   │   ├───PWD
    │   │   ├───SCU
    │   │   ├───STARTUP
    │   │   └───VADC
    │   ├───SWC_BSW_SRV
    │   │   ├───SWC_BSW_SRV_COM
    │   │   │   ├───BASE
    │   │   │   ├───CanIf
    │   │   │   ├───CanTp
    │   │   │   ├───CCP
    │   │   │   ├───CCPUSR
    │   │   │   ├───COM
    │   │   │   ├───COMM
    │   │   │   ├───COM_CFG
    │   │   │   │   └───plugins
    │   │   │   ├───DCM
    │   │   │   ├───E2E
    │   │   │   ├───FrIf
    │   │   │   ├───FrSm
    │   │   │   ├───FrTrcv_1_T01
    │   │   │   ├───IPDUM
    │   │   │   ├───MSG
    │   │   │   ├───OBD
    │   │   │   ├───PDUR
    │   │   │   ├───PSACOM
    │   │   │   ├───SCrc
    │   │   │   ├───SPY
    │   │   │   └───UDS
    │   │   ├───SWC_BSW_SRV_MEM
    │   │   │   ├───CRC
    │   │   │   ├───Fee
    │   │   │   ├───MemIf
    │   │   │   ├───NvM
    │   │   │   └───NVM_CFG
    │   │   └───SWC_BSW_SRV_SYS
    │   │       ├───DEVHAL
    │   │       ├───DSM
    │   │       ├───ESM
    │   │       ├───FAULT
    │   │       ├───FAULT_FF
    │   │       ├───FBL_INTF
    │   │       ├───HOOK
    │   │       ├───IMMO
    │   │       │   ├───AES128
    │   │       │   ├───IMMO
    │   │       │   ├───PRNG
    │   │       │   │   └───stub
    │   │       │   ├───RND
    │   │       │   ├───SAIMMO
    │   │       │   └───SAIMMO_Crypt
    │   │       ├───MEMSRV
    │   │       ├───OS
    │   │       │   └───_OS_DOC_
    │   │       ├───RSTSRV
    │   │       ├───RTMCLD
    │   │       ├───RTMTSK
    │   │       ├───SchM
    │   │       ├───STFSRV
    │   │       ├───SUPSRV
    │   │       ├───SWFAIL
    │   │       └───WDG
    │   └───_CFG_
    ├───SWC_SWA
    ├───SWC_TST
    │   ├───SWTEST_SAFEMECH
    │   ├───TST
    │   ├───TST_DET
    │   ├───TST_DSM
    │   ├───TST_FR_TRCV
    │   ├───TST_GPT
    │   ├───TST_MCD
    │   ├───TST_PWD
    │   ├───TST_PWM
    │   ├───TST_RSTSRV
    │   ├───TST_SPI
    │   ├───TST_VSI
    │   └───TST_WDG
    ├───_CALIB_
    ├───_DELIV_
    │   └───_SWC_BSW_DUMMY_
    ├───_DOCS_
    │   └───10_Release
    ├───_PRJ_
    └───_TOOLS_
        ├───_CANALYZER_
        ├───_CANOE_
        │   ├───VC1
        │   └───VF
        ├───_HEXVIEW
        │   ├───_examples
        │   └───_expdatproc
        ├───_LAUTERBACH_
        ├───_PLS_
        ├───_TELE_COMM_
        ├───_ULPMNGT
        │   └───UlpMngtKeys
        └───_WINADES_
            └───SWA_ERAD_BSWTC26_EnwWinAdes

```
- C Embedded E2CAD - No autosar

```
D:.
│   main.c
│
├───AppliFiles
│   │   adv64.h
│   │   analogH.c
│   │   analogH.h
│   │   analogl.c
│   │   analogl.h
│   ├───FromModel
│   │       cal_app.dat
│   │       cal_app.h
│   │       CM_MUX.c
│   │       CM_MUX.h
│   │       def_app.h
│   │       disp_app.c
│   │       disp_app.h
│   │       save_app.dat
│   │       save_app.h
│   │       tl_basetypes.h
│   │       tl_types.h
│   │
│   └───FromTool
│           DbgTable.c
│           DbgTable.h
│           FctTable.c
│           FctTable.h
│           VarFctSpi.c
│           VarFctSpi.h
│
└───SharedFiles
        cancfg.h
        canlow.h
        canlow.save
        canmgt.c
        canmgt.h
        cannm.h
        common.h


```

- **C Embedded AGCO (S2S3M1_dev) - No Autosar**

```
D:.
├───.vscode
├───Bootloader
│   ├───B_Sample
│   ├───C_Sample
│   │   └───Old
│   └───UDS
│       ├───V6.00
│       └───V9.00
├───Build
│   ├───Middleware
│   │   ├───B_Sample
│   │   │   ├───CORE
│   │   │   │   ├───Doc
│   │   │   │   ├───Include
│   │   │   │   ├───Lib
│   │   │   │   └───Source
│   │   │   ├───EOS
│   │   │   │   ├───Doc
│   │   │   │   ├───Include
│   │   │   │   ├───Lib
│   │   │   │   └───Source
│   │   │   └───KWP
│   │   │       ├───Doc
│   │   │       ├───Include
│   │   │       ├───Lib
│   │   │       └───Source
│   │   ├───GD
│   │   │   ├───Doc
│   │   │   ├───Include
│   │   │   ├───Lib
│   │   │   └───Source
│   │   ├───PLS
│   │   │   ├───Doc
│   │   │   ├───Include
│   │   │   ├───Lib
│   │   │   └───Source
│   │   ├───SRC14-34_32
│   │   │   ├───EOS
│   │   │   │   ├───Doc
│   │   │   │   ├───Include
│   │   │   │   ├───Lib
│   │   │   │   └───Source
│   │   │   ├───J1939_Stack
│   │   │   │   ├───Doc
│   │   │   │   ├───Include
│   │   │   │   ├───Lib
│   │   │   │   └───Source
│   │   │   ├───J1939_Stack_Int
│   │   │   │   ├───Doc
│   │   │   │   │   └───Doxygen
│   │   │   │   │       ├───html
│   │   │   │   │       └───latex
│   │   │   │   ├───Include
│   │   │   │   ├───Lib
│   │   │   │   └───Source
│   │   │   ├───KWP
│   │   │   │   ├───Doc
│   │   │   │   ├───Include
│   │   │   │   ├───Lib
│   │   │   │   └───Source
│   │   │   └───UDS
│   │   │       ├───Doc
│   │   │       ├───Include
│   │   │       ├───Lib
│   │   │       └───Source
│   │   └───Utilities
│   ├───Project
│   │   ├───Output
│   │   ├───Project
│   │   └───Utilities
│   └───WORK
│       ├───Function
│       │   ├───Arbitration
│       │   │   ├───Doc
│       │   │   ├───Include
│       │   │   ├───Lib
│       │   │   └───Source
│       │   ├───Brake
│       │   │   ├───Doc
│       │   │   │   └───ICD
│       │   │   ├───Include
│       │   │   ├───Include_BrakeManual
│       │   │   ├───Lib
│       │   │   ├───Source
│       │   │   └───Source_BrakeManual
│       │   ├───Commun
│       │   │   ├───Doc
│       │   │   ├───Include
│       │   │   │   └───SharedUtils
│       │   │   ├───Lib
│       │   │   └───Source
│       │   │       └───SharedUtils
│       │   ├───Counters
│       │   │   ├───Doc
│       │   │   │   ├───Doc Design
│       │   │   │   ├───Model_html
│       │   │   │   └───Old_doc
│       │   │   │       └───ICD
│       │   │   ├───Generated
│       │   │   │   ├───Counters_Cumul_Model
│       │   │   │   ├───Counters_Delta_Model
│       │   │   │   └───_sharedutils
│       │   │   ├───Include
│       │   │   ├───Lib
│       │   │   ├───Model
│       │   │   └───Source
│       │   ├───Debug
│       │   │   ├───Doc
│       │   │   ├───Include
│       │   │   ├───Lib
│       │   │   └───Source
│       │   ├───DMS
│       │   │   ├───Docs
│       │   │   ├───Include
│       │   │   ├───Lib
│       │   │   └───Source
│       │   ├───DriveAutomation
│       │   │   ├───Doc
│       │   │   │   ├───Level1
│       │   │   │   ├───Organigramme
│       │   │   │   └───Overview
│       │   │   ├───Generated
│       │   │   ├───Include
│       │   │   ├───Library
│       │   │   │   ├───DecelerationFlag
│       │   │   │   │   └───DecelFlag_ert_rtw
│       │   │   │   └───SpeedCalculator
│       │   │   │       └───Speed_Calculator_ert_rtw
│       │   │   └───Source
│       │   ├───EMM
│       │   │   ├───Docs
│       │   │   ├───Include
│       │   │   ├───Lib
│       │   │   └───Source
│       │   ├───Hitch
│       │   │   ├───Doc
│       │   │   ├───Include
│       │   │   ├───Lib
│       │   │   │   ├───hitch_control_model_ert_rtw
│       │   │   │   ├───hitch_grafcet_model_ert_rtw
│       │   │   │   └───hitch_pwm_valve_command_ert_rtw
│       │   │   └───Source
│       └───Main
│           ├───Include
│           └───Source
├───Miscellaneous
│   ├───Architecture
│   │   ├───Design
│   │   │   ├───1-S2-4WD
│   │   │   ├───10-S2-FrontHitch
│   │   │   ├───11-S2-RPTO
│   │   │   ├───12-S2-FPTO
│   │   │   ├───7-S2-HydraulicAux
│   │   │   ├───8-S2-FrontAxleSuspension
│   │   │   ├───9-S2-RearHitch
│   │   │   ├───Engine's Oil
│   │   │   ├───Fuel Level
│   │   │   ├───Keyboard
│   │   │   ├───Trailer Brake
│   │   │   └───Transmission's Oil
│   │   ├───Functionnal specification
│   │   │   ├───1-S2-4WD
│   │   │   ├───10-S2-FrontHitch
│   │   │   ├───11-S2-RPTO
│   │   │   ├───18-S2-Radio
│   │   │   ├───31-S2-Fan
│   │   │   ├───32-S2-Error
│   │   │   ├───33-S2-Information-SpeedManagement
│   │   │   ├───34-S2-Information-TractorHour-Maintenance
│   │   │   ├───35-S2-Information-Counter-Profil
│   │   │   ├───36-S2-Information-Unit-Language
│   │   │   ├───37-S2-Information-Buzzer
│   │   │   ├───38-S2-Dashboard
│   │   │   ├───4-S2-eTCV
│   │   │   ├───5-S2-HVAC
│   │   │   ├───6-S2-HTBV
│   │   │   ├───7-S2-HydraulicAux
│   │   │   ├───8-S2-FrontAxleSuspension
│   │   │   ├───9-S2-RearHitch
│   │   │   ├───Engine's Oil
│   │   │   ├───Fuel Level
│   │   │   ├───Hydraulic-Trailer-Brake
│   │   │   ├───Pneumatic-Trailer-Brake
│   │   │   ├───Polarion Documents
│   │   │   └───Transmission's Oil
│   │   ├───Hardware
│   │   │   └───Global Arch
│   │   │       └───Schéma Electrique
│   │   │           ├───S2-20 PLAN PRE SERIAL
│   │   │           ├───S2-20 SCHEMATIC PER CABINE TYP & TRANSMISION
│   │   │           └───S2-PLAN P1B
│   │   ├───Product specification
│   │   │   └───28-S2-Drive
│   │   └───ProjectScope
│   ├───Cabine_Variants
│   └───CAN_Databases
└───Tools
    ├───EEP
    │   ├───Factory
    │   ├───Mapping
    │   │   └───Macros
    │   └───P1
    │       ├───P1A_MF03
    │       ├───P1A_MF04
    │       ├───P1B_MF05
    │       └───S2
    ├───GD
    ├───GW
    └───PLS
```

- C Embedded - UBUNTU: PIC16F883-TurnKey

```
D:.
├───build
│   ├───codeblocks
│   └───mplab
├───doc
│   ├───Datasheet
│   │   ├───Camera OV6620
│   │   └───RS232
│   └───board
├───include
│   ├───Inputs
│   ├───Outputs
│   └───System
└───src
    ├───Inputs
    ├───Outputs
    └───System
```


- C++ Project

```
E:.
│   .gitignore
│   CMakeLists.txt
│   LICENSE
│   README.md
│
├───.github
│   └───workflows
│           ci.yml
│
├───app
│       CMakeLists.txt
│       adder_app.cpp
│
├───cmake
│       Catch.cmake
│       CatchAddTests.cmake
│
├───ext
│   └───catch2
│           catch.hpp
│
├───include
│   └───adder
│           adder.hpp
│
├───src
│       CMakeLists.txt
│       adder.cpp
│
└───tests
        CMakeLists.txt
        adder_t.cpp
        tests.cpp

```
Src: [SSC Github - Scientific Software Center, IWR, Heidelberg University](https://github.com/ssciwr/cpp-project-template)

- Python Project

```
.
├───data
└───modules
    ├───__pycache__
    └───common
```

- Rust Project


```
.
├───data
└───modules
    ├───__pycache__
    └───common
```

- Rust Embedded Project

```
.
├───data
└───modules
    ├───__pycache__
    └───common
```

- Computer Vision Project


```
.
├───data
└───modules
    ├───__pycache__
    └───common
```



## References

- AI 
- Healthcare
- Mobility: 
	- connectée
	- autonome 
	- durable


Project Structure - Organization and Templates Guidance

- AI - ML/DL
  - [Folder Structure for Machine Learning Projects](https://medium.com/analytics-vidhya/folder-structure-for-machine-learning-projects-a7e451a8caaa)
  - [The Ultimate Deep Learning Project Structure: A Software Engineer’s Guide into the Land of AI](https://ai.plainenglish.io/the-ultimate-deep-learning-project-structure-a-software-engineers-guide-into-the-land-of-ai-c383f234fd2f)
  - [How to Organize Deep Learning Projects – Examples of Best Practices](https://neptune.ai/blog/how-to-organize-deep-learning-projects-best-practices)

- Python: 
  - [Structuring Your Project - from The Hitchhickers guide to python](https://docs.python-guide.org/writing/structure/)
- C/C++
  - [C++ Project Structure and Cross-Platform Build With CMake](https://medium.com/swlh/c-project-structure-for-cmake-67d60135f6f5)
- Embedded / IoT
  - [HOW TO ORGANIZE A FIRMWARE PROJECT](https://www.beningo.com/how-to-organize-a-firmware-project/#)
  - [5 Steps To Designing An Embedded Software Architecture, Step 1 ](https://www.embedded.com/5-steps-to-designing-an-embedded-software-architecture-step-1/)
  - [Modular code and how to structure an embedded C project](https://www.microforum.cc/blogs/entry/46-modular-code-and-how-to-structure-an-embedded-c-project/) 
- Rust 
  - [Rust — Modules and Project Structure ](https://medium.com/codex/rust-modules-and-project-structure-832404a33e2e)
  - [Rust: Project structure example step by step](https://dev.to/ghost/rust-project-structure-example-step-by-step-3ee)

- Tools
  - [cookiecutter - project template generator](https://github.com/cookiecutter/cookiecutter)
  - [Cookiecutter Alternatives](https://alternativeto.net/software/cookiecutter/)





