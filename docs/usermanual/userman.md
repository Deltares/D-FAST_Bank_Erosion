# Introduction

Since 2010 Deltares is working on a bank erosion module that can be used in combination with WAQUA.
The module computes local erosion sensitivity, visualizes the bank movement and gives an estimation of the amount of bank material that is released in the first year and after the equilibrium is reached.
WAQBANK can easily be used as a post processing tool of WAQUA-simulations.
Some examples for which WAQBANK can compute bank erosion are:

* Bank adjustments such as removal of bank protection
* Applying or removing fore shore protection
* Changes in shipping type or intensity
* Changes in currents (e.g. due to construction of side channels)

Output
The output of WAQBANK is:
1. Bank line movement after one year and when the final equilibrium state is reached.
1. Amount of sediment that is released from the banks after one year and when the final equilibrium state is reached.

This output is presented in graphs / figures and written to text files.

Input
The input of WAQBANK is:
1. WAQUA results (SDS-files)
1. Local data: level of bank protection removal, subsoil characteristics, shipping information (quantity, location of fairway)

The layout of this manual is as follows: In chapter 2 a description is given on how to use WAQBANK (installation, input, output).
The background of WAQBANK is outlined in chapters 3 and 4 (partly in Dutch).

# WAQBANK
## Installation

WAQBANK itself consists of 2 executables called 'BankLines.exe' and BankErosion.exe'.
The first is needed to establish the initial bank lines and the second computes the bank line movement and eroded volume.
Both executables of WAQBANK can be called from the DOS prompt.

To be able to run WAQBANK it is necessary to install MCRinstaller.
The required version of the installer is standard provided with WAQBANK.

By default WAQBANK is looking for the definition file deffile.def in the same directory as the executables.
When the definition file has another name or location, this should be passed when the modules are called.

## Scenario plan
WAQBANK is not calibrated for local bank erosion .
Therefore it is strongly advised to use WAQBANK only in a relative way.
In this way an image can be formed of
* at which locations the bank is most sensitive to erosion (by comparing different locations along the river)
* how sensitive a location is for certain measures (by comparing different scenarios at 1 location)

We advise to compute different scenarios and compare between them.
An example is: 1 scenario with a subsoil of sand and 1 scenario with a subsoil of clay.
This means that only the type of subsoil is changed and the other input remains unchanged.

## WAQUA-computations
Before WAQBANK can be used, steady state WAQUA-computations  have to be performed for each discharge level.
This first requires a schematisation of the discharge hydrograph in discharge levels and a probability per discharge step.
The discretisation of the discharge hydrograph is not a trivial procedure.
Deltares advises to use at least ten discharge levels.

By default two figures are generated during WAQBANK computation which the user can use to check if the discharge levels are chosen properly.
These are a figure with water levels and a figure with flow velocities.
In both figures the area that is sensitive to erosion is indicated.
Based on these figures it can be decided to include or remove discharge levels from the computation.
For each discharge level an SDS-file (WAQUA output file) must be provided.
WAQBANK uses the water levels and flow velocities that are stored in the last time step.
It is important that the WAQUA computations are in steady state.
The probability (normalized weighting of the time period) per discharge level is defined by the user in the definition file.

Note: It is of utmost importance that in the SDS-file with the reference (average) discharge (which is used to establish the initial bank lines) the water remains within the main channel during the whole computation.
Practically this means that the simulation has to be started with a low initial water level with no (or as little as possible) water in the flood plains.
When this criterion is not met, strange peaks in the detected bank lines may occur.



An example of discharge levels for the river Meuse in combination with different probabilities for different scenarios (wet/dry/intermediate) is given in Table 2.1.

| Dicharge level nr. | Discharge (m3/s) | 1998-2002 (wet) | 2004-2010 (dry) | 2008-2011 (intermediate) |
|--------------------|------------------|-----------------|-----------------|--------------------------|
| 1 | 75 | 0,2932 | 0,1808 | 0,3110 |
| 2 | 150 | 0,1918 | 0,2466 | 0,2329 |
| 3 | 272 | 0,1918 | 0,2603 | 0,2055 |
| 4 | 400 | 0,0411 | 0,0548 | 0,0685 |
| 5 | 500 | 0,1507 | 0,1370 | 0,0959 |
| 6 | 750 | 0,0548 | 0,0712 | 0,0548 |
| 7 | 900 | 0,0329 | 0,0384 | 0,0164 |
| 8 | 1100 | 0,0164 | 0,0082 | 0,0055 |
| 9 | 1300 | 0,0137 | 0,0014 | 0,0021 |
| 10 | 1500 | 0,0041 | 0,0014 | 0,0021 |
| 11 | 1700 | 0,0041 | 0,0000 | 0,0021 |
| 12 | 1900 | 0,0041 | 0,0000 | 0,0021 |
| 13 | 2300 | 0,0014 | 0,0000 | 0,0014 |

Table 2.1	Probabilities of a discharge level for different scenarios (De Vries, 2012)

## Definitionfile <defile.def>:
The definition file contains input parameters for the executables 'BankLines' en 'BankErosion'.
De executables search for the file 'defile.def' in the same directory as themselves, unless another file or path name is given.
The possible keywords and their description are given in Table 2.2.
The existence, not the order, of the keywords is of importance.

In the table the following abbreviations are used:
M = mandatory
O = optional
E = expert

Filenames can be either given with a single filename, or with their (relative) path and they should contain no spaces.
An example of a definition file is given in Figure 2.1.

Table 2.2  Keywords in the definition file of WAQBANK

| Keyword |  | Value | Description |
|---------|--|-------|-------------|
| Path | M | pathname | Pathname of used m-files (unnecessary when using executables)
| Nbank | M | integer | Number of bank lines, (standard 2 lines) |
| GridFile  | M | filename | Rgf-file for defining grid coordinates |
| RiverKM | M | filename | Textfile with riverkilometres and correspondig xy-coordinates |
| Boundaries  | O | integers | river chainage of the region of interst;  rkm-start:rkm-end, e.g. 81:100 (default: all) |
| NLim | O | integers | Range of N-values from SDS-file that is considered (default: all), only used when keyword 'Boundaries' is not available. |
| Mlim | O | integers | Range of M-values from SDS-file that is considered (default: all), only used when keyword 'Boundaries' is not available. |
| Plotting | O | logical | Plotting results (default: false) |
| SavePlots | O | Logical | Saving plots (default: true) |
| ClosePlots | O | Logical | Close plots and close Quickplot when simulation is finished (default: false) |
| Waqview | O | logical | Generating output for Waqview (default: false) |

Input BankLines.exe

| Keyword |  | Value | Description |
|---------|--|-------|-------------|
| SDS-file | M | filename | SDS-file for determining representative bank line |
| Line1    | M | filename | Textfile with xy-coordinates of search line 1 |
| LineN    | M | filename | Textfile with xy-coordinaten of search line N |
| BankDir | O | string | Directory for storing bank lines (default: current directory) |
| BankFile | O | filename | Text file(s) in which xy-coordinates of bank lines are stored (default 'bankfile') |
| LocalDir | O | filename | Directory for storing local output (default: 'local') |
| SortLim | E | real | Maximum number of vertices used for sorting (default: 50) |
| Waterdepth | E | real | Water depth used for defining bank line (default 0.0) |
| Dlines | E | nline reals | Distance from pre-defined lines used for determining bank lines (default: 50) |
| Dremove | E | nline reals | Ommiting coordinates that are more than this distance from neighbouring points (default: 5000) |

Input BankErosion.exe

| Keyword |  | Value | Description |
|---------|--|-------|-------------|
| Terosion | M | real | Simulation period  [years] |
| RiverAxis | M | filename | Textfile with xy-coordinates of river axis |
| Fairway  | M | filename | Textfile with xy-coordinates of fairway axis |
| BankType | M | filename/real | Bank strength definition (for each bank line per river-km) |
| Vship  | M | filename/real | Relative velocity of the ships (per river-km) [m/s] |
| Nship  | M | filename/integer | Number of ships per year (per river-km) |
| ShipType | M | filename/integer | Type of ship (per river-km) |
| Draught | M | filename/real | Draught of the ships (per river-km) [m] |
| NLevel | M | integer | Number of discharge levels |
| SDSfile1 | M | filename | SDS-file for computing bank erosion for discharge 1 (only used when 'Nlevel'>1) |
| PDischarge1 | M | real | Probability of discharge 1 (sum of probabilities should be 1) |
| SDSfileN | M | filename | SDS-file for computing bank erosion for discharge 'Nlevel' (only used when 'Nlevel'>1) |
| PDischargeN | M | real | Probability of discharge 'Nlevel' (sum of probabilities should be 1) |
| RefLevel | O | integer | Reference level: discharge level with SDS-file that is equal to 'SDS-file' (only used when 'Nlevel'>1)  (default: 1) |
| Classes | O | logical | Use classes (true) or critical shear stress (false) in 'BankType' (default: true) |
| ProtectLevel | O | filename | Text file(s) with level of bank protection for each bank line per river-km (default: -1000) |
| Slope | O | filename | Text file(s) with equilibrium slope for each bank line per river-km  (default: 20) |
| OutputDir | O | String | Directory for storing output files |
| BankNew | O | filename | Text file(s) in which new xy-coordinates of bank lines are stored (default 'banknew') |
| BankEq | O | filename | Text file(s) in which xy-coordinates of equilibrium bank lines are stored (default: 'bankeq') |
| EroVol | O | filename | Text file in which eroded volume per river-km is stored (default: 'erovol.evo') |
| OutputInterval | O | real | interval in which the output (eroded volume) is given (default: 1 km) [km] |
| VelFilter | E | logical | Filtering velocity along bank lines (default: true) |
| Revert | E | nline integers | Reverting direction of erosion (default 0) |
| Wave0 | E | real | Distance from fairway axis at which waveheight is zero (default 200 m) |
| Wave1 | E | real | Distance from fairway axis at which reduction of waveheigth to zero starts (default Wave0-50 m) |
| Nwave | E | integer | Number of waves per ship (default 5) |

    % General input parameters bank erosion module
    NBank          = 2
    GridFile       = inputfiles\maas40m_1.rgf
    RiverKM        = inputfiles\rivkm.xyc
    Boundaries     = 123:128
    WaqView        = true
    Plotting       = true
    %
    % Input parameters bank line detection
    Bankdir        = output\banklines
    SDSfile        = inputfiles\SDS-q272
    Line1          = inputfiles\oeverlijn_links_mod.xyc
    Line2          = inputfiles\oeverlijn_rechts_mod.xyc
    %
    % Input parameters bank erosion module
    OutputDir      = output\bankerosion
    Terosion       = 1
    RiverAxis      = inputfiles\maas_rivieras_mod.xyc
    Fairway        = inputfiles\maas_rivieras_mod.xyc
    Classes        = false
    BankType       = inputfiles\bankstrength_tauc
    Vship          = 5.0
    Nship          = inputfiles\nships_total
    ShipType       = 2
    Draught        = 2.5
    ProtectLevel   = inputfiles\stortsteen
    Slope          = inputfiles\slope
    BankEq         = bankequi
    EroVol         = erovol_standard.evo
    OutputInterval = 0.1
    Revert         = [1,0]
    %
    % Discharge dependent parameters
    NLevel         = 10
    RefLevel       = 3
    SDSfile1       = SDSfiles\SDS-q75
    PDischarge1    =   0.1808
    SDSfile2       = SDSfiles\SDS-q150
    PDischarge2    =   0.2466
    SDSfile3       = SDSfiles\SDS-q272
    PDischarge3    =   0.2603
    SDSfile4       = SDSfiles\SDS-q400
    PDischarge4    =   0.0548
    SDSfile5       = SDSfiles\SDS-q500
    PDischarge5    =   0.1370
    SDSfile6       = SDSfiles\SDS-q750
    PDischarge6    =   0.0712
    SDSfile7       = SDSfiles\SDS-q900
    PDischarge7    =   0.0384
    SDSfile8       = SDSfiles\SDS-q1100
    PDischarge8    =   0.0082
    SDSfile9       = SDSfiles\SDS-q1300
    PDischarge9    =   0.0014
    SDSfile10      = SDSfiles\SDS-q1500
    PDischarge10   =   0.0014

Figure 2.1	Example of a definition file for WAQBANK

## Banklines <BankLines.exe> :
determines the representative bank lines within the area of interest (for background information see chapter 3).
The input of Banklines is given through the definition file (deffile.def), see section 2.4.

When the definition file has the name deffile.def and is located in the same directory as the executable the module can be called as follows:

BankLines

If the definition file has another name and/or is located in another directory the following call should be used:

BankLines path\deffile_other.txt

with path the path to the directory where the definition file is located and deffile_other.txt the name of the definition file.

Required input:
* WAQUA-output file (SDS-file) at representative discharge (SDSfile),
* Number of bank Lines (Nbank default two, the left and right bank),
* For each of the bank lines a file with xy-coordinates of the estimated location of the bank  line (Line1, Line2, ..., LineN, with N the number of bank lines)
* File which links river kilometres to xy-coordinates (RiverKM),
Optional
* Area of interest in the form of a range of river kilometres or in terms of mn-coordinates (Boundaries or NLim and MLim).
* Name of the directory to which the output will be written (BankDir)

Output:
* XY-coordinates of the determined bank lines.
* Plot of the bank lines (optional, Plotting)

The computation has ended successfully when the message "BankLines ended successfully!" appears.

An example of the output as is generated with Banklines.exe is shown in Figure 2.2.
The water depth is given with colored surfaces (per grid cell), the black lines are the determined bank lines and the river kilometers are displayed on the river axis.

Note: To obtain the best results, the points of the estimated location of the banks given in  Line1, Line2, ..., LineN (obtained for example from the 'oeverlijnen' from Baseline data), should be equally distributed along the bank with a distance that is in the same order of the gridcellsize along the bank.
Large distances between the points will result in inaccurate bank lines.
Adding points inbetween the points will resolve this problem.
Points too close to eachother will result in large computation times, which can be solved by removing unnecessary points.

Figure 2.2	Example of output of  BankLines.exe


## BankErosion <BankErosion.exe>:
determines the expected bank erosion within the area of interest (for background information see chapter 4).
The input of BankErosion  is given through the definition file (deffile.def), see section 2.4.

When the definition file has the name deffile.def and is located in the same directory as the executable the module can be called as follows:

BankErosion

If the definition file has another name and/or is located in another directory the following call should be used:

BankErosion path\deffile_other.txt

with path the path to the directory where the definition file is located and deffile_other.txt the name of the definition file.

A clean start can be made with the command:

BankErosion deffile.def clean

This assures that all local files (in the directory LocalDir), which are specific for a certain reach and discharge levels, are removed.
At the next call of BankLines they are created once again.
The creation of these files can be time consuming, so it is recommended not to clean the files when the reach and/or used SDS-files are not changed.

Required input:
* The considered simulation time (Terosion, default 1 year),
* The number of discharge levels (NLevel),
* WAQUA-output files (SDS-files) for the different discharge levels and the corresponding probability distribution (SDSfile1, SDSfile2, ..., SDSfileM and PDischarge1, PDischarge2, ..., PDischargeM, with M the number of discharge levels).
When only 1 discharge level is given, the standard SDS-file (SDSfile) is used,
* Grid for positioning of results (GridFile), only needed when WaqView=true,
* Number of bank Lines (Nbank default two, the left and right bank),
* File which links river kilometres to xy-coordinates (RiverKM),
* XY-coordinates of the river and fairway axis (RiverAxis, Fairway),
* Information about the strength or type of the soil of the banks (BankType).
This can be done either in the form of classes (see Tabel 4.1 for values and explanation) or with a critical shear stress (see Table 5.1 for examples).
In the first case Classes=true should be set and in the second case Classes=false (default).
The bank strength information can be given either with a fixed value for the whole track or in a ASCII-file per river kilometer (first column river-km, second column bank type, see Figure 2.3),
* Shipping information (Vship, Nship, ShipType, Draught).
This can be done either with a fixed value for the whole track or in a file per river kilometer (first column river-km, second column shipping information: similar to entering bank type, see Figure 2.3).
Optional
* ASCII-file with level of bank protection (wrt NAP) for each bank line per river-km.
(First column river-km, second column bank protection level: similar to entering bank type, see Figure 2.3).
By default the bank protection level is 1 meter below the water level of the representative discharge,
* ASCII-file with slope of foreshore (1:n) for each bank line per river-km.
(First column river-km, second column slope parameter n: similar to entering bank type, see Figure 2.3).
Default a slope of 1:20 will be used,
* Name of the directory to which the output will be written (OutputDir),
* Name of the file to which the results will be written (BankNew, BankEq, EroVol),
* Name of the directory to which local files will be written (LocalDir).

Output:
* Map with waterdepth, initial bank lines, river kilometers and fairway axis (see Figure 2.4),
* Map with erosion sensitivity of the banks based on computed bank retreat (see Figure 2.5),
* The computed erosion volume during the simulation period split up for left and right bank and for each discharge level (see Figure 2.6),
* Total erosion volume based on equilibrium bank line (see Figure 2.7),
* Control figures for water levels of each discharge (see Figuur 2.8 and Figuur 2.9),
* Control figures for flow velocity of each discharge (see Figuur 2.10 and Figuur 2.11),
* Map with indication of applied bank type (see Figuur 2.12),
* Bank retreat at the end of the simulation period and for the equilibrium situation (see Figure 2.13),
* XY-coordinates of the computed bank lines at the end of the simulation period and of the equilibrium bank lines,
* Files to be able to visualize information in Waqview.

The computation has ended successfully when the message "BankErosion ended successfully!" appears.

  0.0          1
  75.2         3
  90.3         2
  110.0        0
  130.5        4
  153.1        3
  206.9        2
Figure 2.3 Example of input file for bank strength classes per river kilometer: from 0.0 - 75.2 km class 1, from 75.2 - 90.3 km class 3, etc.
When no information is available, the closest value will be used.



Figure 2.4 Waterdepth, initial bank lines, river kilometers and fairway axis



Figure 2.5 Erosion sensitivity




Figure 2.6 Eroded volume at the end of the simulation period	





Figure 2.7 Total erosion volume based on equilibrium bank line	



Figuur 2.8	Control figure for water levels of each discharge (bank line 1)



Figuur 2.9	Control figure for water levels of each discharge (bank line 2)



Figuur 2.10	



Figuur 2.11	


Figuur 2.12	Map with indication of applied bank type



Figure 2.13 Bank retreat at the end of the simulation period and for the equilibrium situation



# Detection of bank lines
The location of bank lines is determined by looking at the transition of water to land at a reference discharge computed with a WAQUA model.
To determine the exact location of the bank lines, first all cells in the WAQUA-grid are marked that are at the transition from land to water at these discharge level.
The concerned cells have positive water depth and one or more neighbouring cells with a zero water depth.
Within these transition cells a bank line can be defined.
The bank line is at the location where the water depth is equal to zero, that is the water level is equal to the bed level.
In WAQUA the bed level,  , is defined in the corner points of a cell and the water level,  , in the centre of a cell, see Figure 3.1.
The bank line is now found by interpolating the bed level linearly along the grid lines and marking the location where the bed level is equal to the water level.
In this way, two points are found which can be connected to form (a part of) the bank line.

Figure 3.1 Location of a bank line within a transition cell


An example is given in Figure 3.1, where   and   are smaller than   en   en   larger then   (water depth  ).

The found locations of the bank lines will depend on the chosen discharge level.
However, as long as the discharge is within the main channel, the locations will be fairly constant.

Generally, bank erosion only within the main channel.
It is therefore possible to only take into account those cells that are within a certain range of predefined lines.
This is also necessary when more than one bank line is present (as is common in rivers), because otherwise it is not clear which coordinates belong to one bank or the other.

For the Dutch rivers 'oeverlijnen' from Baseline can be used for these pre-defined lines.
These lines should be defined in a simple text file consisting of x- and y- coordinates (Line1,.., LineN, in the definition file).


One of the disadvantages of the 'oeverlijen' from Baseline is that they only depict the main channel.
Possible side channels, shortcuts or lakes will then not be detected by the tool, see
Figure 3.2.
If they are important, extra lines should be added that (globally) represent the bank lines of these features.


Figure 3.2 Detection of bank lines at a shortcut in the Muese river.
Red: Baseline 'oeverlijnen', Blue: detected bank line from WAQUA computation (Q=278,5 m3/s, average discharge)

Groynes are not detected as such by the tool, because they are defined on subgrid level in WAQUA, see Figure 3.3.
The detected bank line is in this case following the banks within groyne sections.
This is an advantage, since possible bank erosion only takes place in the groyne sections and not along the groynes themselves.

Figure 3.3 Detection of a bank line close to groynes.
Red: Baseline 'oeverlijnen', Blue: detected bank line from WAQUA computation (Q=278,5 m3/s, average discharge)


# Potential bank line shift and bank erosion
Wanneer de ligging van de initiele oeverlijn bekend is, kan de potentiele oeverlijnverschuiving en oeverafslag worden bepaald.
De ontwikkelde oevererosiemodule is bedoeld als hulpmiddel om de potentiele oevererosie in te kunnen schatten en niet om de daadwerkelijke oevererosie te voorspellen.
Binnen de oevererosiemodule worden twee erosiemechanismen meegenomen: erosie door scheepsgolven en erosie door stroming.
Deze mechanismen worden in de volgende twee paragrafen verder uitgewerkt.
Verder wordt in paragraaf 0 uitgelegd hoe de oeverlijnverschuiving in zijn werk gaat.
De bepaling van het potentieel geerodeerd volume wordt uitgewerkt in paragraaf 4.4.
In paragraaf 4.5 wordt uitgelegd hoe er kan worden omgegaan met een variabel afvoerniveau.
Het bepalen van de evenwichtsoever en het daarbij behorende geerodeerde volume wordt uitgelegd in paragraaf 4.6.
Ten slotte worden enkele van de beperkingen van de oevererosiemodule aangestipt in paragraaf 4.7.

## Determining potential erosion by ship waves

Scheepsgolven zijn een van de belangrijkste factoren wat betreft oeverafslag (Verheij (2000))  en worden daarom als eerste oevererosiemechanisme meegenomen in de oevererosiemodule.
 De afleiding voor de erosieformulering voor scheepsgolven is overgenomen uit de BEM module (Verheij (2000), Stolker & Verheij (2001b)) en definities van grootheden zijn gegeven in Figuur 4.1.



Figuur 4.1 Definities grootheden erosie door golven


Op basis van grootschalige Deltagootproeven, waarbij diverse taluds van verschillende samenstelling werden belast onder loodrechte golfaanval, is afgeleid dat de erosiesnelheid kwadratisch toeneemt met de golfhoogte bij de oever:

	 	(1.1)

met:
 	: breedte van de afgekalfde oeverstrook [ ]
 	: sterkte coefficient voor oevermateriaal [ ]
 	: golfhoogte bij de oever [ ]

In het algemeen wordt verondersteld dat de afname van de golfhoogte via een negatieve e-macht is gerelateerd aan de breedte van de oeverstrook:

	 	(1.2)

Hierin is:
 	: initiele golfhoogte aan het begin van de vooroever [ ]
 	: parameter voor golfdemping [ ]

Substitutie van vergelijking (1.2) in vergelijking (1.1) levert een differentiaalvergelijking op met de algemene oplossing:
	
waarbij:

 	: afstand waarmee de oeverlijn verschuift [ ]
 	: tijd [ ]

Deze formule is toepasbaar voor zowel wind- als scheepsgolven.
In geval van scheepsgolven kan voor de tijd   worden uitgegaan van de volgende relatie

  met  .

waarbij:

 	: periode van de scheepsgolven [ ]
 	: aantal ongeladen schepen per jaar [ - ]
 	: aantal golven per schip [ - ]
 	: vaarsnelheid schepen [ ]
 	: valversnelling [9.81  ]
  	: beschouwde periode [jaar]

De waarde voor de initiele golfhoogte aan het begin van de vooroever   kan met behulp van formules zoals uit DIPRO worden berekend aan de hand van het type maatgevende schepen, hun vaarsnelheid en diepgang, de afstand tussen de vaargeul en de oever en de waterdiepte (zie Bijlage A5A).

De parameter   kan worden gebruikt om de invloed in rekening te brengen van de vorm van de vooroever op de golfdemping, maar ook het dempende effect van vooroeverconstructries, vegetaties, en afzettingen van oevermateriaal.
In de oevererosiemodule wordt alleen het effect van de helling van de vooroever op de golfdemping meegenomen.
De dempingsterm voor glooiende bodemhellingen wordt als volgt bepaald (Verheij (2000)):

	
Waarbij   voor een vooroever met een helling van 1 : n (zie ook Figuur 4.1).
Uitgaande van een initiele golfhoogte   leidt een helling tussen de 1 : 100 en 1 : 20 tot waarden van   liggend tussen 0.025 en 0.125.
In Figuur 4.2 is een voorbeeld gegeven van de invloed van de dempingsparameter   op de oevererosie door golven.

Figuur 4.2 Voorbeeld van oevererosie door golven voor matig/goede klei.



Naast de golfdemping door de helling van de vooroever kunnen de inkomende golven ook worden gedempt door de begroeiing.
Voor golfdemping door riet is de volgende relatie beschikbaar
	
met   de rietstengeldichtheid (aantal stengels per vierkante meter).
De totale golfdemping door de helling van de vooroever en riet wordt dan
	

De waarde voor de sterkte van het oevermateriaal   hangt af van de samenstelling van de oever en kan ruimtelijk varieren.
Waarden voor   voor verschillende oeversamenstellingen zijn te vinden in Tabel 4.1.
Per oeverlijn moet in een tekstbestand worden aangegeven uit welke klasse het oevermateriaal bestaat voor een bepaald riviertraject.

Tabel 4.1 Klassenindeling grondsoorten oevererosiemodule
Klasse	Grond	cE (m-1s-1)	  (Pa)

0	Beschermde oeverlijn	0	8
1	Begroeide oeverlijn	0,02 10-4	95
2	Goede klei	0,6 10-4	3
3	Matig / slechte klei 	2 10-4	0,95
4	Zand	12,5 10-4	0,15

## Determining potential erosion by currents

Naast oeverafslag door scheepsgolven kan ook een sterke stroming langs de oever zelf zorgen voor oevererosie.
Dit mechanisme wordt ook meegenomen in de oevererosiemodule.
Voor elke oeverlijn kan de potentiele oevererosie door stroming bij een bepaalde afvoer   worden bepaald aan de hand van de volgende formule:
	
	
met

 	: afstand waarmee de oeverlijn verschuift in periode   [ ]
 	: erosiecoefficient van de oever [ ]
 	: stroomsnelheid langs de oeverlijn [ ]
 	: kritische stroomsnelheid voor erosie [ ]
 	: beschouwde periode [ ]

De erosiecoefficient wordt bepaald volgens:

	
met     en   [ ] de kritische schuifspanning voor erosie.
Deze waarde voor de erosiecoefficient is vergelijkbaar met erosiecoefficienten die in de literatuur gevonden worden (bijv.
Crosato, 2007).

Deze relatie tussen de kritische schuifspanning en de erosiecoefficient is echter niet universeel en daarom is het ook mogelijk om de erosiecoefficient als afzonderlijke invoer op te geven.

Voor de kritische stroomsnelheid voor erosie geldt:

	

met   [ ] de Chezy coefficient voor hydraulische ruwheid.
De waarde voor de Chezy coefficient wordt overgenomen uit de WAQUA berekening.

De stroomsnelheid langs de oever wordt bepaald aan de hand van de stroomsnelheid uit WAQUA.
De waarde voor de kritische schuifspanning voor erosie,  , is gerelateerd aan de sterkte coefficient voor oevermateriaal,  , zoals beschreven in Bijlage B)

## Total bank shift

De totale oeverlijnverschuiving wordt gevonden door te sommeren over de oeverlijnverschuivingen veroorzaakt door de verschillende erosiemechanismen:

	

De nieuwe locatie van een oeverlijn wordt bepaald door elk lijnsegment te verplaatsen volgens zijn lokale verschuiving.
De nieuwe locatie van een punt van de oeverlijn wordt gevonden door het snijpunt van de twee naburige segmenten te berekenen, zie Figuur 4.3.
Echter, in sommige situaties kan dit resulteren in erg grote verplaatsingen van punten, vooral wanneer naburige segmenten bijna in elkaars verlengde liggen.
In deze gevallen worden eerst twee locaties bepaald gebaseerd op de verplaatsing van elk van de naburige segmenten (rode stippen in Figuur 4.4).
De uiteindelijke locatie van een punt wordt dan bepaald door het gemiddelde van deze twee punten te nemen (groene stip in Figuur 4.4).


 	
Figuur 4.3 Het verschuiven van een oeverlijn gebaseerd op het snijpunt van twee lijnsegmenten.
Blauw: oorspronkelijke locatie, Groen: nieuwe locatie
	Figuur 4.4	Het verschuiven van een oeverlijn gebaseerd op de gemiddelde verplaatsing van twee lijnsegmenten.
Blauw: oorspronkelijke locatie, Rood: nieuwe locatie gebaseerd op individuele segmenten, Groen: nieuwe locatie (gemiddelde van de rode punten).

Om numerieke problemen te voorkomen worden te kleine oeverlijnsegementen samengevoegd met hun buren.
## Potential bank erosion volume

Naast de potentiele oeverlijnverschuiving is ook het potentiele volume aan sediment dat vrijkomt door erosie van belang.
Deze hoeveelheid sediment komt uiteindelijk in de rivier terecht, wat gevolgen kan hebben voor de bodemligging en eventueel noodzaak kan geven tot extra baggerwerkzaamheden.
Het potentiele volume aan sediment dat vrijkomt door erosie kan worden afgeschat door

	

waarbij   de totale oeverlijnverschuiving en   het invloedsgebied van het afslagproces (boven en onder de waterspiegel).

Er wordt verondersteld dat de oever in zijn geheel terugschrijdt en daarom geldt voor het invloedsgebied boven de waterspiegel:
	
met
 	: waterspiegelniveau [ ]
 	: niveau bovenkant steiloever [ ]

De grootte van het invloedsgebied onder de waterspiegel wordt bepaald door de golfhoogte en het niveau tot waar het stortsteen blijft liggen.
Hiervoor wordt de volgende relatie gebruikt:

	
waarbij:
 	: waterspiegelniveau [ ]
 	: niveau bovenkant stortsteen [ ]
H	: golfhoogte bij de oever [ ]

In Figuur 4.5 is geschematiseerd weergegeven wat het geerodeerd volume is voor verschillende situaties van de waterstand.


	 	

Figuur 4.5 Geerodeerd volume voor verschillende situaties van de waterstand.


De terugschrijding van de oever wordt volledig bepaald door de oeverafslag, ongeacht of het materiaal dat voor de oever op de bodem terecht komt meteen wordt afgevoerd of niet.
Het afgeslagen materiaal beinvloedt wel de vervolgerosie, omdat het in zekere zin de oever beschermt.
De invloed hiervan kan niet worden bepaald door alleen een sedimentbalans op te stellen a.d.h.v.
een sedimenttransportformule, omdat soms grote hompen klei blijven liggen die eerst moeten desintegreren voordat de stroming ze kan meenemen.
De invloed van het afgeslagen materiaal op de golfhoogte (en daardoor de erosie door scheepsgolven) kan wel worden meegenomen door de parameter voor golfdemping, , te verhogen.

## Variable discharge

Het meenemen van een variabele afvoer is mogelijk door de afvoerverdeling te schematiseren met een hydrograaf.
Deze hydrograaf bestaat uit   afvoerniveaus en hun bijbehorende kans op voorkomen.
Voor elk afvoerniveau wordt een aparte WAQUA-berekening uitgevoerd.
Uit deze WAQUA simulaties kunnen dan de stroomsnelheid langs de oeverlijn en het niveau van de waterspiegel worden afgeleid.
Hierbij wordt er vanuit gegaan dat in ieder geval de afvoer wordt meegenomen die is gebruikt om de initiele oeverlijn te bepalen (de gemiddelde afvoer).

Vervolgens worden oeverlijnverschuivingingen voor de afzonderlijke afvoerniveaus bepaald en deze worden daarna gewogen gesommeerd aan de hand van de kans van voorkomen van het betreffende afvoerniveau.
Voor elk afvoerniveau   wordt dus eerst de totale erosie   voor die afvoer bepaald, die bestaat uit een deel veroorzaakt door scheepsgolven en een deel veroorzaakt door stroming.

Een variabele afvoer betekent dat ook het niveau van de waterspiegel tijdsafhankelijk is en daarmee de locatie waar de oevererosie optreedt.
In de oevererosiemodule wordt echter uitgegaan van een initiele oeverlijn (behorende bij een gemiddelde afvoer).
De mate van erosie van deze lijn varieert echter wel met de afvoer.


### Erosion by ship waves

Een varierend afvoerniveau zorgt voor een varierend niveau van de waterspiegel.
De initiele oeverlijn is echter alleen onderhevig aan erosie door golven bij een gegeven afvoer   als de hoogte van de lijn in het invloedsgebied   ligt, waarbij   de golfhoogte bij de oever en   het niveau van de waterspiegel bij afvoer  .
In de voorbeelden weergegeven in Figuur 4.6 is de oeverlijn dus alleen onderhevig aan erosie door golven bij afvoerniveaus   en  .
Of een oeverlijn binnen het invloedsgebied ligt waarin erosie door golven moet worden meegenomen, kan plaatsafhankelijk zijn.
In het benedenstroomse gebied liggen de waterniveaus bij verschillende afvoeren in het algemeen dichter bij elkaar dan bovenstrooms en ook vlakbij een stuw met een opgegeven stuwpeil kan het waterniveau redelijk constant blijven voor verschillende afvoeren.


a)	
b)

c)	
d)
Figuur 4.6 Erosie door golven bij meerdere afvoerniveaus: a) geen erosie, oeverlijn ligt boven invloedsgebied b) erosie, c) erosie, d) geen erosie, oeverlijn ligt onder invloedsgebied

### Erosion by currents

Voor elk afvoerniveau wordt de erosie door stroming bepaald met behulp van de stroomsnelheid langs de oeverlijn behorende bij het betreffende afvoerniveau.
Hierbij is er natuurlijk alleen stroming langs de oeverlijn als deze lijn op of onder de waterspiegel ligt.
In de voorbeelden weergegeven in Figuur 4.6 is de oeverlijn dus alleen onderhevig aan erosie door stroming bij afvoerniveaus   en  .
### Total erosion

De totale verschuiving van de oeverlijn bij het meenemen van meerdere afvoerniveaus wordt gevonden door de oeverlijnverschuivingen bij de verschillende afvoeren gewogen te sommeren over alle afvoerniveaus:

	 	(1.3)
waarbij:

  		: totale verschuiving van de oeverlijn [m]
 		: aantal gebruikte afvoerniveaus [-]
 		: jaarlijkse kans op afvoer   [-]
 	: totale verschuiving van de oeverlijn bij afvoer   [m]
 	: verschuiving van de oeverlijn door scheepsgolven bij afvoer   [m]
 	: verschuiving van de oeverlijn door stroming bij afvoer   [m]

De verschuiving van de oeverlijn kan dan weer op dezelfde manier worden bepaald als uitgewerkt in paragraaf 0, maar nu met de totale oeverlijnverschuiving zoals gegeven in vergelijking (1.3).
De potentiele oeverafslag dient eerst per afvoerniveau te worden berekend (zie paragraaf 4.4).
Daarna kan de totale potentiele oeverafslag worden bepaald door gewogen te sommeren met de kans van voorkomen van het betreffende afvoerniveau, analoog aan de bepaling van de totale oeverlijnverschuiving (vergelijking (1.3)).

## Determining equilibrium bank
In van der Mark (2011), Hoofdstuk 3 is een analyse gemaakt van kentallen voor te verwachten oevererosie langs de IJssel.
Daarin wordt gesteld dat de meest waarschijnlijke te verwachten taludhelling van de evenwichtsoever 1:20 is.
Aan de hand hiervan kan een schatting worden gegeven van de erosieafstand   voor het behalen van de evenwichtssituatie (zie ook Figuur 4.7):
	
Waarbij   de inverse van de taludhelling van de evenwichtsoever (default waarde 1/20) en   met   en
	
Het totaal afgeslagen volume voor het bereiken van een evenwichtsoever is dan gelijk aan

	
met


Figuur 4.7	Geschatte profiel evenwichtsoever.

## Limitations of WAQBANK

De oevererosiemodule is bedoeld als hulpmiddel om in te kunnen schatten waar potentieel oevererosie plaats kan vinden en niet om de daadwerkelijke oevererosie te voorspellen.
In deze sectie worden een paar beperkingen van de oevererosiemodule aangestipt.
### Homogeneous bank

Een van de belangrijkste beperkingen is dat de samenstelling van de oever homogeen wordt verondersteld.
Dit is in de werkelijkheid niet het geval.
Er worden vaak horizontale of verticale lagen waargenomen.
In Figure 4.8 worden beide situaties weergegeven en de manier waarop het erosieproces plaatsvindt in het geval van horizontale lagen.
Om verticale lagen te kunnen modelleren, kunnen verschillende waarden van   worden gebruikt voor elke laag.
De situatie met horizontale lagen is complexer, omdat in dit geval plotseling grote hoeveelheden oevermateriaal in een keer naar beneden kunnen glijden.


a) vertical layers						b) horizontal layers


c) erosion process in case of horizontal layers

Figure 4.8 Non-homogeneous banks.


### Only erosion along initial bank line

Er wordt alleen oevererosie bepaald voor de opgegeven initiele oeverlijn (behorende bij een gemiddelde afvoer).
In werkelijkheid is het echter ook mogelijk dat er oevererosie op andere locaties plaatsvindt.

### Only erosion by ship waves and currents
In de oevererosiemodule wordt alleen erosie door scheepsgolven en stroming gemodelleerd.
Andere erosiemechanismen zoals windgolven, uitstromend grondwater, bevriezing en vertrapping door vee worden in de module niet meegenomen.

### Profile and bed level remain constant

Door oevererosie verandert lokaal de breedte van de rivier en ook zorgt het afgeslagen materiaal voor een andere bodemligging.
Hierdoor veranderen ook de stromingscondities in de rivier en langs de oever.
Aangezien de oevererosiemodule is gebaseerd uitkomsten van WAQUA berekeningen wordt deze dynamiek van de bodem en oevers echter niet meegenomen.

# References
Crosato, A. (2007), Effects of smoothing and regridding in numerical meander migration models. Water Resources Research, Vol 43.

CUR, (1996). Erosie van onverdedigde oevers, CUR-rapport 96-7, Gouda.

IWACO, Waterloopkundig Laboratorium en CSO (1998). Trajectnota/MER Zandmaas-Maasroute, Ontwerp natuurvriendelijke oevers. eindrapport 3361070.

Mark, R. van der, R.A.M. van der Sligte, A. Becker, E. Mosselman & H.J. Verheij (2011), Morfologische effectstudie KRW-maatregelen IJssel.
Rapport 1204885, Deltares, Delft.

Schuurman, F. (2010), Casestudie oevererosie Maas, DHV memo kenmerk LW-AF20100800/RK, 30 december 2010.

Sieben, J., J.A.F. van Essen, M.J.M. Scholten and L.W.J. van Hal, (2005), Inschatting van de kans op achterloopsheid bij kribben, RIZA werkdocument 2005.148x.

Sieben, J. (2005), Wie het water deert, die het water keert - Inventarisatie van beheer en onderhoud van kribben en oeververdedigingen, RIZA werkdocument 2005.034x.

Spruyt, A. (2009), Shifting margins, behaviour of dynamic banks, Internal memorandum 200186-003-ZWS-0003-v2-m-Oeverdetectie, Deltares, Delft, 4 november 2010.

Spruyt, A. & E. Mosselman (2010), Deelproject 3: Schuivende marges, gedrag van dynamische oevers (Shifting margins, behaviour of dynamic banks).
Appendix C in TO Rivierkundige Onderzoeksthema's; Rapportage 2009, Rapport 1200186.000, Deltares.

Stolker, C. & H.J. Verheij (2001a), Gevoeligheidsonderzoek sedimentatie en erosie in kribvakken langs de Lek. Rapport Q2792, WL | Delft Hydraulics, Delft, Februari 2001.

Stolker, C. & H.J. Verheij (2001b), Calibratie van een oeverafslagmodel voor de Zandmaas. Rapport Q3060, WL | Delft Hydraulics, Delft, November 2001.

TAW (1998), Technisch rapport erosiebestendigheid van grasland als dijkbekleding, Technische Adviescommissie voor de Waterkeringen, Delft.

Verheij, H.J., Meijer, D.G., Kruse, G.A.M., Smith, G.M., Vesseur, M., (1995). Investigation of the strength of a grass cover upon river dikes [in Dutch], Report Q1878, Deltares, Delft.

Verheij, H.J. (2000), Samenwerkingsproject Modellering Afslagoevers. Rapport Q2529, WL | Delft Hydraulics, Delft.

Verheij, H.J., F. van der Knaap & H. Sessink (2007), Verder ontwikkelen van oeverafslagmodel BEM. Rapport Q4264.02, WL | Delft Hydraulics, Delft.

Vroeg, J.H. de (1999), Breach growth in cohesive materials : selection of cases. Rapport H3468, WL | Delft Hydraulics, Delft.


# Strength of bank material for different soil types
The value for the coefficient for strength of bank material, cE , and the critical shear stress for erosion,  , for different soil types is given in Table 5.1 (Verheij et al, 1995, CUR, 1996 TAW, 1998, de Vroeg, 1999).

The following relation is applicable:
	
with   [ ].

Table 5.1 Corresponding values for cE and  (source: Verheij et al, 1995)
Grond	cE (m-1s-1)	  (Pa)

Protected bank	0	8
Sturdy grass	0,01 10-4	185
Mediocre grass	0,02 10-4	93
Bad grass	0,03 10-4	62
Very good clay (compact)	0,5 10-4	4
Clay with 60% sand (firm)	0,6 10-4	2,5
Good clay with  little structure	0,75 10-4	2
Strongly structure good clay
(mediocre)	1,5 10-4	1,5
Bad clay (weak)	3,5 10-4	0,65
Sand with 17% silt	10 10-4	0,20
Sand with 10% silt	12,5 10-4	0,15
Sand with 0% silt	15 10-4	0,10


# Determining ship induced wave height at the beginning of the fore shore
To determine the ship induced wave height at the beginning of the fore shore,   [m], the following formula is used (source: Handleiding DIPRO, 1997)

	
with

 	: ship dependent coefficient [ - ]
 	: Froude number (F<0.8) [ - ]
 	: water depth (considering a  trapezoidal profile) [ ]
 	: distance of shore to ship [ ]

The Froude number is computed according to:
	
with

 	: gravity acceleration [ 9,81  ]
 	: ship velocity [ ]

The value of the Froude number is limited to 0.8.

For the coefficient   the following values are used, with   [m] the draught of the ships:
Ship type	

Push barge	0,5
RHK ship / Motorship	

Towboat	1,0

By using this formulas, the value of  can be computed based on the dominant ship type, their velocity and draught, the distance between fairway and shore and the water depth.

To prevent wave load on smaller channels far from the main channel, the wave height is smoothly reduced to zero  from distance   to   from the fairway.
This is accomplished by multiplying   with the following function:

	
	 	

The value for   will be in the order of 150 - 200 m and for  the following relation is used:   .
 In Figure 5.1 the wave height   as function of the distance from the fairway   is depicted for various values of the water depth    for a moter ship with a draught of 1.2 m with a relative velocity of  m/s.
The wave height is reduced to zero between  m and  m.

Figure 5.1 Wave height as function from the distance from the fairway for different values of the water dept.
