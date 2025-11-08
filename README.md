# Analysis scripts for the manuscript "Radar Multiple Bin Selection for Breathing and Heart Rate Monitoring in Acute Stroke Patients in a Clinical Setting"
__Authors__: Benedek Szmola, Lars Hornig, Jan Paul Vox, Thomas Liman, Andreas Radeloff, Birger Kollmeier, Karen Insa Wolf, and Karsten Witt  
__Contact:__ benedek.ernoe.szmola@uni-oldenburg.de  

This repository contains the scripts which were used for the analyses presented in the manuscript. The scripts to make the plots are also included.

## Description of scripts
- _MBS\_....py_ - these scripts are for the multiple bin selection
- _SBS\_....py_ - these scripts are for the single bin selection
- the other scripts are for plotting, statistics computation, or they store parameters  

## Usage
### Defining paths for saving/loading files
- In the below listed scripts, there are variables where you can input the paths where you want the results savefiles to go, and then you should give those paths to the stats and plotter scripts so they can retrieve the data. In each of the scripts you can search for "\### ->" to find the places to fill in
    - _helperFunctions.py_
    - _MBS\_main.py_
    - _SBS\_main.py_
    - _overnight\_result\_plotter.py_
    - _stats\_computer.py_
    - _summary\_result\_plotter.py_

### Computing breathing and heart rate
- _MBS\_main.py_ and _SBS\_main.py_ have both a main function which executes the breathing and heart rate computation using multiple and single range bin selection respectively
    - The parameters which were used for the manuscript are set as defaults in the function definition. Just execute the functions without specifying any of the parameters.

### Computing statistics, creating plots
For all of the below scripts, find the _recIDList_ variable to set which patients you want to include in the stats/plots
- Use the _stats\_computer.py_ script to compute the summary statistics (error values, correlations, time windows with detections, etc...)
- for the summary plots (Bland-Altman and distribution of range bins) use the _summary\_result\_plotter.py_
- for the individual overnight plots use the _overnight\_result\_plotter.py_ script

## License
MIT License

Copyright (c) 2025 BenedekSzmola

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
