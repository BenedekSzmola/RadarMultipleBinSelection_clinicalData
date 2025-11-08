"""
Module for loading saved vital-rate analysis results, converting timestamps, and plotting PSG vs Radar rates
for selected recordings.
"""

####################
##################
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
import numpy as np
import os
import warnings

# import again so the plotting works
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams['text.usetex'] = False
import matplotlib.pyplot as plt

import helperFunctions as hF
########################################################################################################################

def sec2hourMinSec(timeInput):
    """
    Convert seconds to (hours, minutes, seconds).

    Args:
        timeInput (int | float): Time in seconds.

    Returns:
        tuple: (hours, minutes, seconds) as integers.
    """
    hour = timeInput // 3600
    minRem = timeInput % 3600
    min = minRem // 60
    sec = minRem % 60

    return hour,min,sec

def makeSecTax2Hour(timeVec,ax2conv,recID=None):
    """
    Relabel the x-axis ticks from seconds to hour:minute format, optionally offset by subject start clock.

    Args:
        timeVec (array-like): Sequence of epoch start times in seconds.
        ax2conv (matplotlib.axes.Axes): Axes whose x-ticks and label are to be set.
        recID (str | None): Subject ID. If provided, offsets labels by perSubjStartClocks()[recID].

    Returns:
        None
    """
    
    newStamps = []
    newStampLabels = []
    roundStart = 1800 * np.ceil(timeVec[0]/1800).astype("int")
    for stampi in range(roundStart,int(timeVec[-1]),1800):
        if recID is None:
            stampiH,stampiM,_ = sec2hourMinSec(stampi)
        else:
            stampiH,stampiM,_ = sec2hourMinSec(stampi+hF.convertTimeStamp(hF.perSubjStartClocks()[recID]))
            if stampiH > 23:
                stampiH = stampiH - 24

        newStamps.append(stampi)
        newStampLabels.append(f'{stampiH:d}:{stampiM:02d}')

    ax2conv.set_xticks(newStamps,newStampLabels, rotation=45)
    ax2conv.set_xlabel("Time [hours:minutes]")
    
    return

### loading in the data, some parameters for that ###################################################################
radarLabels = {31: 'Radar@FootEnd'}
filePath = hF.giveSaveFilePath()
###############################################

# for paper2 consolidated vital rate interval parameters, filters
### -> put the path to your save files here
subDir = ""

saveFiles = np.array(os.listdir(filePath+subDir))

# Set whether you want to look at breathing ("BR") or heart rate ("HR") results
typeStr = "BR"

# recIDList = ["S0"+str(i) for i in range(27,75+1)] # plot all patients
recIDList = ["S070"] # plot just one
# recIDList = np.setdiff1d(recIDList,["S028","S032"]) # to exclude certain patients

saveFig = False
doCut = True

# Read dictionary pkl file
for recID in recIDList:
    saveFname = saveFiles[hF.findCorrectSave(saveFiles,[recID,typeStr,"","","",""])]

    if len(saveFname) == 0:
        if len(recIDList) == 1:
            raise Exception("No fitting savefile found!")
        else:
            warnings.warn(f"The recording with ID {recID} was not found! Skipping to next in list...")
            continue

    elif len(saveFname) != 1:
        prompt = "\n".join(f"{i}: {name}" for i, name in enumerate(saveFname))
        prompt = f"Select a file by index:\n{prompt}\n> "

        choice = int(input(print(prompt)))

        print('You chose: ',saveFname[choice])
        print('')
        if choice > (len(saveFname)-1):
            raise Exception("Bad index input!")
        else:
            saveFname = saveFname[choice]
    elif len(saveFname) == 1:
        saveFname = saveFname[0]

    with open(filePath+subDir+saveFname, 'rb') as fp:
        resultsDict = pickle.load(fp)

    print('###############################################################')
    print('### Showing plots from file: ###')
    print(f'### Directory: {filePath+subDir} ###')
    print(f'### File name: {saveFname} ###\n')

    if 'timeStarts' in resultsDict:
        timeStarts   = resultsDict['timeStarts']
    else:
        timeStarts = resultsDict['epochStarts']

    if 'timeWinLen' in resultsDict:
        timeWinLen   = resultsDict['timeWinLen']
    elif 'epochLen' in resultsDict:
        timeWinLen   = resultsDict['epochLen']
    elif 'mergeWin_epochLen' in resultsDict:
        timeWinLen   = resultsDict['mergeWin_epochLen']

    bins2check   = resultsDict['bins2check']
    if 'bestBins' in resultsDict:
        bestBins     = resultsDict['bestBins']
    elif 'selectedBins' in resultsDict:
        bestBins     = resultsDict['selectedBins']

    if 'computeTimes' in resultsDict:
        computeTimes = resultsDict['computeTimes']

    if 'radarIdx' in resultsDict:
        radarIdx = resultsDict['radarIdx']
    elif 'radar_idx' in resultsDict:
        radarIdx = resultsDict['radar_idx']
    else:
        radarIdx = np.nan

    boundOfUncertain = (1/timeWinLen) * 60

    if 'radarVitalRates' in resultsDict:
        radarData = resultsDict['radarVitalRates']
        psgData   = resultsDict['psgVitalRates']
    elif 'medianRadarVitalRates' in resultsDict:
        radarData = resultsDict['medianRadarVitalRates']
        psgData   = resultsDict['psgVitalRates']

    else:
        if typeStr == "BR":
            radarData = resultsDict['radarBRs']
            psgData   = resultsDict['psgBRs']

        elif typeStr == "HR":
            radarData = resultsDict['radarHRs']
            psgData   = resultsDict['psgHRs']

    print(recID, '# windows where bin was selected but no rate computed: ', sum((np.nansum(bestBins, axis=0) > 0) & np.isnan(radarData)))
    print('Total num windows: ', len(radarData))
    binSel_noRate_wins = timeStarts[(np.nansum(bestBins, axis=0) > 0) & np.isnan(radarData)]

    if bestBins.dtype == "int": # if bestbins is int it cannot be set to NaN where there were no vital rates computed
        bestBins = bestBins.astype("float")

    bestBins[:,radarData == 0]  = np.nan # in earlier saves, 0 signified no detection, but Ive changed to NaN since a while as that is clearer
    radarData[radarData == 0] = np.nan
    psgData[psgData == 0]     = np.nan

    # Handle mismatched saved lengths: trim timeStarts to radarData length.
    if len(timeStarts) > len(radarData):
        timeStarts = timeStarts[:len(radarData)]

    # if you want to cut the data and also have the stats reflect that:
    if doCut:
        if hF.perSubjCuts()[recID][0] > 0:
            cutStartInd = np.argmin(np.abs(hF.perSubjCuts()[recID][0] - timeStarts)) + 1
        else:
            cutStartInd = 0

        if np.isinf(hF.perSubjCuts()[recID][1]):
            cutEndInd = len(timeStarts)
        else:
            cutEndInd = np.argmin(np.abs(hF.perSubjCuts()[recID][1] - timeStarts)) + 1
        
        bestBins = bestBins[:,cutStartInd:cutEndInd]
        radarData = radarData[cutStartInd:cutEndInd]
        psgData = psgData[cutStartInd:cutEndInd]
        timeStarts = timeStarts[cutStartInd:cutEndInd]

    ###########################################
    # Code for creating the plot

    mae = np.nanmean(np.abs(psgData - radarData))
    winsWithBoth = len(np.where((~np.isnan(radarData)) & (~np.isnan(psgData)))[0])
    if winsWithBoth > 0:
        mape = (1/winsWithBoth) * np.nansum(np.abs(psgData - radarData) / np.abs(psgData)) * 100
    else:
        mape = np.nan

    diffStd = np.nanstd(np.abs(psgData - radarData))

    print(f'\n----------------------------\nShowing recID {recID}:\n----------------------------')

    # %matplotlib widget # if you want an interactive plot in the interactive window/jupyter environment
    plt.rc('font',size=15)
    fig,ax = plt.subplots(3,1,figsize=(16,10),sharex=True)
    fig.tight_layout()

    if 'radarPos' in resultsDict:
        radarPos = resultsDict['radarPos']
    else:
        try:
            radarPos = radarLabels[radarIdx]
        except:
            radarPos = 'Radar_PosNaN'

    if "withREMUSpaperMethods" in subDir:
        ax[0].set_title(f"{'Breathing' if typeStr=='BR' else 'Heart'} Rate Computed from Radar and PSG\nSingle Bin Selection")
    elif "with_multiBinSelPaper_scripts" in subDir:
        ax[0].set_title(f"{'Breathing' if typeStr=='BR' else 'Heart'} Rate Computed from Radar and PSG\nMultiple Bin Selection")

    ax[0].plot(timeStarts, psgData, 'bx', label='PSG', alpha=.75, markersize=6, fillstyle="none")
    ax[0].plot(timeStarts, radarData, 's', color='r', label="Radar", alpha=.75, markersize=3, fillstyle="none")
    if typeStr == "BR":
        ax[0].set_ylabel('Breathing rate [1/min]')
        ax[0].set_ylim([5,30])
    elif typeStr == "HR":
        ax[0].set_ylabel('Heart rate [1/min]')
        ax[0].set_ylim([40,120])

    ax[0].legend(loc="upper right")
    ax[0].grid(visible=True, which='major', axis='y')

    ax[1].plot(timeStarts, psgData - radarData,'o',markersize=3,fillstyle="none",label=f"PSG - Radar")
    ax[1].plot([timeStarts[0],timeStarts[-1]+timeWinLen],[0,0],'k--',label="Diff = 0")
    if typeStr == "BR":
        ax[1].set_title(f"Difference of PSG and Radar Breathing Rates (MAPE = {mape:.2f} %)")
        ax[1].set_ylabel('Breathing rate\ndifference [1/min]')
    elif typeStr == "HR":
        ax[1].set_title(f"Difference of PSG and Radar Heart Rates (MAPE = {mape:.2f} %)")
        ax[1].set_ylabel('Heart rate\ndifference [1/min]')
    
    ax[1].set_ylim([-5, 5])
    outOfBoundsTop = np.where((psgData - radarData) > 5)[0]
    if any(outOfBoundsTop):
        ax[1].plot(timeStarts[outOfBoundsTop],np.zeros(len(outOfBoundsTop))+4.7,'r^',label="Diff > 5")

    outOfBoundsBot = np.where((psgData - radarData) < -5)[0]
    if any(outOfBoundsBot):
        ax[1].plot(timeStarts[outOfBoundsBot],np.zeros(len(outOfBoundsBot))-4.7,'rv',label="Diff < -5")

    ax[1].legend(loc="upper right")
    ax[1].set_yticks(np.arange(-4,5))
    ax[1].grid(visible=True, which='major', axis='y')
    
    cax = ax[2].matshow(bestBins, extent=[timeStarts[0]-timeWinLen//2, timeStarts[-1]+timeWinLen//2, bins2check[0]*.05-.025, bins2check[-1]*.05+.025], aspect='auto', origin='lower', cmap='plasma')
    ax[2].xaxis.set_ticks_position('bottom')
    ax[2].set_xlabel('Time [s]')
    ax[2].set_ylabel("Distance from radar [m]")
    if typeStr == "BR":
        ax[2].set_title("Radar Breathing Range Profile")
    elif typeStr == "HR":
        ax[2].set_title("Radar Heartbeat Range Profile")

    makeSecTax2Hour(timeStarts,ax[2],recID)

    plt.tight_layout()

    if saveFig:
        ### -> put your desired path for the figures here
        figSavePath = ""
        if "withREMUSpaperMethods" in subDir:
            binSelTypeStr = "SBS"
        elif "with_multiBinSelPaper_scripts" in subDir:
            binSelTypeStr = "MBS"
        
        plt.savefig(figSavePath + f'{typeStr}_{binSelTypeStr}_exampResult_subj{recID}_{timeStarts[0]}-{timeStarts[-1]}.jpg',dpi=300,bbox_inches="tight")

    plt.show()

    plt.rc('font',size=10)