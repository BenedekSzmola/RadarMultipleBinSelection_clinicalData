"""
Module for loading saved vital-rate analysis results, selecting files for specific subjects,
cutting to sensible data intervals, and computing/printing summary statistics comparing PSG vs Radar
(e.g., coverage, agreement thresholds, error metrics, and correlations).

The script expects saved .pkl result files in a configured directory.
"""

import pickle
import numpy as np
from scipy import stats
import os
import copy

import helperFunctions as hF
###########################################################################


### loading in the data, some parameters for that ###################################################################
radarLabels = {31: 'Radar@FootEnd'}
filePath = hF.giveSaveFilePath()
###############################################

### -> put the path to the savefiles here
subDir = ""
saveFiles = np.array(os.listdir(filePath+subDir))

typeStr = "HR"
binPreFilt = ""
binSel = ""
rateComp = ""
xtraTag = ""

# recIDList = ["S0"+str(i) for i in range(27,75+1)] # plot all patients
recIDList = ["S070"] # plot just one
# recIDList = np.setdiff1d(recIDList,["S028","S032"]) # to exclude certain patients

doCut = True

# Read dictionary pkl file
allResultsDict = {}
for recID in recIDList:
    saveFname = saveFiles[hF.findCorrectSave(saveFiles,[recID,typeStr,"","","",""])]

    if len(saveFname) == 0:
        raise Exception("No fitting savefile found!")
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
        allResultsDict[recID] = pickle.load(fp)
    ##########################################################################

allComputeTimes  = np.array([])
fullPsgData   = np.array([])
fullRadarData = np.array([])
numWins            = 0
timeInSec          = 0
winsWithPSG        = 0
winsWithRadar      = 0
winsWithBoth       = 0
winsWithPsgNoRadar = 0
winsWithNoPsgRadar = 0
winsWithNeither    = 0
winsWithinCritDist = 0
winsRealClose      = 0
psgNonCoveredSec   = 0
radarNonCoveredSec = 0
bothValidIntervalSec = 0
psgAllBadIntervalLens = np.array([])
radarAllBadIntervalLens = np.array([])

for rInd,recID in enumerate(recIDList):
    resultsDict = allResultsDict[recID]

    if 'timeStarts' in resultsDict:
        timeStarts   = resultsDict['timeStarts']
    else:
        timeStarts = resultsDict['epochStarts']

    timeStep     = timeStarts[1] - timeStarts[0]
    
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

    if 'doBinPreSel' in resultsDict:
        binFiltMethod = resultsDict['doBinPreSel']
        if binFiltMethod:
            binFiltMethod = 'magStd'
        else:
            binFiltMethod = 'noBinFilt'

    elif 'binFiltMethod' in resultsDict:
        binFiltMethod = resultsDict['binFiltMethod']
    else:
        binFiltMethod = 'not saved'

    if 'binSelMethod' in resultsDict:
        binSelMethod = resultsDict['binSelMethod']
    else:
        binSelMethod = 'not saved'

    if 'detMethod' in resultsDict:
        detMethod = resultsDict['detMethod']
    else:
        detMethod = 'not saved'

    if 'useSameForRef' in resultsDict:
        useSameForRef = resultsDict['useSameForRef']
    else:
        useSameForRef = False

    if 'radarIdx' in resultsDict:
        radarIdx = resultsDict["radarIdx"]
    elif 'radar_idx' in resultsDict:
        radarIdx = resultsDict["radar_idx"]
    else:
        radarIdx = np.nan

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

    bestBins[:,radarData == 0]  = np.nan
    radarData[radarData == 0] = np.nan
    psgData[psgData == 0]     = np.nan

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

    if np.sum(np.isnan(psgData)) > (timeWinLen/timeStep):
        _,badIntervalLens = hF.intervalExtractor(np.where(np.isnan(psgData))[0])
        badIntervalLens = badIntervalLens[badIntervalLens >= (timeWinLen/timeStep)] - (timeWinLen/timeStep - 1)
        psgAllBadIntervalLens = np.concatenate((psgAllBadIntervalLens,badIntervalLens))
        psgNonCoveredSec += np.sum(badIntervalLens) * timeStep

    if np.sum(np.isnan(radarData)) > (timeWinLen/timeStep):
        _,badIntervalLens = hF.intervalExtractor(np.where(np.isnan(radarData))[0])
        badIntervalLens = badIntervalLens[badIntervalLens >= (timeWinLen/timeStep)] - (timeWinLen/timeStep - 1)
        radarAllBadIntervalLens = np.concatenate((radarAllBadIntervalLens,badIntervalLens))
        radarNonCoveredSec += np.sum(badIntervalLens) * timeStep

    bothValidInds = np.where(~np.isnan(radarData) & ~np.isnan(psgData))[0]
    bothValidIntervals,bothValidIntervalLens = hF.intervalExtractor(bothValidInds)
    if len(bothValidIntervalLens) > 0:
        bothValidIntervals = bothValidInds[bothValidIntervals].astype("float")
        for i in range(len(bothValidIntervalLens)-1):
            if (bothValidIntervals[i+1,0] - bothValidIntervals[i,1]) < (timeWinLen / timeStep):
                bothValidIntervals[i+1,0] = copy.deepcopy(bothValidIntervals[i,0])
                bothValidIntervals[i,:] = np.nan

        bothValidIntervals = bothValidIntervals[np.all(~np.isnan(bothValidIntervals), axis=1),:]
        bothValidIntervalLens = np.diff(bothValidIntervals, axis=1)
        bothValidIntervalSec += np.sum([timeWinLen + (leni-1)*timeStep for leni in bothValidIntervalLens])
    
    fullPsgData   = np.concatenate((fullPsgData,psgData))
    fullRadarData = np.concatenate((fullRadarData,radarData))
    if 'computeTimes' in resultsDict:
        allComputeTimes = np.concatenate((allComputeTimes,computeTimes))

    # how many timewindows had detected activity PSG & radar
    numWins            += len(timeStarts)

    timeInSec          += timeStarts[-1] + timeWinLen - timeStarts[0]

    winsWithPSG        += len(np.where(~np.isnan(psgData))[0])

    winsWithRadar      += len(np.where(~np.isnan(radarData))[0])

    winsWithBoth       += len(np.where((~np.isnan(radarData)) & (~np.isnan(psgData)))[0])

    winsWithPsgNoRadar += len(np.where((np.isnan(radarData)) & (~np.isnan(psgData)))[0])

    winsWithNoPsgRadar += len(np.where((~np.isnan(radarData)) & (np.isnan(psgData)))[0])

    winsWithNeither    += len(np.where((np.isnan(radarData)) & (np.isnan(psgData)))[0])

    # how many timewindows had PSG & radar within a certain value (probably different for HR BR)
    if typeStr == "BR":
        crtiDist = 3

    elif typeStr == "HR":
        crtiDist = 5

    winsWithinCritDist += len( np.where( np.abs(psgData - radarData) <= crtiDist )[0] )

    boundOfUncertain = (1/timeWinLen) * 60
    winsRealClose += len( np.where( np.abs(psgData - radarData) <= boundOfUncertain )[0] )

print('##################################################')
print('Results from subdir: ')
print(subDir)
print('##################################################\n')

print(f'######### {"Respiratory rate" if typeStr=="BR" else "Heart rate"} analysis subjects (n={len(recIDList)}): {recIDList} #########')
print(f'Radar used: #{radarIdx} ({radarLabels[radarIdx]})')
print('Info about methods used:')
print(f'Bin pre filtering enabled: {binFiltMethod}')
print(f'Bin selection method used: {binSelMethod}')
print(f'Activity detection method used: {detMethod}')
print(f'Were the same methods used for reference: {useSameForRef}')
print('##################################################\n')

## compute times stuff
if 'compute_times' in allResultsDict[recIDList[0]]:
    allSubj_computeTimesDict = {timeName: np.zeros(len(recIDList)) for timeName in allResultsDict[recIDList[0]]['compute_times'].keys()}
    for recInd,recID in enumerate(allResultsDict.keys()):
        for timeName in allSubj_computeTimesDict.keys():
            allSubj_computeTimesDict[timeName][recInd] = allResultsDict[recID]['compute_times'][timeName]

    print('--------------------------------------')
    print('Running time stats: ')
    for runningTimeStat in allSubj_computeTimesDict.keys():
        print(f'{runningTimeStat}: mean={np.mean(allSubj_computeTimesDict[runningTimeStat])} | 1-25-50-75-99 percentiles: {[np.percentile(allSubj_computeTimesDict[runningTimeStat], perc) for perc in [1,25,50,75,99]]}')
else:
    allSubj_computeTimesSum = []
    for recID in allResultsDict.keys():
        allSubj_computeTimesSum.append(np.sum(allResultsDict[recID]['computeTimes']))
    allSubj_computeTimesSum = np.array(allSubj_computeTimesSum)
    print('--------------------------------------')
    print('Running time stats: ')
    print(f'Compute times sum: mean={np.mean(allSubj_computeTimesSum)} | 1-25-50-75-99 percentiles: {[np.percentile(allSubj_computeTimesSum, perc) for perc in [1,25,50,75,99]]}')

print('----------------------------------------------\n')

recHours = timeInSec // 3600
minRem = timeInSec % 3600
recMins = minRem // 60
recSecs = minRem % 60
recDurStr = f"{recHours:.0f} hours, {recMins:.0f} minutes, {recSecs:.1f} seconds"
print(f"Total number of time windows: {numWins} , with window length: {timeWinLen} s , step of {int(timeStep)} s , full duration analysed: {recDurStr}")
print(f"Percent of windows with detected PSG {typeStr} activity: {100*winsWithPSG/numWins: .1f}% ({winsWithPSG} windows)")
print(f"Percent of windows with detected Radar {typeStr} activity: {100*winsWithRadar/numWins: .1f}% ({winsWithRadar} windows)")
print(f"Percent of windows with both PSG and Radar detected {typeStr} activity: {100*winsWithBoth/numWins: .1f}% ({winsWithBoth} windows)")
bothValidH = bothValidIntervalSec // 3600
minRem = bothValidIntervalSec % 3600
bothValidMin = minRem // 60
bothValidS = minRem % 60
bothValidStr = f"{bothValidH:.0f} hours, {bothValidMin:.0f} minutes, {bothValidS:.1f} seconds"
print(f'Seconds where both psg and radar had rates: {bothValidIntervalSec} ({bothValidStr})')
print('')

nonCovH = psgNonCoveredSec // 3600
minRem = psgNonCoveredSec % 3600
nonCovMin = minRem // 60
nonCovS = minRem % 60
nonCovStr = f"{nonCovH:.0f} hours, {nonCovMin:.0f} minutes, {nonCovS:.1f} seconds"
print(f'Seconds where no psg rate was computed: {psgNonCoveredSec} ({nonCovStr})')

nonCovH = radarNonCoveredSec // 3600
minRem = radarNonCoveredSec % 3600
nonCovMin = minRem // 60
nonCovS = minRem % 60
nonCovStr = f"{nonCovH:.0f} hours, {nonCovMin:.0f} minutes, {nonCovS:.1f} seconds"
print(f'Seconds where no radar rate was computed: {radarNonCoveredSec} ({nonCovStr})')
print('')

print(f'              | With PSG | Without PSG ')
print(f'---------------------------------------')
print(f'With Radar    | {winsWithBoth:8.0f} | {winsWithNoPsgRadar:11.0f} ')
print(f'---------------------------------------')
print(f'Without Radar | {winsWithPsgNoRadar:8.0f} | {winsWithNeither:11.0f} ')
print(f'---------------------------------------\n')

print(f"Difference between PSG & Radar within +- {crtiDist}: {winsWithinCritDist} windows")
print(f"    as proportion of time windows with both: {100*winsWithinCritDist/winsWithBoth: .1f}%")
print(f"    as proportion of time windows with PSG:  {100*winsWithinCritDist/winsWithPSG: .1f}%")
print(f"Difference between PSG & Radar within boundaries of uncertainty (+-{boundOfUncertain:.1f}): {winsRealClose} windows")
print(f"    as proportion of time windows with both: {100*winsRealClose/winsWithBoth: .1f}%")
print(f"    as proportion of time windows with PSG:  {100*winsRealClose/winsWithPSG: .1f}%")
print('')
# MSE
mse = np.nanmean((fullPsgData - fullRadarData)**2)
print(f"MSE between PSG and Radar:  {mse: .2f}")
print("")

mae = np.nanmean(np.abs(fullPsgData - fullRadarData))
print(f"MAE between PSG and Radar:  {mae: .2f}")
print("")

mape = (1/winsWithBoth) * np.nansum(np.abs(fullPsgData - fullRadarData) / np.abs(fullPsgData)) * 100
print(f"MAPE between PSG and Radar: {mape: .2f}%")
print("")

## correlation coefficients
# Remove NaN values pairwise
mask = ~np.isnan(fullPsgData) & ~np.isnan(fullRadarData)
x_clean = fullPsgData[mask]
y_clean = fullRadarData[mask]

# Check if data is sufficient
if len(x_clean) < 3:
    print("Not enough data points after removing NaNs.")
else:
    # Test for normality (Shapiro-Wilk)
    shapiro_x = stats.shapiro(x_clean)
    shapiro_y = stats.shapiro(y_clean)

    print("Shapiro-Wilk Test Results:")
    print(f"  x: W = {shapiro_x.statistic:.3f}, p = {shapiro_x.pvalue:.3e}")
    print(f"  y: W = {shapiro_y.statistic:.3f}, p = {shapiro_y.pvalue:.3e}")

    # Interpretation
    if shapiro_x.pvalue > 0.05 and shapiro_y.pvalue > 0.05:
        print("Both distributions are likely normal (p > 0.05)")
    else:
        print("One or both distributions may not be normal (p <= 0.05)")

    # Pearson correlation
    pearson_corr, pearson_p = stats.pearsonr(x_clean, y_clean)
    print(f"\nPearson correlation: r = {pearson_corr:.3f}, p = {pearson_p:.3e}")

    # Spearman correlation
    spearman_corr, spearman_p = stats.spearmanr(x_clean, y_clean)
    print(f"Spearman correlation: ρ = {spearman_corr:.3f}, p = {spearman_p:.3e}")

