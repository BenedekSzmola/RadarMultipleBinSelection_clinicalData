"""
Module for loading saved vital-rate analysis results, generating Bland–Altman plots,
and visualizing range-bin selection distributions across subjects.

It:

- Builds the save directory path.
- Locates result files by substring matching.

- Provides per-subject sensible data cut intervals.
- Plots Bland–Altman comparisons for PSG vs Radar (for BR/HR) under Single- vs Multiple-Bin Selection.
- Aggregates and plots per-bin selection counts across subjects.

"""

import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
import seaborn as sns
import pandas as pd
import numpy as np
import os

# import again so the plotting works
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams['text.usetex'] = False
import matplotlib.pyplot as plt

import helperFunctions as hF
###########################################################################

### custom Bland-Altman plot
def make_custom_BlandAltman_plot(sensor1data_dict,sensor2data_dict,typeStr,saveFig=False):
    """
    Create side-by-side Bland–Altman plots (SBS vs MBS) comparing PSG and Radar rates.

    Args:
        sensor1data_dict (dict): Dictionary with keys:

            - "SBS": 1D array-like of PSG rates (or sensor A) across epochs.
            - "MBS": 1D array-like of PSG rates (or sensor A) across epochs.

        sensor2data_dict (dict): Dictionary with keys:

            - "SBS": 1D array-like of Radar rates (or sensor B) across epochs.
            - "MBS": 1D array-like of Radar rates (or sensor B) across epochs.

        typeStr (str): "BR" for breathing rate or "HR" for heart rate (used in titles/labels).
        saveFig (bool): If True, save the figure to a hard-coded path.

    Returns:
        None: Displays the figure and optionally saves it.

    Notes:

        - NaN pairs are excluded per point before computing mean/difference.
        - Mean difference and ±1.96 SD agreement lines are drawn and annotated.

        - Uses hard-coded save directory if saveFig=True; consider parametrizing in production.

    """
    _,ax = plt.subplots(1,2,figsize=(24,10))

    for binSel_ind,binSelType in enumerate(["SBS","MBS"]):
        sensor1data = sensor1data_dict[binSelType]
        sensor2data = sensor2data_dict[binSelType]
        
        bothNotNan = np.where(~np.isnan(sensor1data) & ~np.isnan(sensor2data))[0]
        sensorDiffs = sensor1data[bothNotNan] - sensor2data[bothNotNan]
        sensorAvgs = np.mean(np.concatenate((sensor1data[bothNotNan].reshape(1,-1), sensor2data[bothNotNan].reshape(1,-1))), axis=0)

        xrange = np.max(sensorAvgs) - np.min(sensorAvgs)
        xmin = np.min(sensorAvgs) - xrange*.025
        xmax = np.max(sensorAvgs) + xrange*.3

        meanLine = np.mean(sensorDiffs)
        upperCIline = np.mean(sensorDiffs) + 1.96*np.std(sensorDiffs, ddof=1)
        lowerCIline = np.mean(sensorDiffs) - 1.96*np.std(sensorDiffs, ddof=1)

        yrange = upperCIline - lowerCIline
        ymin = lowerCIline - yrange*.1
        ymax = upperCIline + yrange*.1

        ax[binSel_ind].plot(sensorAvgs, sensorDiffs, 'bo', markersize=3, alpha=.5)

        lineText_fontsize = 24
        ax[binSel_ind].hlines(y=meanLine, xmin=xmin, xmax=xmax, colors='k', linestyles='-')
        if meanLine < 0:
            ax[binSel_ind].text(.93*xmax, meanLine + yrange*.05, f'mean=$-${abs(meanLine):.2f}',
                horizontalalignment='center', verticalalignment='center',
                fontsize=lineText_fontsize, weight="normal", color='black')
        else:
            ax[binSel_ind].text(.93*xmax, meanLine + yrange*.05, f'mean={meanLine: .2f}',
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=lineText_fontsize, weight="normal", color='black')

        ax[binSel_ind].hlines(y=upperCIline, xmin=xmin, xmax=xmax, colors='k', linestyles='--')
        ax[binSel_ind].text(.91*xmax, upperCIline + yrange*.05, f'$+$1.96SD={upperCIline: .2f}',
                horizontalalignment='center', verticalalignment='center',
                fontsize=lineText_fontsize, weight="normal", color='black')

        ax[binSel_ind].hlines(y=lowerCIline, xmin=xmin, xmax=xmax, colors='k', linestyles='--')
        ax[binSel_ind].text(.91*xmax, lowerCIline - yrange*.05, f'$-$1.96SD=$-${abs(lowerCIline):.2f}',
                horizontalalignment='center', verticalalignment='center',
                fontsize=lineText_fontsize, weight="normal", color='black')

        ax[binSel_ind].set_title(
            f"Bland\u2013Altman Plot: PSG vs Radar {"Breathing" if typeStr=="BR" else "Heart"} Rates \n {"Single" if binSelType=="SBS" else "Multiple"} Bin Selection",
            fontsize=28
        )

        ax[binSel_ind].set_xlabel("Mean of sensors [BPM]", fontsize=24)
        ax[binSel_ind].tick_params(axis='x', labelsize=22)
        ax[binSel_ind].set_ylabel("Difference of sensors [BPM]", fontsize=24)
        ax[binSel_ind].tick_params(axis='y', labelsize=22)

        ax[binSel_ind].set_ylim([ymin,ymax])

    plt.tight_layout()

    if saveFig:
        ### -> put the path where the figures should be saved here
        figSavePath = ""
        plt.savefig(figSavePath + f"{typeStr}_SBS_and_MBS_B-A_plots.jpg",dpi=300,bbox_inches="tight")

    plt.show()
    plt.rc('font',size=10)


filePath = hF.giveSaveFilePath()

binPreFilt = ""
binSel = ""
rateComp = ""
xtraTag = ""

recIDList = ["S0"+str(i) for i in range(27,75+1)]
recID_count = len(recIDList)

doCut = True
saveFigs = False

binSel_counts_dict_bothTypes = {}
fullPsgData_dict_bothTypes = {}
fullRadarData_dict_bothTypes = {}
for typeStr in ["BR","HR"]:
    fullPsgData_dict = {}
    fullRadarData_dict = {}
    binSel_metrics_dict = {}
    binSel_counts_dict = {}
    for binSel in ["SBS","MBS"]:

        ### -> put the paths where the savefiles are here
        if binSel == "SBS":
            subDir = f""
        elif binSel == "MBS":
            subDir = f""

        saveFiles = np.array(os.listdir(filePath+subDir))

        # Read dictionary pkl file
        allResultsDict = {}
        for recID in recIDList:

            saveFname = saveFiles[hF.findCorrectSave(saveFiles,[recID,typeStr,binPreFilt,"",rateComp,xtraTag])]

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

        fullPsgData   = np.array([])
        fullRadarData = np.array([])
        # collect metrics per subject
        perSubj_metric_list = []
        perSubj_selBinSums = np.zeros((42,len(recIDList)))

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

            fullPsgData   = np.concatenate((fullPsgData,psgData))
            fullRadarData = np.concatenate((fullRadarData,radarData))

            perSubj_selBinSums[:,rInd] = np.sum(bestBins,axis=1)

            currSubj_numWins            = len(timeStarts)

            currSubj_timeInSec          = timeStarts[-1] + timeWinLen - timeStarts[0]

            currSubj_winsWithPSG        = len(np.where(~np.isnan(psgData))[0])

            currSubj_winsWithRadar      = len(np.where(~np.isnan(radarData))[0])

            currSubj_winsWithBoth       = len(np.where((~np.isnan(radarData)) & (~np.isnan(psgData)))[0])

            currSubj_winsWithPsgNoRadar = len(np.where((np.isnan(radarData)) & (~np.isnan(psgData)))[0])

            currSubj_winsWithNoPsgRadar = len(np.where((~np.isnan(radarData)) & (np.isnan(psgData)))[0])

            currSubj_winsWithNeither    = len(np.where((np.isnan(radarData)) & (np.isnan(psgData)))[0])

            boundOfUncertain = (1/timeWinLen) * 60
            currSubj_winsRealClose = len( np.where( np.abs(psgData - radarData) <= boundOfUncertain )[0] )

            # collect the per subject metrics
            if len(np.where((~np.isnan(radarData)) & (~np.isnan(psgData)))[0]) > 0:
                currSubj_epochDiffs_realClose_ratio = currSubj_winsRealClose/currSubj_winsWithBoth
            else:
                currSubj_epochDiffs_realClose_ratio = np.nan

            currSubj_mae = np.nanmean(np.abs(psgData - radarData))

            if len(np.where((~np.isnan(radarData)) & (~np.isnan(psgData)))[0]) > 0:
                currSubj_mape = (1/len(np.where((~np.isnan(radarData)) & (~np.isnan(psgData)))[0])) * np.nansum(np.abs(psgData - radarData) / np.abs(psgData)) * 100
            else:
                currSubj_mape = np.nan

            perSubj_metric_list.append({
                'Patient ID': recID,
                'Epochs_w_psgRate_count': currSubj_winsWithPSG,
                'Epochs_w_psgRate_ratio': currSubj_winsWithPSG/currSubj_numWins,
                'Epochs_w_radarRate_count': currSubj_winsWithRadar,
                'Epochs_w_radarRate_ratio': currSubj_winsWithRadar/currSubj_numWins,
                f'Epochs_diff_<={boundOfUncertain}_count': currSubj_winsRealClose,
                f'Epochs_diff_<={boundOfUncertain}_ratio': currSubj_epochDiffs_realClose_ratio,
                'MAE': currSubj_mae,
                'MAPE': currSubj_mape
            })

        fullPsgData_dict[binSel] = fullPsgData
        fullRadarData_dict[binSel] = fullRadarData

        fullPsgData_dict_bothTypes[typeStr] = fullPsgData_dict
        fullRadarData_dict_bothTypes[typeStr] = fullRadarData_dict


        binSel_metrics_dict[binSel] = pd.DataFrame(perSubj_metric_list)
        binSel_metrics_dict[binSel]['RangeBinSel'] = binSel

        binSel_counts_dict[binSel] = perSubj_selBinSums
    #############################################################################
    binSel_counts_dict_bothTypes[typeStr] = binSel_counts_dict

    full_perSubj_metric_df = pd.concat((binSel_metrics_dict['SBS'],binSel_metrics_dict['MBS']), ignore_index=True)

    metricNames = list(full_perSubj_metric_df.columns)[1:-1]

    make_custom_BlandAltman_plot(fullPsgData_dict,fullRadarData_dict,typeStr,saveFig=saveFigs)
    
##############################################################################################################################################
##############################################################################################################################################
fig,ax = plt.subplots(2,2,figsize=(24,12))
for type_ind, typeStr in enumerate(["BR","HR"]):
    vitalRateStr = "Breathing" if typeStr=="BR" else "Heartbeat"

    for binSel_ind,binSel in enumerate(["SBS","MBS"]):
        testDF = pd.DataFrame(binSel_counts_dict_bothTypes[typeStr][binSel], columns=recIDList, index=np.round(np.arange(9,51)*0.05,2))
        testDF_long = testDF.reset_index().melt(id_vars='index', var_name='Subject', value_name='Bin Selection Count')
        testDF_long = testDF_long.rename(columns={'index': 'Range Bin'})

        sns.barplot(data=testDF_long, x='Range Bin', y='Bin Selection Count', color='b', err_kws={'color': 'r'}, ax=ax[type_ind,binSel_ind])
        ax[type_ind,binSel_ind].set_xticks(np.arange(0,42,3), np.round(np.arange(9,51,3)*0.05,2), rotation=45)
        ax[type_ind,binSel_ind].tick_params(axis='x', labelsize=20)
        ax[type_ind,binSel_ind].set_xlabel("Distance from radar [m]", fontsize=22)
        ax[type_ind,binSel_ind].set_ylabel("Selection count", fontsize=22)
        ax[type_ind,binSel_ind].tick_params(axis='y', labelsize=20)
        ax[type_ind,binSel_ind].set_title(f'{vitalRateStr} {"Single" if binSel=="SBS" else "Multiple"} Bin Selection - Selected Range Bins', fontsize=26)

plt.tight_layout()

if saveFigs:
    ### -> put the path where the figures should be saved here
    figSavePath = ""
    plt.savefig(figSavePath + f'binSelDistributions_allInOne.jpg',dpi=300,bbox_inches="tight")

plt.show()

##############################################################################################################################################
##############################################################################################################################################

