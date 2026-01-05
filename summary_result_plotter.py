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
from scipy import stats,signal
import os
import copy

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
        ax[binSel_ind].hlines(y=meanLine, xmin=xmin, xmax=xmax, colors='k', linestyles='-', linewidth=3)
        if meanLine < 0:
            ax[binSel_ind].text(.93*xmax, meanLine + yrange*.05, f'mean=$-${abs(meanLine):.2f}',
                horizontalalignment='center', verticalalignment='center',
                fontsize=lineText_fontsize, weight="normal", color='black')
        else:
            ax[binSel_ind].text(.93*xmax, meanLine + yrange*.05, f'mean={meanLine: .2f}',
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=lineText_fontsize, weight="normal", color='black')

        ax[binSel_ind].hlines(y=upperCIline, xmin=xmin, xmax=xmax, colors='k', linestyles='--', linewidth=3)
        ax[binSel_ind].text(.91*xmax, upperCIline + yrange*.05, f'$+$1.96SD={upperCIline: .2f}',
                horizontalalignment='center', verticalalignment='center',
                fontsize=lineText_fontsize, weight="normal", color='black')

        ax[binSel_ind].hlines(y=lowerCIline, xmin=xmin, xmax=xmax, colors='k', linestyles='--', linewidth=3)
        ax[binSel_ind].text(.91*xmax, lowerCIline - yrange*.05, f'$-$1.96SD=$-${abs(lowerCIline):.2f}',
                horizontalalignment='center', verticalalignment='center',
                fontsize=lineText_fontsize, weight="normal", color='black')

        # shading area within 95% confidence interval
        ax[binSel_ind].axhspan(lowerCIline, upperCIline, color='grey', alpha=0.3)

        # compute how many points are outside CI lines
        ratio_aboveCI = sum(sensorDiffs > upperCIline) / len(sensorDiffs)
        ratio_belowCI = sum(sensorDiffs < lowerCIline) / len(sensorDiffs)
        print(f"{"Breathing" if typeStr=="BR" else "Heart"} Rates {"Single" if binSelType=="SBS" else "Multiple"} Bin Selection")
        print(f'% of points below CI {ratio_belowCI*100:.2f} , and above CI {ratio_aboveCI*100:.2f}')
        
        ax[binSel_ind].set_title(
            f"Bland\u2013Altman Plot: PSG vs. Radar {"Breathing" if typeStr=="BR" else "Heart"} Rates \n {"Single" if binSelType=="SBS" else "Multiple"} Bin Selection",
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

    return

def plot_subjectLvl_stats_MBSvsSBS(typeStr,stat_df,stat_name,boundOfUncertain=1.0,saveFig=False):
    """
    Plot and summarize subject-level performance statistics comparing
    Single Range Bin Selection (SBS) versus Multiple Range Bin Selection (MBS).

    This function computes descriptive statistics (percentiles, min/max values,
    and threshold counts) for a specified metric separately for SBS and MBS,
    prints these summaries to the console, and visualizes the distributions
    using a boxplot overlaid with a swarmplot.

    Args:
        typeStr (str):
            Signal type identifier. Typically `"BR"` for breathing rate or
            `"HR"` for heart rate. This value is used to format plot titles
            and axis labels.
        stat_df (pandas.DataFrame):
            DataFrame containing subject-level statistics. It must include
            at least the following columns:
            - `'RangeBinSel'`: categorical variable with values `'SBS'` or `'MBS'`
            - `stat_name`: the metric to be analyzed
            - `'Patient ID'`: subject identifier (used for reporting extrema)
        stat_name (str):
            Name of the statistic/metric column in `stat_df` to analyze and plot
            (e.g., `'MAE'`, `'MAPE'`, `'CorrCoeff'`,
            `'Epochs_w_radarRate_ratio'`,
            `'Epochs_diff_<=X_ratio'`).
        boundOfUncertain (float, optional):
            Error bound used for labeling metrics related to low-error ratios
            (e.g., epochs with error ≤ bound). Default is 1.0.
        saveFig (bool, optional):
            If True, saves the generated figure to disk using a predefined
            filename pattern. Default is False.

    Returns:
        None
            The function does not return any value. It prints summary statistics
            to the console and displays (and optionally saves) a matplotlib figure.
    """

    title_by_stat = {
        'Epochs_w_radarRate_ratio': f'Recall',
        f'Epochs_diff_<={boundOfUncertain}_ratio': f'Low Error Ratio',
        'MAE': f'Mean Absolute Error',
        'MAPE': f'Mean Absolute Percent Error',
        'CorrCoeff': f"Spearman's Correlation Coefficient"
    }


    ylabel_by_stat = {
        'Epochs_w_radarRate_ratio': f'Epochs with radar {"breathing" if typeStr=="BR" else "heart"} rate [%]',
        f'Epochs_diff_<={boundOfUncertain}_ratio': f'Ratio of epochs with error <= {boundOfUncertain} {"breaths" if typeStr=="BR" else "beats"}/min',
        'MAE': f'Radar {"breathing" if typeStr=="BR" else "heart"} rate MAE [1/min]',
        'MAPE': f'Radar {"breathing" if typeStr=="BR" else "heart"} rate MAPE [%]',
        'CorrCoeff': f"Radar {"breathing" if typeStr=="BR" else "heart"} rate correlation coefficient"
    }

    sbs_stats = stat_df.loc[stat_df['RangeBinSel'] == 'SBS', stat_name]
    mbs_stats = stat_df.loc[stat_df['RangeBinSel'] == 'MBS', stat_name]

    sbs_min_idx = sbs_stats.idxmin()
    sbs_max_idx = sbs_stats.idxmax()

    mbs_min_idx = mbs_stats.idxmin()
    mbs_max_idx = mbs_stats.idxmax()

    print(f'{typeStr} SBS {stat_name} 10-25-50-75-90 percentiles: {[sbs_stats.quantile(q).round(2) for q in [.1,.25,.5,.75,.9]]}')
    print(f'{typeStr} SBS {stat_name} below 25 percentile: {sum(sbs_stats < sbs_stats.quantile(.25))} , above 75: {sum(sbs_stats > sbs_stats.quantile(.75))}')
    print(f'{typeStr} SBS {stat_name} min: {np.min(sbs_stats)} | max: {np.max(sbs_stats)}')
    print(f'{typeStr} SBS {stat_name} min: {sbs_stats.loc[sbs_min_idx]} pat: {stat_df.loc[sbs_min_idx, "Patient ID"]} | max: {sbs_stats.loc[sbs_max_idx]} pat: {stat_df.loc[sbs_max_idx, "Patient ID"]}')
    if stat_name == "Epochs_w_radarRate_ratio":
        print(f'{typeStr} SBS {stat_name} count >= 50%: {sum(sbs_stats >= 50)}')
    print(f'{typeStr} SBS {stat_name} count >= 10-25-50-75-90 percentiles: {[sum(sbs_stats >= sbs_stats.quantile(q).round(2)) for q in [.1,.25,.5,.75,.9]]}')

    print(f'{typeStr} MBS {stat_name} 10-25-50-75-90 percentiles: {[mbs_stats.quantile(q).round(2) for q in [.1,.25,.5,.75,.9]]}')
    print(f'{typeStr} MBS {stat_name} below 25 percentile: {sum(mbs_stats < mbs_stats.quantile(.25))} , above 75: {sum(mbs_stats > mbs_stats.quantile(.75))}')
    print(f'{typeStr} MBS {stat_name} min: {np.min(mbs_stats)} | max: {np.max(mbs_stats)}')
    print(f'{typeStr} MBS {stat_name} min: {mbs_stats.loc[mbs_min_idx]} pat: {stat_df.loc[mbs_min_idx, "Patient ID"]} | max: {mbs_stats.loc[mbs_max_idx]} pat: {stat_df.loc[mbs_max_idx, "Patient ID"]}')
    if stat_name == "Epochs_w_radarRate_ratio":
        print(f'{typeStr} MBS {stat_name} count >= 50%: {sum(mbs_stats >= 50)}')
    print(f'{typeStr} MBS {stat_name} count >= 10-25-50-75-90 percentiles: {[sum(mbs_stats >= mbs_stats.quantile(q).round(2)) for q in [.1,.25,.5,.75,.9]]}')

    plot_df = pd.concat([
        sbs_stats.rename("value").to_frame().assign(condition="Single range bin selection"),
        mbs_stats.rename("value").to_frame().assign(condition="Multiple range bin selection")
    ])


    _, ax = plt.subplots(figsize=(7, 5))

    sns.boxplot(
        data=plot_df,
        x="condition",
        y="value",
        hue="condition",
        palette=["#56B4E9", "#E69F00"],
        showfliers=False,   # avoid duplicate outliers
        width=0.75,
        legend=False,
        ax=ax
    )

    sns.swarmplot(
        data=plot_df,
        x="condition",
        y="value",
        color="0.25",
        size=5,
        alpha=0.85,
        ax=ax
    )

    ax.set_xlabel("")
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylabel(ylabel_by_stat[stat_name],fontsize=14)
    ax.set_title(title_by_stat[stat_name],fontsize=15)

    plt.tight_layout()

    if saveFig:
        ### -> put the path where the figures should be saved here
        figSavePath = ""
        plt.savefig(figSavePath + f"{typeStr}_SBSvsMBS_boxplot_{stat_name}.jpg",dpi=300,bbox_inches="tight")


    plt.show()

    return

filePath = hF.giveSaveFilePath()

binPreFilt = ""
binSel = ""
rateComp = ""
xtraTag = ""

recIDList = ["S0"+str(i) for i in range(27,75+1)]
recID_count = len(recIDList)

doClockBasedCut = True
doMotionParamThr = True
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
            if doClockBasedCut:
                nightClock = [22,0,0]
                nightClock_sec = hF.convertTimeStamp(nightClock)
                if recID != 'S048':
                    morningClock = [6+24,0,0] # add 24h because its next day
                elif recID == 'S048': # S048 terminated early 
                    morningClock = [3+24,30,0] # add 24h because its next day
                morningClock_sec = hF.convertTimeStamp(morningClock)

                nightClock_sec_afterRecStart = nightClock_sec - hF.convertTimeStamp(hF.perSubjStartClocks()[recID]) # recordings always started before 22:00
                morningClock_sec_afterRecStart = morningClock_sec - hF.convertTimeStamp(hF.perSubjStartClocks()[recID])

                inds2use = (timeStarts >= nightClock_sec_afterRecStart) & (timeStarts <= morningClock_sec_afterRecStart)
                bestBins = bestBins[:,inds2use]
                radarData = radarData[inds2use]
                psgData = psgData[inds2use]
                timeStarts = timeStarts[inds2use]
            
            # adding motion parameter based removal of epochs
            if doMotionParamThr:
                analysisResSaveFile = f"recID{recID}_motionParam.pkl"
                saveDirPath = hF.giveSaveFilePath() + "motionParam_win60s_step5s//"
                
                analysisResSaveFile = saveDirPath + analysisResSaveFile

                with open(analysisResSaveFile, 'rb') as fp:
                    motionParamSaveDict = pickle.load(fp)

                motionParam_timestamps = motionParamSaveDict['timeStarts']
                motionParam = motionParamSaveDict['motionParameter']

                if doClockBasedCut:
                    inds2use = (motionParam_timestamps >= nightClock_sec_afterRecStart) & (motionParam_timestamps <= morningClock_sec_afterRecStart)
                    motionParam_timestamps = motionParam_timestamps[inds2use]
                    motionParam = motionParam[inds2use]

                # compare the timestarts from the vital rate result, cut them to match if they dont
                if motionParam_timestamps[-1] != timeStarts[-1]:
                    if motionParam_timestamps[-1] > timeStarts[-1]:
                        motionParam = motionParam[motionParam_timestamps <= timeStarts[-1]]
                        motionParam_timestamps = motionParam_timestamps[motionParam_timestamps <= timeStarts[-1]]
                    elif motionParam_timestamps[-1] < timeStarts[-1]:
                        bestBins = bestBins[:,timeStarts <= motionParam_timestamps[-1]]
                        radarData = radarData[timeStarts <= motionParam_timestamps[-1]]
                        psgData = psgData[timeStarts <= motionParam_timestamps[-1]]
                        timeStarts = timeStarts[timeStarts <= motionParam_timestamps[-1]]

                # smoothing
                k = (5*60) // 5 # 5 minutes window with epoching of 60s win 5s step
                motionParam = signal.filtfilt(np.ones(k)/k,1, motionParam)

                motionParamThr = np.nanmean(motionParam)
                belowThr = motionParam < motionParamThr
                
                bestBins = bestBins[:,belowThr]
                radarData = radarData[belowThr]
                psgData = psgData[belowThr]
                timeStarts = timeStarts[belowThr]
            

            fullPsgData   = np.concatenate((fullPsgData,psgData))
            fullRadarData = np.concatenate((fullRadarData,radarData))

            perSubj_selBinSums[:,rInd] = np.sum(bestBins,axis=1)

            consecutiveEpochs = np.where(np.diff(timeStarts) < (timeStep+1))[0]
            consecutiveEpochsIntervals,consecutiveEpochsIntervalLens = hF.intervalExtractor(consecutiveEpochs)
            if len(consecutiveEpochsIntervalLens) > 0:
                consecutiveEpochsIntervals = consecutiveEpochs[consecutiveEpochsIntervals].astype("float")
                for i in range(len(consecutiveEpochsIntervalLens)-1):
                    if (consecutiveEpochsIntervals[i+1,0] - consecutiveEpochsIntervals[i,1]) < (timeWinLen / timeStep):
                        consecutiveEpochsIntervals[i+1,0] = copy.deepcopy(consecutiveEpochsIntervals[i,0])
                        consecutiveEpochsIntervals[i,:] = np.nan

                consecutiveEpochsIntervals = consecutiveEpochsIntervals[np.all(~np.isnan(consecutiveEpochsIntervals), axis=1),:]
                consecutiveEpochsIntervalLens = np.diff(consecutiveEpochsIntervals, axis=1)
                consecutiveEpochsIntervalSec = np.sum([timeWinLen + (leni-1)*timeStep for leni in consecutiveEpochsIntervalLens]) #np.sum(consecutiveEpochsIntervalLens) * timeStep
            

            currSubj_numWins            = len(timeStarts)

            currSubj_timeInSec          = consecutiveEpochsIntervalSec

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

            ## correlation coefficients
            # Remove NaN values pairwise
            mask = ~np.isnan(psgData) & ~np.isnan(radarData)
            x_clean = psgData[mask]
            y_clean = radarData[mask]

            # Check if data is sufficient
            if len(x_clean) < 3:
                currSubj_spearmanCorr = np.nan
            else:
                # Spearman correlation
                currSubj_spearmanCorr, _ = stats.spearmanr(x_clean, y_clean)

            perSubj_metric_list.append({
                'Patient ID': recID,
                'Epochs_w_psgRate_count': currSubj_winsWithPSG,
                'Epochs_w_psgRate_ratio': currSubj_winsWithPSG/currSubj_numWins,
                'Epochs_w_radarRate_count': currSubj_winsWithRadar,
                'Epochs_w_radarRate_ratio': currSubj_winsWithRadar/currSubj_numWins,
                f'Epochs_diff_<={boundOfUncertain}_count': currSubj_winsRealClose,
                f'Epochs_diff_<={boundOfUncertain}_ratio': currSubj_epochDiffs_realClose_ratio,
                'MAE': currSubj_mae,
                'MAPE': currSubj_mape,
                'CorrCoeff': currSubj_spearmanCorr
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
    
    for stat_name in ['Epochs_w_radarRate_ratio','MAE','MAPE','CorrCoeff']:
        plot_subjectLvl_stats_MBSvsSBS(typeStr, full_perSubj_metric_df, stat_name, boundOfUncertain, saveFig=saveFigs)

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

