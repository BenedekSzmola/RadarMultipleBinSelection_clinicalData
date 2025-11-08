### module imports #################################################################################################
# import python libraries
import pickle
import numpy as np
import time
import copy
from joblib import Parallel, delayed

# import from other files of the project
import MBS_readRecordingData as readData
import MBS_breathing_classification_rateCompute as bD
import MBS_heartbeat_classification_rateCompute as hD
import helperFunctions as hF
from radarSettings_IdA import radarSettings
### end of importing
#####################################################################################################################


def run_multiBinSel_algorithms(
    recID=[f"S{i:03}" for i in range(27, 75 + 1)],
    breathOrHeartInput=["BR", "HR"],
    shortWin_timeStart=0,
    shortWin_timeEnd=np.inf,
    useHannForRadarRange=True,
    radarChirps2use="all",
    chirpSumMeth="median",
    bins2check=np.arange(9, 51),
    saveResults=True,
    saveNameExtra="",
):
    """
    Run multi-bin-selection pipelines to estimate breathing or heart rates from radar data,
    using short windows for bin classification and merged windows for rate computation, and optionally save results.

    Args:
        recID (str): Recording ID (e.g., "S070"). Although the default shows a list, this function expects a single ID.
        breathOrHeartInput (str | list[str]): Which vital(s) to compute: "BR" (breathing), "HR" (heart), or both.
        shortWin_timeStart (float): Start time in seconds for short-window epoching.
        shortWin_timeEnd (float): End time in seconds for short-window epoching (np.inf allowed).
        useHannForRadarRange (bool): Apply Hann window to radar ADC samples before FFT.
        radarChirps2use (str | int | array-like): "all" to use all chirps; otherwise a single chirp index (e.g., 0).
        chirpSumMeth (str): Method to combine selected chirps: "mean" or "median".
        bins2check (array-like of int): Range-bin indices to evaluate for radar vital-rate extraction.
        saveResults (bool): If True, saves a .pkl with results and compute-time breakdown.
        saveNameExtra (str): Optional suffix for the save-file name; a leading underscore is added if missing.

    Process:

        - Load measurement data and select radar + reference sensor(s).
        - Epoch radar into:

          - shortWin: 15 s windows (step=5 s) for per-bin classification (BR/HR-specific).
          - mergeWin: 60 s windows aligned to shortWin starts (except last 10) for rate computation.

        - Classify bins over shortWin epochs and enforce continuity (contiBestBins) over numContWin=10 with up to 3 gaps.
        - For each mergeWin epoch:

          - Compute reference rate (acorr for BR, SWT for HR).
          - Compute radar rate for bins that pass the continuity rule.

          - Aggregate per-epoch radar rates via mean and median.
        - Collect timing metrics and optionally save results.

    Returns:
        None

    Side Effects:

        - Prints progress and status messages.
        - If saveResults=True, saves a results dictionary to giveSaveFilePath() with keys:

          ['recID','breathOrHeart','radar_idx','bins2check','shortWin_epochInputDict','shortWin_epochLen',
           'shortWin_binClasses','numContWin','numAllowedGaps','mergeWin_epochLen','epochStarts','timeStarts',
           'useHannForRadarRange','radarChirps2use','radarChirpSumMethod','detMethod','refSensor','refMethod',
           'selectedBins','medianRadarVitalRates','meanRadarVitalRates','radarAllBinVitalRates','psgVitalRates',
           'compute_times'].

    Notes:

        - Reference sensor selection differs for recID "S074" due to a different PSG channel mapping.
        - mergeWin epochs are aligned to shortWin starts except the last 10 to ensure full 60 s coverage.

        - If radar and reference mergeWin epoch counts differ, both are truncated to the minimum.

    """
    compute_times = {}
    overall_tic = time.perf_counter()

    if not isinstance(breathOrHeartInput, list):
        breathOrHeartInput = [breathOrHeartInput]

    if (len(saveNameExtra) > 0) and (saveNameExtra[0] != "_"):
        saveNameExtra = "_" + saveNameExtra

    ### loading in the data, some parameters for that ###################################################################
    filePath = hF.giveMeasFilePath(recID)
    ###############################################

    tic = time.perf_counter()
    # Load the measurement from its save file
    file_info,_,synchro_info,measurement_data = readData.readSaveFile(file_name=filePath)
    compute_times['load_measurement_data'] = time.perf_counter() - tic

    radar_idx = hF.getSensorIdx(file_info['Measurement Data Format'],"Radar")
    radar_srate = synchro_info['Effective sampling frequency given by xdf.load() (Radar_1)']
    #####################################################################################################################

    # cleaning out measurement_data for memory
    sensor_idx_to_keep = [radar_idx]

    if "BR" in breathOrHeartInput:
        if recID == "S074": # This recording has a different PSG channel mapping; adjust sensor selection accordingly.
            brRef_idx = hF.getSensorIdx(file_info['Measurement Data Format'],"Thorax")
        else:
            brRef_idx = hF.getSensorIdx(file_info['Measurement Data Format'],"RIP Thorax")

        sensor_idx_to_keep.append(brRef_idx)

    if "HR" in breathOrHeartInput:
        if recID == "S074": # This recording has a different PSG channel mapping; adjust sensor selection accordingly.
            hrRef_idx = hF.getSensorIdx(file_info['Measurement Data Format'],"EKG2")
        else:
            hrRef_idx = hF.getSensorIdx(file_info['Measurement Data Format'],"ECG II")
        sensor_idx_to_keep.append(hrRef_idx)

    measurement_data = measurement_data[sensor_idx_to_keep]
    sensor_srates = file_info['Measurement Data Sampling Rate'][sensor_idx_to_keep]
    sensor_names = file_info['Measurement Data Format'][sensor_idx_to_keep]

    del file_info

    radar_idx = np.where(np.array(sensor_idx_to_keep) == radar_idx)[0][0]

    # getting the first timestamp, so the timeStarts variable doesnt need to be placed into memory until later for the plots
    radar_first_timestamp = measurement_data[radar_idx][0][0]

    ### setup for the time interval to check ############################################################################
    shortWin_epochInputDict = {
        'timeStart': shortWin_timeStart,
        'timeEnd' : shortWin_timeEnd,
        'epochStepSize': 5
    }

    shortWin_epochLen = 15
    #####################################################################################################################

    ### Making epochs out of the full radar data, and settings for that #################################################
    # Which chirps to use, using all of them can lead to clearer signals
    if radarChirps2use == "all":
        radarChirps2use = np.arange(radarSettings['radar_loop_num'])
    else:
        radarChirps2use = 0

    tic = time.perf_counter()
    _,shortWin_phaseEpochs,_,shortWin_epochStartTimestamps,_ = readData.readRadarMakeEpochs(radarSettings,measurement_data,radar_idx,radar_srate,shortWin_epochLen,epochInput=shortWin_epochInputDict,useHann=useHannForRadarRange,chirp=radarChirps2use,chirpSumMethod=chirpSumMeth,doUnwrap=True,norm01=False,removeDC=True,parallelize=True)

    mergeWin_epochInputDict = {
        'epochStarts': shortWin_epochStartTimestamps[:-10]
    }
    mergeWin_epochLen = 60
    _,mergeWin_phaseEpochs,_,_,_ = readData.readRadarMakeEpochs(radarSettings,measurement_data,radar_idx,radar_srate,mergeWin_epochLen,epochInput=mergeWin_epochInputDict,useHann=useHannForRadarRange,chirp=radarChirps2use,chirpSumMethod=chirpSumMeth,doUnwrap=True,norm01=False,removeDC=True,parallelize=True)

    compute_times['shortWin_epoching'] = time.perf_counter() - tic
    #####################################################################################################################

    compute_times['before_brORhr_loop'] = time.perf_counter() - overall_tic

    ### Selecting what type of analysis to do (BR/HR) and then loading the correct reference sensors ####################
    for breathOrHeart in breathOrHeartInput:
        brORhr_loop_tic = time.perf_counter()

        ### Selecting what type of analysis to do (BR/HR) and then loading the correct reference sensors ####################
        if breathOrHeart == "BR":
            refSensor_idx = np.where(np.array(sensor_idx_to_keep) == brRef_idx)[0][0]

        elif breathOrHeart == "HR":
            refSensor_idx = np.where(np.array(sensor_idx_to_keep) == hrRef_idx)[0][0]

        refSensor_srate = int(float(sensor_srates[refSensor_idx]))
        refTimestampsFull,refDataFull = readData.readFullRefData(measurement_data,refSensor_idx)

        #####################################################################################################################
        ### starting the loop through the selected time interval ############################################################
        tic = time.perf_counter()

        ### version which gives to the bD module the full epochs which parallelizes them piece by piece
        if breathOrHeart == "BR":
            binClasses = bD.checkBin_allEpochs(radar_srate, shortWin_phaseEpochs, shortWin_epochLen, bins2check)
        elif breathOrHeart == "HR":
            binClasses = hD.checkBin_allEpochs(radar_srate, shortWin_phaseEpochs, shortWin_epochLen, bins2check)
        #############################################################


        compute_times['shortWin_binClassification'] = time.perf_counter() - tic

        print('\n###### Done with short win bin classification ########\n')
        if (len(breathOrHeartInput) == 1) or (np.where(np.array(breathOrHeartInput) == breathOrHeart)[0][0] == 1):
            del shortWin_phaseEpochs
        #####################################################################################################################
        #####################################################################################################################

        numContWin = 10
        numAllowedGaps = 3

        if breathOrHeart == "BR":
            detMethod = 'acorr'
            refDetMethod = 'acorr'
        elif breathOrHeart == "HR":
            detMethod = 'normPeaks'
            refDetMethod = "swt"

        #####################################################################################################################

        contiBestBins = copy.deepcopy(binClasses)

        temp = copy.deepcopy(binClasses)

        for bini in range(contiBestBins.shape[0]):
            for epochi in range(contiBestBins.shape[1]):
                if np.sum(contiBestBins[bini,epochi:epochi+numContWin] > 0) >= (numContWin-numAllowedGaps):
                    temp[bini,epochi:epochi+numContWin] = 10

            temp[bini,temp[bini,:] < 10] = 0

        contiBestBins = copy.deepcopy(temp)
        #####################################################################################################################

        #
        vitalRateFromContWins = np.full((len(bins2check),len(shortWin_epochStartTimestamps)),np.nan)
        vitalRateBinAvg = np.full(len(shortWin_epochStartTimestamps),np.nan)
        vitalRateBinMedian = np.full(len(shortWin_epochStartTimestamps),np.nan)
        vitalRateRef = np.full(len(shortWin_epochStartTimestamps),np.nan)

        tic = time.perf_counter()

        ### another parallelization mode, this stores first which bins in which epochs should be used, and then does the actual vital rate comp ###########
        def determine_which_bins(epochi):
            curr_epoch_bins2compute = np.array([np.sum(binClasses[bini, epochi:epochi + numContWin] > 0) >= (numContWin - numAllowedGaps) for bini in range(len(bins2check))]) # precompute from which bins should the vital rate be computed this epoch
            return bins2check[curr_epoch_bins2compute] # apply the above mask to the bin numbers

        perEpoch_bins2compute = Parallel(n_jobs=24, backend='loky', verbose=50)(
            delayed(determine_which_bins)(epochi) for epochi in range(len(shortWin_epochStartTimestamps))
        )

        mergeWin_refEpochInputDict = {'radarEpochStarts': shortWin_epochStartTimestamps[:-10]}
        _,mergeWin_refEpochs,_,_ = readData.makeRefDataEpochs(refTimestampsFull,refDataFull,refSensor_srate,mergeWin_epochLen,mergeWin_refEpochInputDict,norm01=True)

        # some recordings (like S047) have shorter reference data, probably because of the missing end synch
        # so here check that the radar and reference epochs are of the same length
        if len(mergeWin_phaseEpochs) != len(mergeWin_refEpochs):
            radar_ref_min_epochNum = np.min((len(mergeWin_phaseEpochs), len(mergeWin_refEpochs)))

            mergeWin_phaseEpochs = mergeWin_phaseEpochs[:radar_ref_min_epochNum]
            mergeWin_refEpochs = mergeWin_refEpochs[:radar_ref_min_epochNum]

            print('\n----------------------------------------------')
            print('The number of epochs for radar and reference didnt match, they were cut to fit together!')
            print('----------------------------------------------\n')

        for epochi in range(len(mergeWin_phaseEpochs)):
            print('-----------------------------------------')
            print(f'Currently on epoch {epochi}/{len(mergeWin_phaseEpochs)}')

            currPhase = mergeWin_phaseEpochs[epochi]
            currRef = mergeWin_refEpochs[epochi]

            if breathOrHeart == "BR":
                refReturDict = bD.computeBR_acorr(refSensor_srate, currRef)

                vitalRateRef[epochi] = refReturDict['breathRate']

            elif breathOrHeart == "HR":
                refReturDict = hD.computeHR_swt(refSensor_srate, currRef)

                vitalRateRef[epochi] = refReturDict['heartRate']

            for binNum in perEpoch_bins2compute[epochi]:
                bini = np.where(bins2check == binNum)[0][0]
                if breathOrHeart == "BR":
                    returDict = bD.computeBR_acorr(radar_srate, currPhase[binNum,:])

                    vitalRateFromContWins[bini,epochi] = returDict['breathRate']

                elif breathOrHeart == "HR":
                    returDict = hD.computeHR_normPeaks(radar_srate, currPhase[binNum,:])

                    vitalRateFromContWins[bini,epochi] = returDict['heartRate']

        ###############################################################################################################################


        # Compute the mean and median for each epoch after epoch loop is done
        epochs_with_vitalRate = np.where(np.any(~np.isnan(vitalRateFromContWins), axis=0))[0]
        if len(epochs_with_vitalRate) > 0:
            vitalRateBinAvg[epochs_with_vitalRate] = np.nanmean(vitalRateFromContWins[:,epochs_with_vitalRate], axis=0)
            vitalRateBinMedian[epochs_with_vitalRate] = np.nanmedian(vitalRateFromContWins[:,epochs_with_vitalRate], axis=0)

        ### end of parallelizing epochs of vital rate computation #############################################################

        compute_times['mergeWin_rateCompute'] = time.perf_counter() - tic
        #####################################################################################################################
        if 'overall_runtime' in compute_times:
            compute_times['overall_runtime']  = compute_times['before_brORhr_loop'] + time.perf_counter() - brORhr_loop_tic
        else:
            compute_times['overall_runtime'] = time.perf_counter() - overall_tic

        shortWin_timeStarts = np.array([np.round(shortWin_epochStartTimestamps[epochi] - radar_first_timestamp).astype("int") for epochi in range(len(shortWin_epochStartTimestamps))])

        ### save the results as a dictionary to pkl file ####################################################################
        if saveResults:
            saveDict = {'recID': recID, 'breathOrHeart': breathOrHeart, 'radar_idx': sensor_idx_to_keep[radar_idx],
                        'bins2check': bins2check, 'shortWin_epochInputDict': shortWin_epochInputDict, 'shortWin_epochLen': shortWin_epochLen, 'shortWin_binClasses': binClasses,
                        'numContWin': numContWin, 'numAllowedGaps': numAllowedGaps, 'mergeWin_epochLen': mergeWin_epochLen,
                        'epochStarts': shortWin_epochStartTimestamps, 'timeStarts': shortWin_timeStarts, 'useHannForRadarRange': useHannForRadarRange, 'radarChirps2use': radarChirps2use,
                        'radarChirpSumMethod': chirpSumMeth, 'detMethod': detMethod, 'refSensor': sensor_names[refSensor_idx], 'refMethod': refDetMethod,
                        'selectedBins': contiBestBins, 'medianRadarVitalRates': vitalRateBinMedian, 'meanRadarVitalRates': vitalRateBinAvg,
                        'radarAllBinVitalRates': vitalRateFromContWins, 'psgVitalRates': vitalRateRef, 'compute_times': compute_times}

            analysisResSaveFile = f"subj{recID}_{breathOrHeart}_MBS{saveNameExtra}.pkl"
            saveDirPath = hF.giveSaveFilePath()

            analysisResSaveFile = saveDirPath + analysisResSaveFile

            with open(analysisResSaveFile, 'wb') as fp:
                pickle.dump(saveDict, fp)
        #####################################################################################################################
    return