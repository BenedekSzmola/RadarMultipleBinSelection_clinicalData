### import python libraries
import numpy as np
import time
import pickle

### import from other files of the project
import SBS_readRecordingData as readData
import helperFunctions as hF
import SBS_breathingDetection as bD
import SBS_heartDetection as hD
from radarSettings_IdA import radarSettings

### end of importing
############################################################################

def computeVitalRates(
        recID=[f"S{i:03}" for i in range(27,75+1)],
        breathOrHeartInput=["BR","HR"],
        useHannForRadarRange=True,
        radarChirps2use="all",
        chirpSumMeth="median",
        epochs_timeStart=0,
        epochs_timeEnd=np.inf,
        bins2check = np.arange(9,51),
        saveResults=True,
        saveNameExtra=""
    ):
    """
    Compute breathing or heart rates from radar data (and corresponding reference sensor) over fixed-length epochs,
    and optionally save results to a pickle file.

    Args:
        recID (str | Sequence[str]): Recording ID. Typically a single ID like "S027".
            If a sequence is provided, behavior depends on giveMeasFilePath; pass a single ID for a single run.
        breathOrHeartInput (str | list[str]): Which vital(s) to compute: "BR" (breathing), "HR" (heart), or both.
        useHannForRadarRange (bool): Apply Hann window to radar ADC samples before FFT.
        radarChirps2use (str | int | array-like): "all" to use all chirps; otherwise uses a single chirp (index 0).
        chirpSumMeth (str): Method to combine selected chirps: "mean" or "median".
        epochs_timeStart (float): Start time (s) for epoching window.
        epochs_timeEnd (float): End time (s) for epoching window (np.inf allowed to go to end).
        bins2check (array-like of int): Range-bin indices to evaluate for radar vital-rate extraction.
        saveResults (bool): If True, saves a .pkl with results.
        saveNameExtra (str): Optional suffix for the save-file name; a leading "_" is added if missing.

    Returns:
        None: Results are printed and, if saveResults is True, saved to a .pkl file in giveSaveFilePath().

    Side Effects:

        - Reads the measurement file for recID.
        - Epochs radar and reference data (epoch length fixed to 60 s; step size fixed to 5 s).

        - Prints progress and summary per epoch.
        - Saves a dictionary with keys:

          ['recID','vitalType','radarIdx','timeWinLen','epochStarts','timeStarts','epochInput','epochLen',
           'useHannForRadarRange','radarChirps2use','radarChirpSumMethod','bins2check','computeTimes','compute_times',
           'refSensor','bestBins','radarVitalRates','psgVitalRates','radarPos'].

    Notes:

        - Reference sensor selection differs for recID "S074" (PSG channel mapping differs).
        - Underlying I/O and processing errors (e.g., file not found, data format issues) may propagate as exceptions.

    """

    compute_times = {}
    overall_tic = time.perf_counter()

    if not isinstance(breathOrHeartInput, list):
        breathOrHeartInput = [breathOrHeartInput]
    
    if (len(saveNameExtra) > 0) and (saveNameExtra[0] != "_"):
        saveNameExtra = "_" + saveNameExtra

    # loading the save file
    filePath = hF.giveMeasFilePath(recID)
    ###############################################

    tic = time.perf_counter()
    # Load the measurement from its save file
    file_info,_,synchro_info,measurement_data = readData.readSaveFile(file_name=filePath)
    compute_times['load_measurement_data'] = time.perf_counter() - tic
    # Get the index and sampling rate for the radar
    radar_idx = hF.getSensorIdx(file_info['Measurement Data Format'],"Radar")
    radar_srate = synchro_info['Effective sampling frequency given by xdf.load() (Radar_1)']


    ### setup for the time interval to check ############################################################################
    epochInputDict = {
        'timeStart': epochs_timeStart,
        'timeEnd' : epochs_timeEnd,
        'epochStepSize': 5
    }

    epochLen = 60
    #####################################################################################################################

    ### Making epochs out of the full radar data, and settings for that #################################################
    
    # Which chirps to use, using all of them can lead to clearer signals
    if radarChirps2use == "all":
        radarChirps2use = np.arange(radarSettings['radar_loop_num'])
    else:
        radarChirps2use = 0

    print('Using these chirps: ', radarChirps2use)

    tic = time.perf_counter()
    timestampEpochs,phaseEpochs,magnitudeEpochs,epochStartTimestamps,timeStarts = readData.readRadarMakeEpochs(radarSettings,measurement_data,radar_idx,radar_srate,epochLen,epochInput=epochInputDict,useHann=useHannForRadarRange,chirp=radarChirps2use,chirpSumMethod=chirpSumMeth,doUnwrap=True,norm01=False,removeDC=True,parallelize=True)
    compute_times['epoching'] = time.perf_counter() - tic
    #####################################################################################################################

    compute_times['before_brORhr_loop'] = time.perf_counter() - overall_tic
    # if both breathing and heartbeat analysis is requested, do them here in a loop, the variables until here are shared for both analyses
    for breathOrHeart in breathOrHeartInput:
        brORhr_loop_tic = time.perf_counter()
        tic = time.perf_counter()
        ### Selecting what type of analysis to do (BR/HR) and then loading the correct reference sensors ####################
        if breathOrHeart == "BR":
            if recID == "S074": # This recording has a different PSG channel mapping; adjust sensor selection accordingly.
                refSensor_idx = hF.getSensorIdx(file_info['Measurement Data Format'],"Thorax")
            else:
                refSensor_idx = hF.getSensorIdx(file_info['Measurement Data Format'],"RIP Thorax")
            
        elif breathOrHeart == "HR":
            if recID == "S074":
                refSensor_idx = hF.getSensorIdx(file_info['Measurement Data Format'],"EKG2")
            else:
                refSensor_idx = hF.getSensorIdx(file_info['Measurement Data Format'],"ECG II")

        refSensor_srate = int(float(file_info['Measurement Data Sampling Rate'][refSensor_idx]))
        refTimestampsFull,refDataFull = readData.readFullRefData(measurement_data,refSensor_idx)
        refEpochInputDict = {'radarEpochStarts': epochStartTimestamps}
        refTimestampEpochs,refEpochs,refEpochStartTimestamps,refTimeStarts = readData.makeRefDataEpochs(refTimestampsFull,refDataFull,refSensor_srate,epochLen,refEpochInputDict,norm01=True)

        if len(timestampEpochs) != len(refTimestampEpochs):
            minEpochs = np.min((len(timestampEpochs),len(refTimestampEpochs)))

            timestampEpochs      = timestampEpochs[:minEpochs]
            phaseEpochs          = phaseEpochs[:minEpochs]
            magnitudeEpochs      = magnitudeEpochs[:minEpochs]
            epochStartTimestamps = epochStartTimestamps[:minEpochs]
            timeStarts           = timeStarts[:minEpochs]

            refTimestampEpochs      = refTimestampEpochs[:minEpochs]
            refEpochs               = refEpochs[:minEpochs]
            refEpochStartTimestamps = refEpochStartTimestamps[:minEpochs]
            refTimeStarts           = refTimeStarts[:minEpochs]

        compute_times['reference_loading+epoching'] = time.perf_counter() - tic

        ### initializing vectors for the results ############################################################################
        computeTimes = np.zeros(len(timestampEpochs))

        psgVitalRates   = np.full(len(timestampEpochs), np.nan)
        radarVitalRates = np.full(len(timestampEpochs), np.nan)

        bestVitalRateBins = np.full((len(bins2check),len(timestampEpochs)),np.nan)

        tic = time.perf_counter()
        ### Start going through the analysis windows #################################################################
        for epochi in range(len(timestampEpochs)):
            print('##############################################')
            print(f"### Starting analysis of window @ timestamp: {timeStarts[epochi]:.0f}")

            curr_radar_data = phaseEpochs[epochi]
            curr_radar_magData = magnitudeEpochs[epochi]
            curr_refSensor_data = refEpochs[epochi]

            # Breathing rate computation
            if breathOrHeart == "BR":

                st = time.process_time()
                outputDict = bD.checkBins(srate=radar_srate, data=curr_radar_data, bins2check=bins2check, magData=curr_radar_magData, binSelMethod="TPC", doBinPreFilt=True, doDiffTest=True)
                computeTimes[epochi] = time.process_time() - st

                refOutputDict = bD.checkBins(srate=refSensor_srate, data=curr_refSensor_data.reshape((1,-1)), bins2check=np.array([0]), binSelMethod=None, doDiffTest=False)

                bestVitalRateBins[:,epochi] = outputDict['bestBin']
                
                radarVitalRates[epochi]   = outputDict['respRate']
                psgVitalRates[epochi]     = refOutputDict['respRate']

                print('#########################################')
                print(f"Timestart@{timeStarts[epochi]}: bestbin={outputDict['bestBin']} | radar BR = {outputDict['respRate']: .4f} | PSG BR = {refOutputDict['respRate']: .4f}")
                print('#########################################')

            # Heart rate computation
            elif breathOrHeart == "HR":

                st = time.process_time()
                outputDict = hD.checkBins(srate=radar_srate, data=curr_radar_data, bins2check=bins2check, binSelMethod="TPC", doDiffTest=True)
                computeTimes[epochi] = time.process_time() - st

                refOutputDict = hD.checkBins(srate=refSensor_srate, data=curr_refSensor_data.reshape((1,-1)), bins2check=np.array([0]), binSelMethod=None, doDiffTest=False)
                
                bestVitalRateBins[:,epochi] = outputDict['bestBin']
                
                radarVitalRates[epochi]   = outputDict['heartRate']
                psgVitalRates[epochi]     = refOutputDict['heartRate']

                print('#########################################')
                print(f"Timestart@{timeStarts[epochi]}: bestbin={outputDict['bestBin']} | radar HR = {outputDict['heartRate']: .4f} | PSG HR = {refOutputDict['heartRate']: .4f}")
                print('#########################################')

        compute_times['main_VR_compute_loop'] = time.perf_counter() - tic
        if 'overall_runtime' in compute_times:
            compute_times['overall_runtime']  = compute_times['before_brORhr_loop'] + time.perf_counter() - brORhr_loop_tic
        else:
            compute_times['overall_runtime'] = time.perf_counter() - overall_tic
        #####################################################################################################################
        
        ### save the results as a dictionary to pkl file ####################################################################
        if saveResults:
            saveDict = {'recID': recID, 'vitalType': breathOrHeart, 'radarIdx': radar_idx,
                        'timeWinLen': epochLen, 'epochStarts': epochStartTimestamps, 'timeStarts': timeStarts, 'epochInput': epochInputDict, 'epochLen': epochLen,
                        'useHannForRadarRange': useHannForRadarRange, 'radarChirps2use': radarChirps2use, 'radarChirpSumMethod': chirpSumMeth,
                        'bins2check': bins2check, 'computeTimes': computeTimes, 'compute_times': compute_times, 'refSensor': file_info['Measurement Data Format'][refSensor_idx],
                        'bestBins': bestVitalRateBins, 'radarVitalRates': radarVitalRates, 'psgVitalRates': psgVitalRates}

            saveDict['radarPos'] = "Radar@FootEnd"
                
            analysisResSaveFile = f"recID{recID}_{breathOrHeart}_PSGvsRadar_SBS{saveNameExtra}.pkl"
            saveDirPath = hF.giveSaveFilePath()
                
            analysisResSaveFile = saveDirPath + analysisResSaveFile

            with open(analysisResSaveFile, 'wb') as fp:
                pickle.dump(saveDict, fp)


    return