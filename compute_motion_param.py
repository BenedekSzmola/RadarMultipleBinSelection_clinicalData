### module imports #################################################################################################
# import python libraries
import numpy as np
import pickle

# import from other files of the project
from radarSettings_IdA import radarSettings
import MBS_readRecordingData as readData
import helperFunctions as hF
#####################################################################################################################

#####################################################################################################################

#####################################################################################################################
def calc_radar_motion_parameter(radarSettings,epoch,range_start, range_end):
    """
    Take one epoch of doppler-range plots and calculate the motion parameter based on this. 
    Use only the space section defined by the range bin numbers in range_start and range_end.

    Args:
        epoch (python list): Collection of doppler-range plots for a whole epoch. 
            Each doppler-range plot consists of a range dimension and a velocity dimension.
        range_start (int): Index of range bin defining the start of the space 
            that should be summarized into the motion parameter.
        range_end (int): Index of range bin defining the end of the space 
            that should be summarized into the motion parameter.
    """
    velocity_weight = np.hamming(radarSettings['radar_loop_num'])  # Use Hamming Window because it does not completely exterminate the border values. Alternative: Hanning Window
    pre_parameter = [np.sum(doppler_range[range_start:range_end], axis=0) for doppler_range in epoch]  # Only taking values inside of the bed. 
    clutter = np.median(np.transpose(pre_parameter), axis = 1)  # one range-velocity bin over the epoch duration has a clutter. median to omit the influence of outlier
    pre_parameter = np.abs(pre_parameter - clutter)
    pre_parameter = [velocity_info*velocity_weight for velocity_info in pre_parameter]  # Weighing velocity values.
    motion_parameter = np.sum(pre_parameter) # add up velocity and time dimension

    return motion_parameter
#####################################################################################################################
#####################################################################################################################

"""
This script loops through every patient's data and computes the motion parameter as described in the manuscript:
    "Radar Multiple Bin Selection for Breathing and Heart Rate Monitoring in Acute Stroke Patients in a Clinical Setting"
"""

# Loop through patients
for recID in [f'S{id:03d}' for id in range(27,75+1)]:
    radarPos = 'Radar@FootEnd'

    filePath = hF.giveMeasFilePath(recID)

    # Load the measurement from its save file
    file_info,radar_var,synchro_info,measurement_data = readData.readSaveFile(file_name=filePath)

    radar_idx = hF.getSensorIdx(file_info['Measurement Data Format'],"Radar")
    radar_srate = synchro_info['Effective sampling frequency given by xdf.load() (Radar_1)']
    
    # Defining the interval to be segmented
    epochInputDict = {
        'timeStart': 0,
        'timeEnd' : np.inf,
        'epochStepSize': 5
    }           
    epochLen = 60

    # Segmenting the full recordings into epochs 
    useHannForRadarRange = True

    _,dopplerEpochs,epochStartTimestamps,timeStarts = readData.readRadarMakeDopplerEpochs(
        radarSettings,
        measurement_data,
        radar_idx,
        radar_srate,
        epochLen,
        epochInput=epochInputDict,
        useHann=useHannForRadarRange,
        parallelize=True)

    saveResults = True # should the results be saved at the end
    saveNameExtra = "" # extra text to append onto the savename
    if (len(saveNameExtra) > 0) and (saveNameExtra[0] != "_"):
        saveNameExtra = "_" + saveNameExtra

    # Selecting which range bins to work with
    bins2check = np.arange(9,51)

    # Initializing vector for the results
    motionParameter   = np.full(len(dopplerEpochs), np.nan)

    # Starting the loop through the epochs
    for epochi in range(len(dopplerEpochs)):
        print('##############################################')
        print(f"### Starting epoch {epochi} | start timestamp: {epochStartTimestamps[epochi]:.4f} | seconds from recording's beginning: {timeStarts[epochi]:.0f} ")

        ### Extract radar data
        currDoppler   = dopplerEpochs[epochi]

        motionParameter[epochi] = calc_radar_motion_parameter(radarSettings,currDoppler,bins2check[0],bins2check[-1])
        
    # Save the results as a dictionary to pkl file
    if saveResults:
        saveDict = {
            'recID': recID,
            'radarIdx': radar_idx,
            'radarPos': radarPos,
            'timeWinLen': epochLen,
            'epochStarts': epochStartTimestamps,
            'timeStarts': timeStarts,
            'epochInput': epochInputDict,
            'useHannForRadarRange': useHannForRadarRange,
            'bins2check': bins2check,
            'motionParameter': motionParameter
        }

        analysisResSaveFile = f"recID{recID}_motionParam{saveNameExtra}.pkl"
        saveDirPath = hF.giveSaveFilePath()
            
        analysisResSaveFile = saveDirPath + analysisResSaveFile

        with open(analysisResSaveFile, 'wb') as fp:
            pickle.dump(saveDict, fp)

#####################################################################################################################