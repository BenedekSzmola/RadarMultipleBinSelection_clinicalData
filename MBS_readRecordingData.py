
# Import External Packages

import copy

## load

import pickle  # to save the large measurement data

## calculation

import numpy as np
from joblib import Parallel, delayed

# Import internal packages

from helperFunctions import normDataTo_0_1
#####################################################

def readSaveFile(file_name):
    """
    Load synchronized measurement data from a pickle file containing repeated blocks.

    Args:
        file_name (str | os.PathLike): Path to a .pkl file containing repeated groups of four pickled objects:

            - file_info: Measurement metadata.
            - radar_var: Radar configuration (e.g., constants).

            - synchro_info: Synchronization information.
            - measurement_data: PSG and radar data.

    Returns:
        tuple:

            - file_info (Any): Measurement metadata from the last complete block read.
            - radar_var (Any): Radar configuration from the last complete block read.

            - synchro_info (Any): Synchronization info from the last complete block read.
            - measurement_data (Any): PSG and radar data from the last complete block read.

    Raises:
        FileNotFoundError: If the file cannot be opened.
        OSError: If there is an I/O error when opening the file.
        Note: Exceptions during reading inside the loop are caught and "EoF" is printed; the function returns the
        last successfully read block.
    """
    # Load Data
    with open(file_name, 'rb') as load_file:
        try:
            while True:
                file_info = (pickle.load(load_file))  # Information about measurement
                radar_var = (pickle.load(load_file))  # Configuration of radar (the same as constants.py)
                synchro_info = (pickle.load(load_file)) # Synchronsation Informationen
                measurement_data = (pickle.load(load_file)) # Measurement data of the PSG and all 3 Radars
                print('----------------File read----------------')
        except:
            print("EoF")

    return file_info,radar_var,synchro_info,measurement_data

def readRadarMakeEpochs(radarSettings,measurement_data,radar_idx,srate,epochLen,epochInput,useHann=True,chirp=np.array([0]),chirpSumMethod="mean",doUnwrap=True,norm01=False,removeDC=False,parallelize=False):
    """
    Convert raw radar IQ data into range-magnitude and range-phase time series, and epoch the data.

    Args:
        radarSettings (dict): Radar configuration. Required keys:

            - 'radar_loop_num'
            - 'radar_tx_num'

            - 'radar_rx_num'
            - 'radar_adcsamp_num'

            - 'speed_of_light'
            - 'radar_freq_min'

            - 'pi'

        measurement_data (Sequence): Container where measurement_data[radar_idx] = (timestamps, rawInputData).

            - timestamps (np.ndarray): Shape (N,), seconds.
            - rawInputData (np.ndarray): Interleaved int16 IQ per measurement; will be deinterleaved to complex.

        radar_idx (int): Index into measurement_data to select the radar stream.
        srate (float): Sampling rate in Hz (used for epoch sizing and filtering).
        epochLen (float): Epoch length in seconds.
        epochInput (dict): Epoch selection, one of:

            - {'timeStart': float, 'timeEnd': float, 'epochStepSize': float}
            - {'epochStarts': array-like of float} absolute timestamps.

        useHann (bool): Apply Hann window across ADC samples before FFT.
        chirp (int or array-like of int): Chirp indices to include/aggregate. Default [0].
        chirpSumMethod (str): How to combine selected chirps: 'mean' or 'median'.
        doUnwrap (bool): Unwrap phase (angle) before distance conversion.
        norm01 (bool): Normalize magnitude and phase to [0,1] per epoch using normDataTo_0_1.
        removeDC (bool): Remove per-range-bin DC from phase (subtract mean over time).
        parallelize (bool): Process epochs in parallel with joblib (fixed workers/block length in this script).

    Returns:
        tuple:

            - timestampEpochs (list[np.ndarray]): List of length numEpochs; each shape (epochLen_inSamples,).
            - phaseEpochs (list[np.ndarray]): List of length numEpochs; each shape (radar_adcsamp_num, epochLen_inSamples).

              Units: millimeters (distance from phase) after angle conversion.

            - magnitudeEpochs (list[np.ndarray]): List of length numEpochs; each shape (radar_adcsamp_num, epochLen_inSamples).
            - exactEpochStartTimestamps (np.ndarray): Shape (numEpochs,), first timestamp of each epoch (seconds).

            - epochStartOffsets (np.ndarray): Shape (numEpochs,), integer seconds from recording start:

              round(exactEpochStartTimestamps - timestamps[0]).

    Raises:
        Exception: If timeStart/timeEnd are invalid (order or too short for epochLen).
        Exception: If timeStart is inf or too close to end of recording.
        Exception: If any epochStarts are too close to the end to fit epochLen.
        Exception: If epochInput is missing required keys.

    Notes:

        - Deinterleaving converts interleaved int16 IQ into complex samples, reshaped to (frames, chirps, ADC samples).
        - If parallelize=True, joblib Parallel is used with fixed n_jobs=16 and block length ~3000 epochs.

    """
    def deinterleaveRadarData(rawInputData,TX,RX):
        """
        Deinterleave raw int16 IQ radar data and select TX/RX, returning complex range data.

        Args:
            rawInputData (np.ndarray): Raw interleaved IQ data for multiple measurements.
            TX (int): Transmitting antenna index to select.
            RX (int): Receiving antenna index to select.

        Returns:
            np.ndarray: Complex data with shape (num_meas, radar_loop_num, radar_adcsamp_num) after selecting TX/RX.
        """
        ## Deinterleaving the Radar Data
        num_meas = rawInputData.shape[0]  # number of measurements in data. each subarray is one measurement in an interleaved int 16 format
        data_raw = rawInputData.flatten()
        data_raw = np.reshape(data_raw, (int(len(data_raw)/4),2,2)) # creates a 2x2 subarray 
        data_raw = data_raw.transpose(0,2,1) #switches the places for data in the 2x2 array to revoke the interleaved structure
        data_raw = np.reshape(data_raw,(len(data_raw)*2,2)) # reshapes everything into size 2 subarrays which represent real and imaginary part of the complex number
        data_raw = data_raw.transpose() # transposes everything to get two large subarray. the first is every imaginary part and the seconds every real part
        data_raw = 1j*data_raw[0] + data_raw[1] # combining real and imaginary parts for everything at once => array with complex numbers along the time axis
        data_comp_all = np.reshape(data_raw, (num_meas,radarSettings['radar_loop_num'],radarSettings['radar_tx_num'],radarSettings['radar_rx_num'],radarSettings['radar_adcsamp_num'])) # reshaping complex numbers back to individual measurmements

        ## Converting chirps to range representations
        data_comp = data_comp_all[:,:,TX,RX]

        return data_comp
    
    ## Radar        
    TX = 0  # TX =  Transmitting Antenna -> We only dont use angle information here so we only use the same antenna configuration
    RX = 0  # RX =  Receiving Antenna

    if np.isscalar(chirp):
        chirp = np.array([chirp])
    
    epochLen_inSamples = int(np.floor(epochLen * srate))
    
    ## Extracting Radar data
    timestamps = measurement_data[radar_idx][0]
    rawInputData = measurement_data[radar_idx][1]

    fullSamplesLen = len(timestamps)
    measLenInSec = timestamps[-1] - timestamps[0]
    
    if 'timeStart' in epochInput:
        timeStart     = epochInput['timeStart']
        timeEnd       = epochInput['timeEnd']
        epochStepSize = epochInput['epochStepSize']

        epochStepSize_inSamples = int(np.floor(epochStepSize * srate))

        # Selecting the interval to be extracted (timeStart - timeEnd)
        if (timeStart > timeEnd) or ((timeEnd - timeStart) < epochLen):
            raise Exception("Inputs for timeStart and timeEnd are incorrect! (either timeStart is after timeEnd, or the difference between them is less then epochLen)")

        if np.isinf(timeStart) or (timeStart > (measLenInSec - epochLen)):
            raise Exception("The input for timeStart is incorrect! (either infinity or too close to end of recording)")

        timeStart_inSamples = np.argmin(np.abs(timestamps - (timeStart + timestamps[0])))
        if timeStart_inSamples < 0:
            print('Time start cannot be before the first sample! Setting to first sample')
            timeStart_inSamples = 0

        if np.isinf(timeEnd):
            timeEnd_inSamples = copy.deepcopy(fullSamplesLen)
        else:
            timeEnd_inSamples = np.argmin(np.abs(timestamps - (timeEnd + timestamps[0]))) + 1
            if timeEnd_inSamples > fullSamplesLen:
                print('Time end cannot be after the last sample! Setting to the last sample')
                timeEnd_inSamples = copy.deepcopy(fullSamplesLen)

        epochStartInds = np.arange(timeStart_inSamples, timeEnd_inSamples-epochLen_inSamples+1, epochStepSize_inSamples)

    elif 'epochStarts' in epochInput:
        epochStarts = epochInput['epochStarts']
        if np.isscalar(epochStarts):
            epochStarts = np.array([epochStarts])
        
        epochStartInds = np.zeros(len(epochStarts),dtype=int)

        for epochi in range(len(epochStarts)):
            epochStartInds[epochi] = np.argmin(np.abs(timestamps - epochStarts[epochi]))

        if any(epochStarts > (timestamps[-1] - epochLen)) and ((len(timestamps) - epochStartInds[-1]) < epochLen_inSamples):
            raise Exception("At least one of the epochStarts is incorrect! (is too close to the end of the recording)")

    
    else: 
        raise Exception("False input for parameter 'epochInputs'!")
  

    numEpochs = len(epochStartInds)

    timestampEpochs = [0]*numEpochs
    exactEpochStartTimestamps = np.zeros(numEpochs)
    magnitudeEpochs = [0]*numEpochs
    phaseEpochs     = [0]*numEpochs

    if not parallelize:
        for epochi,currEpochStart in enumerate(epochStartInds):
            currEpochEnd = currEpochStart + epochLen_inSamples

            timestampEpochs[epochi] = timestamps[currEpochStart:currEpochEnd]
            exactEpochStartTimestamps[epochi] = timestampEpochs[epochi][0]

            dataCompEpoch = deinterleaveRadarData(rawInputData[currEpochStart:currEpochEnd],TX,RX)

            if useHann:
                dataCompEpoch = dataCompEpoch * np.hanning(radarSettings['radar_adcsamp_num'])
            
            rangeDataEpoch = np.fft.fft(dataCompEpoch)

            tempMagnitude = np.zeros((len(chirp),radarSettings['radar_adcsamp_num'],len(timestampEpochs[epochi])))
            tempPhase     = np.zeros((len(chirp),radarSettings['radar_adcsamp_num'],len(timestampEpochs[epochi])))
            for chirpi,chirpNum in enumerate(chirp):
                tempMagnitude[chirpi,:,:] = np.abs(rangeDataEpoch[:,chirpNum,:].transpose())

                tempPhase[chirpi,:,:] = np.angle(rangeDataEpoch[:,chirpNum,:].transpose())
                if doUnwrap: 
                    tempPhase[chirpi,:,:] = np.unwrap(tempPhase[chirpi,:,:])

                tempPhase[chirpi,:,:] = (tempPhase[chirpi,:,:]* radarSettings['speed_of_light']) / (radarSettings['radar_freq_min'] * 4*radarSettings['pi'] )  * 1000  # Calculating actual distance and converting from meter into mm

                if removeDC:
                    tempPhase[chirpi,:,:] = tempPhase[chirpi,:,:] - np.mean(tempPhase[chirpi,:,:], axis=1, keepdims=True)

            if chirpSumMethod == "mean":
                magnitudeEpochs[epochi] = np.mean(tempMagnitude, axis=0)
                phaseEpochs[epochi]     = np.mean(tempPhase, axis=0)
                
            elif chirpSumMethod == "median":
                magnitudeEpochs[epochi] = np.median(tempMagnitude, axis=0)
                phaseEpochs[epochi]     = np.median(tempPhase, axis=0)

            if norm01:
                magnitudeEpochs[epochi] = normDataTo_0_1(magnitudeEpochs[epochi])
                
                phaseEpochs[epochi] = normDataTo_0_1(phaseEpochs[epochi])

    else:
        def prepare_epochs(currEpochStart):
            """
            Prepare one epoch: deinterleave, window, FFT, phase/magnitude extraction, optional normalization.

            Args:
                currEpochStart (int): Start index of the epoch (samples).

            Returns:
                tuple: (timestampEpoch, phaseEpoch, magnitudeEpoch, exactEpochStartTimestamp)
            """
            currEpochEnd = currEpochStart + epochLen_inSamples

            curr_timestampEpoch = timestamps[currEpochStart:currEpochEnd]
            curr_exactEpochStartTimestamp = curr_timestampEpoch[0]

            dataCompEpoch = deinterleaveRadarData(rawInputData[currEpochStart:currEpochEnd],TX,RX)

            if useHann:
                dataCompEpoch = dataCompEpoch * np.hanning(radarSettings['radar_adcsamp_num'])
            
            rangeDataEpoch = np.fft.fft(dataCompEpoch)

            tempMagnitude = np.zeros((len(chirp),radarSettings['radar_adcsamp_num'],len(curr_timestampEpoch)))
            tempPhase     = np.zeros((len(chirp),radarSettings['radar_adcsamp_num'],len(curr_timestampEpoch)))
            for chirpi,chirpNum in enumerate(chirp):
                tempMagnitude[chirpi,:,:] = np.abs(rangeDataEpoch[:,chirpNum,:].transpose())

                tempPhase[chirpi,:,:] = np.angle(rangeDataEpoch[:,chirpNum,:].transpose())
                if doUnwrap: 
                    tempPhase[chirpi,:,:] = np.unwrap(tempPhase[chirpi,:,:])

                tempPhase[chirpi,:,:] = (tempPhase[chirpi,:,:]* radarSettings['speed_of_light']) / (radarSettings['radar_freq_min'] * 4*radarSettings['pi'] )  * 1000  # Calculating actual distance and converting from meter into mm

                if removeDC:
                    tempPhase[chirpi,:,:] = tempPhase[chirpi,:,:] - np.mean(tempPhase[chirpi,:,:], axis=1, keepdims=True)

            if chirpSumMethod == "mean":
                curr_magnitudeEpoch = np.mean(tempMagnitude, axis=0)
                curr_phaseEpoch     = np.mean(tempPhase, axis=0)
                
            elif chirpSumMethod == "median":
                curr_magnitudeEpoch = np.median(tempMagnitude, axis=0)
                curr_phaseEpoch     = np.median(tempPhase, axis=0)

            if norm01:
                curr_magnitudeEpoch = normDataTo_0_1(curr_magnitudeEpoch)
                
                curr_phaseEpoch = normDataTo_0_1(curr_phaseEpoch)

            return curr_timestampEpoch, curr_phaseEpoch, curr_magnitudeEpoch, curr_exactEpochStartTimestamp
        
        # create blocks of epochs to parallelize:
        epoch_block_len = 3000
        for epochBlocki in range(0,len(epochStartInds),epoch_block_len):
            print('Starting new epoch block, starting epochind: ',epochBlocki)
            curr_epoch_inds = np.arange(epochBlocki, np.min((len(epochStartInds), epochBlocki+epoch_block_len)))
            parallel_output = Parallel(n_jobs=16, backend='loky', verbose=50)(
                delayed(prepare_epochs)(currEpochStart) for currEpochStart in epochStartInds[curr_epoch_inds]
            )
            for currInd,epochi in enumerate(curr_epoch_inds):
                timestampEpochs[epochi] = parallel_output[currInd][0]
                phaseEpochs[epochi] = parallel_output[currInd][1]
                magnitudeEpochs[epochi] = parallel_output[currInd][2]
                exactEpochStartTimestamps[epochi] = parallel_output[currInd][3]
        ####################################################################

    return timestampEpochs,phaseEpochs,magnitudeEpochs,exactEpochStartTimestamps,np.round(exactEpochStartTimestamps-timestamps[0]).astype("int")

def readRadarMakeDopplerEpochs(
    radarSettings,
    measurement_data,
    radar_idx,
    srate,
    epochLen,
    epochInput,
    useHann=True,
    parallelize=False,
    parallel_workers=16,
    parallel_blockLen=3000,
    parallel_verboseLvl=50
):
    """
    Read raw radar data and compute Doppler FFT spectra for time-aligned epochs.

    This function extracts radar measurements, segments them into epochs
    based on user-defined timing inputs, deinterleaves the raw ADC data into
    complex samples, and computes range–Doppler representations using FFTs.
    Epoch processing can be performed sequentially or in parallel for
    improved performance on large datasets.

    Args:
        radarSettings (dict):
            Dictionary containing radar configuration parameters. Expected
            keys include (but are not limited to):
            - 'radar_loop_num'
            - 'radar_tx_num'
            - 'radar_rx_num'
            - 'radar_adcsamp_num'
        measurement_data (list or tuple):
            Container holding measurement data for all sensors. The element
            at index `radar_idx` must be a tuple or list of the form
            `(timestamps, rawInputData)`, where `timestamps` is a 1D array of
            time values and `rawInputData` contains interleaved radar ADC data.
        radar_idx (int):
            Index specifying which radar measurement entry in
            `measurement_data` to process.
        srate (float):
            Sampling rate of the radar measurements in Hz.
        epochLen (float):
            Length of each epoch in seconds.
        epochInput (dict):
            Dictionary defining how epochs are constructed. Supported formats:
            - Time-based definition with keys:
              `'timeStart'`, `'timeEnd'`, `'epochStepSize'`
            - Explicit definition with key:
              `'epochStarts'` (scalar or array of start times in seconds)
        useHann (bool, optional):
            If True, applies a Hann window in the fast-time (range) and
            slow-time (Doppler) dimensions before FFT computation.
            Default is True.
        parallelize (bool, optional):
            If True, epochs are processed in parallel using joblib.
            Default is False.
        parallel_workers (int, optional):
            Number of worker processes to use when `parallelize` is True.
            Default is 16.
        parallel_blockLen (int, optional):
            Number of epochs processed per parallel block. This limits memory
            usage when processing long recordings. Default is 3000.
        parallel_verboseLvl (int, optional):
            Verbosity level passed to joblib's Parallel backend.
            Default is 50.

    Returns:
        tuple:
            A tuple containing:
            - timestampEpochs (list of ndarray):
                List of timestamp arrays, one per epoch.
            - dopplerEpochs (list of ndarray):
                List of Doppler power spectra (magnitude squared of FFT)
                computed for each epoch.
            - exactEpochStartTimestamps (ndarray):
                Array of exact start timestamps (in seconds) for each epoch.
            - epochStartSampleOffsets (ndarray):
                Integer array of epoch start offsets in samples, relative to
                the first radar timestamp.
    """
    def deinterleaveRadarData(rawInputData,TX,RX):
        ## Deinterleaving the Radar Data
        num_meas = rawInputData.shape[0]  # number of measurements in data. each subarray is one measurement in an interleaved int 16 format
        data_raw = rawInputData.flatten()
        data_raw = np.reshape(data_raw, (int(len(data_raw)/4),2,2)) # creates a 2x2 subarray 
        data_raw = data_raw.transpose(0,2,1) #switches the places for data in the 2x2 array to revoke the interleaved structure
        data_raw = np.reshape(data_raw,(len(data_raw)*2,2)) # reshapes everything into size 2 subarrays which represent real and imaginary part of the complex number
        data_raw = data_raw.transpose() # transposes everything to get two large subarray. the first is every imaginary part and the seconds every real part
        data_raw = 1j*data_raw[0] + data_raw[1] # combining real and imaginary parts for everything at once => array with complex numbers along the time axis
        data_comp_all = np.reshape(data_raw, (num_meas,radarSettings['radar_loop_num'],radarSettings['radar_tx_num'],radarSettings['radar_rx_num'],radarSettings['radar_adcsamp_num'])) # reshaping complex numbers back to individual measurmements

        ## Converting chirps to range representations
        data_comp = data_comp_all[:,:,TX,RX]

        return data_comp
    
    ## Radar        
    TX = 0  # TX =  Transmitting Antenna -> We only dont use angle information here so we only use the same antenna configuration
    RX = 0  # RX =  Receiving Antenna

    epochLen_inSamples = int(np.floor(epochLen * srate))
    
    ## Extracting Radar data
    timestamps = measurement_data[radar_idx][0]
    rawInputData = measurement_data[radar_idx][1]

    fullSamplesLen = len(timestamps)
    measLenInSec = timestamps[-1] - timestamps[0]
    
    if 'timeStart' in epochInput:
        timeStart     = epochInput['timeStart']
        timeEnd       = epochInput['timeEnd']
        epochStepSize = epochInput['epochStepSize']

        epochStepSize_inSamples = int(np.floor(epochStepSize * srate))

        if (timeStart > timeEnd) or ((timeEnd - timeStart) < epochLen):
            raise Exception("Inputs for timeStart and timeEnd are incorrect! (either timeStart is after timeEnd, or the difference between them is less then epochLen)")

        if np.isinf(timeStart) or (timeStart > (measLenInSec - epochLen)):
            raise Exception("The input for timeStart is incorrect! (either infinity or too close to end of recording)")

        timeStart_inSamples = np.argmin(np.abs(timestamps - (timeStart + timestamps[0])))
        if timeStart_inSamples < 0:
            print('Time start cannot be before the first sample! Setting to first sample')
            timeStart_inSamples = 0

        if np.isinf(timeEnd):
            timeEnd_inSamples = copy.deepcopy(fullSamplesLen)
        else:
            timeEnd_inSamples = np.argmin(np.abs(timestamps - (timeEnd + timestamps[0]))) + 1
            if timeEnd_inSamples > fullSamplesLen:
                print('Time end cannot be after the last sample! Setting to the last sample')
                timeEnd_inSamples = copy.deepcopy(fullSamplesLen)

        epochStartInds = np.arange(timeStart_inSamples, timeEnd_inSamples-epochLen_inSamples+1, epochStepSize_inSamples)

    elif 'epochStarts' in epochInput:
        epochStarts = epochInput['epochStarts']
        if np.isscalar(epochStarts):
            epochStarts = np.array([epochStarts])
        
        epochStartInds = np.zeros(len(epochStarts),dtype=int)

        for epochi in range(len(epochStarts)):
            epochStartInds[epochi] = np.argmin(np.abs(timestamps - epochStarts[epochi]))

        if any(epochStarts > (timestamps[-1] - epochLen)) and ((len(timestamps) - epochStartInds[-1]) < epochLen_inSamples):
            raise Exception("At least one of the epochStarts is incorrect! (is too close to the end of the recording)")
    
    else: 
        raise Exception("False input for parameter 'epochInputs'!")
  

    numEpochs = len(epochStartInds)

    timestampEpochs = [0]*numEpochs
    exactEpochStartTimestamps = np.zeros(numEpochs)
    dopplerEpochs = [0]*numEpochs

    if not parallelize:
        for epochi,currEpochStart in enumerate(epochStartInds):
            currEpochEnd = currEpochStart + epochLen_inSamples

            timestampEpochs[epochi] = timestamps[currEpochStart:currEpochEnd]
            exactEpochStartTimestamps[epochi] = timestampEpochs[epochi][0]

            dataCompEpoch = deinterleaveRadarData(rawInputData[currEpochStart:currEpochEnd],TX,RX)

            if useHann:
                dataCompEpoch = dataCompEpoch * np.hanning(radarSettings['radar_adcsamp_num'])
            
            rangeDataEpoch = np.fft.fft(dataCompEpoch)

            rangeDataEpoch = rangeDataEpoch.transpose(0,2,1)
            rangeDataEpoch = rangeDataEpoch * np.hanning(rangeDataEpoch.shape[-1])
            dopplerEpoch = np.fft.fft(rangeDataEpoch)
            dopplerEpoch = np.square(np.absolute(dopplerEpoch))  # Magnitude Calculation
            dopplerEpochs[epochi] = dopplerEpoch
    else:
        def process_epochs(currEpochStart):
            currEpochEnd = currEpochStart + epochLen_inSamples

            curr_timestampEpoch = timestamps[currEpochStart:currEpochEnd]
            curr_exactEpochStartTimestamp = curr_timestampEpoch[0]

            dataCompEpoch = deinterleaveRadarData(rawInputData[currEpochStart:currEpochEnd],TX,RX)

            if useHann:
                dataCompEpoch = dataCompEpoch * np.hanning(radarSettings['radar_adcsamp_num'])
            
            rangeDataEpoch = np.fft.fft(dataCompEpoch)

            rangeDataEpoch = rangeDataEpoch.transpose(0,2,1)
            rangeDataEpoch = rangeDataEpoch * np.hanning(rangeDataEpoch.shape[-1])
            curr_dopplerEpoch = np.fft.fft(rangeDataEpoch)
            curr_dopplerEpoch = np.square(np.absolute(curr_dopplerEpoch))  # Magnitude Calculation

            return curr_timestampEpoch, curr_dopplerEpoch, curr_exactEpochStartTimestamp

        # create blocks of epochs to parallelize:
        for epochBlocki in range(0,len(epochStartInds),parallel_blockLen):
            print('Starting new epoch block for doppler epochs, starting epochind: ',epochBlocki)
            curr_epoch_inds = np.arange(epochBlocki, np.min((len(epochStartInds), epochBlocki+parallel_blockLen)))

            parallel_output = Parallel(n_jobs=parallel_workers, backend='loky', verbose=parallel_verboseLvl)(
                delayed(process_epochs)(currEpochStart) for currEpochStart in epochStartInds[curr_epoch_inds]
            )
            for currInd,epochi in enumerate(curr_epoch_inds):
                timestampEpochs[epochi] = parallel_output[currInd][0]
                dopplerEpochs[epochi] = parallel_output[currInd][1]
                exactEpochStartTimestamps[epochi] = parallel_output[currInd][2]

    return timestampEpochs,dopplerEpochs,exactEpochStartTimestamps,np.round(exactEpochStartTimestamps-timestamps[0]).astype("int")

def readFullRefData(measurement_data,sensor_idx):
    """
    Read full reference sensor data for the specified sensor index.

    Args:
        measurement_data (Sequence): Container where measurement_data[sensor_idx] = (timestamps, data).

            - timestamps (np.ndarray): Shape (N,), in seconds (or sample times).
            - data (np.ndarray | Any): Sensor measurements aligned with timestamps.

        sensor_idx (int): Index of the reference sensor stream to read.

    Returns:
        tuple:

            - timestamps (np.ndarray): Shape (N,), timestamps for the selected sensor.
            - data (np.ndarray | Any): Sensor data aligned with timestamps.

    """
    ## Extracting reference sensor data
    timestamps = measurement_data[sensor_idx][0]    
    data = measurement_data[sensor_idx][1]    

    print('-----Successfully read full reference data-----')

    return timestamps, data


def makeRefDataEpochs(timestamps,data,srate,epochLen,epochInput,norm01=False):
    """
    Epoch reference sensor data aligned to timestamps.

    Args:
        timestamps (np.ndarray): Shape (N,), time vector in seconds (or sample times).
        data (np.ndarray | Any): Shape (N, ...) reference sensor data aligned with timestamps.
        srate (float): Sampling rate in Hz.
        epochLen (float): Epoch length in seconds; converted to samples via floor(epochLen * srate).
        epochInput (dict): Epoch selection, one of:

            - {'radarEpochStarts': array-like of float} start times to align with (clipped if beyond data).
            - {'timeStart': float, 'timeEnd': float, 'epochStepSize': float} range and step (seconds).

            - {'epochStarts': array-like of float} absolute start times (clipped if beyond data).

        norm01 (bool): If True, normalize each epoch to [0,1] using normDataTo_0_1.

    Returns:
        tuple:

            - timestampEpochs (list[np.ndarray]): List of length numEpochs; each shape (epochLen_inSamples,).
            - dataEpochs (list[np.ndarray]): List of length numEpochs; each shape (epochLen_inSamples, ...),

              optionally normalized.

            - exactEpochStartTimestamps (np.ndarray): Shape (numEpochs,), first timestamp of each epoch.
            - epochStartOffsets (np.ndarray): Shape (numEpochs,), integer seconds offset from recording start:

              round(exactEpochStartTimestamps - timestamps[0]).

    Raises:
        Exception: If epochInput does not contain one of the supported keys ('radarEpochStarts', 'timeStart',
        or 'epochStarts').

    Notes:

        - If requested starts extend beyond available data, indices are clipped and a message is printed.
        - For 'timeStart'/'timeEnd', timeEnd can be np.inf to select until the end of data.

    """
    epochLen_inSamples = int(np.floor(epochLen * srate))
    
    if 'radarEpochStarts' in epochInput:
        radarEpochStarts = epochInput['radarEpochStarts']
        epochStartInds = np.zeros(len(radarEpochStarts),dtype=int)

        for i in range(len(radarEpochStarts)):
            epochStartInds[i] = np.argmin(np.abs(timestamps - radarEpochStarts[i]))

        if any(radarEpochStarts > (timestamps[-1] - epochLen)) and ((len(timestamps) - epochStartInds[-1]) < epochLen_inSamples):
            epochStartInds = epochStartInds[epochStartInds <= (len(timestamps) - epochLen_inSamples)]
            print('The radarEpochStarts went beyond the actual reference timestamps, they were cut to fit!')

    elif 'timeStart' in epochInput:
        timeStart     = epochInput['timeStart']
        timeEnd       = epochInput['timeEnd']
        epochStepSize = epochInput['epochStepSize']

        epochStepSize_inSamples = int(np.floor(epochStepSize * srate))

        # Selecting the interval to be extracted (timeStart - timeEnd)
        timeStart_inSamples = np.argmin(np.abs(timestamps - (timeStart + timestamps[0])))
        if timeStart_inSamples < 0:
            print('Time start cannot be before the first sample! Setting to first sample')
            timeStart_inSamples = 0

        if np.isinf(timeEnd):
            timeEnd_inSamples = data.shape[0]
        else:
            timeEnd_inSamples = np.argmin(np.abs(timestamps - (timeEnd + timestamps[0]))) + 1
            if timeEnd_inSamples > data.shape[0]:
                print('Time end cannot be after the last sample! Setting to the last sample')
                timeEnd_inSamples = data.shape[0]

        epochStartInds = np.arange(timeStart_inSamples, timeEnd_inSamples-epochLen_inSamples+1, epochStepSize_inSamples)

    elif 'epochStarts' in epochInput:
        epochStarts = epochInput['epochStarts']
        if np.isscalar(epochStarts):
            epochStarts = np.array([epochStarts])
        
        epochStartInds = np.zeros(len(epochStarts),dtype=int)

        for epochi in range(len(epochStarts)):
            epochStartInds[epochi] = np.argmin(np.abs(timestamps - epochStarts[epochi]))

        # check if the input epochs dont go beyond the actual timestamps
        if any(epochStarts > (timestamps[-1] - epochLen)) and ((len(timestamps) - epochStartInds[-1]) < epochLen_inSamples):
            epochStartInds = epochStartInds[epochStartInds <= (len(timestamps) - epochLen_inSamples)]
            print('The input epochStarts went beyond the actual timestamps, they were cut to fit!')
    
    else: 
        raise Exception("False input for parameter 'epochInputs'!")

    numEpochs = len(epochStartInds)

    timestampEpochs = [0]*numEpochs
    exactEpochStartTimestamps = np.zeros(numEpochs)
    dataEpochs   = [0]*numEpochs
    
    for epochi,currEpochStart in enumerate(epochStartInds):
        currEpochEnd = currEpochStart + epochLen_inSamples

        timestampEpochs[epochi] = timestamps[currEpochStart:currEpochEnd]
        exactEpochStartTimestamps[epochi] = timestampEpochs[epochi][0]

        dataEpochs[epochi] = data[currEpochStart:currEpochEnd]

        if norm01:            
            dataEpochs[epochi] = normDataTo_0_1(dataEpochs[epochi])

    return timestampEpochs,dataEpochs,exactEpochStartTimestamps,np.round(exactEpochStartTimestamps-timestamps[0]).astype("int")