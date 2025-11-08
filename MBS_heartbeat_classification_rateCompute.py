########################################################################
###### import python packages ##########################################
########################################################################
import numpy as np
from scipy import signal
import copy
import pywt
from joblib import Parallel, delayed
########################################################################

########################################################################
### import methods from other scripts ##################################
########################################################################
import helperFunctions as hF
from MBS_persDiagFeatureMiner import subLevelFiltPersDgmChecker_DBSCAN_HR
########################################################################

#########################################################################
### complete epochs checkBins
#########################################################################
def checkBin_allEpochs(srate, dataEpochs, epochLen, bins2checkFull):
    """
    Classify bins across multiple epochs for heartbeat detection.

    Args:
        srate (float): Sampling rate in Hz.
        dataEpochs (list[np.ndarray]): List of 2D arrays per epoch, each shape (n_bins, n_samples).
        epochLen (float): Window length in seconds (used in classification thresholds).
        bins2checkFull (array-like of int): All candidate bin indices to evaluate.

    Returns:
        np.ndarray: binClasses of shape (len(bins2check), numEpochs), with values
            0=failure, 1=possibly flawed, 2=good.

    Notes:

        - Each epoch is prefiltered (bandpass 0.65–5 Hz, ~0.15 s smoothing) and normalized.
        - Classification per bin uses subLevelFiltPersDgmChecker_DBSCAN_HR.

        - Epochs are processed in chunks with parallelization across epochs and bins.

    """
    bins2check = copy.deepcopy(bins2checkFull)

    numEpochs = len(dataEpochs)

    def prepare_filtData(epochi):
        b,a = signal.butter(N=2, Wn=[.65, 5], fs=srate, btype="bandpass")
        data2use = signal.filtfilt(b,a,dataEpochs[epochi])
        
        k = np.round(.15*srate).astype("int")
        data2use = signal.filtfilt(np.ones(k)/k,1,data2use)

        return hF.normDataTo_0_1(data2use)
    
    data2use_epochs = []
    for epochi in range(numEpochs):
        print(f'Preparing filt data, epoch {epochi} / {numEpochs}')
        data2use_epochs.append(prepare_filtData(epochi))

    # Parallelized bin handling
    def process_bin(curr_data2use):
        subLvlReturDict = subLevelFiltPersDgmChecker_DBSCAN_HR(curr_data2use,epochLen,showLogs=False)
        return subLvlReturDict['class']
    
    def compute_epoch_class(curr_data2use_epoch):
        curr_epoch_binCats = Parallel(n_jobs=3, backend='loky')(
            delayed(process_bin)(curr_data2use_epoch[binNum,:]) for binNum in bins2check
        )

        return curr_epoch_binCats

    binClasses = np.full((len(bins2check), numEpochs), np.nan)

    epoch_chunk_len = 1000
    for curr_chunk_start in range(0, numEpochs, epoch_chunk_len):
        print('Starting new epoch chunk for computing the diagrams and then classifying, starting epochind: ',curr_chunk_start)
        curr_chunk_inds = np.arange(curr_chunk_start, np.min((numEpochs, curr_chunk_start+epoch_chunk_len)))

        curr_chunk_binCats = Parallel(n_jobs=8, backend='loky', verbose=50)(
            delayed(compute_epoch_class)(data2use_epochs[epochi]) for epochi in curr_chunk_inds
        )

        for i,epochi in enumerate(curr_chunk_inds):
            binClasses[:,epochi] = curr_chunk_binCats[i]

    return binClasses

##############################################################################################################################################################################
##############################################################################################################################################################################

##################################################################################################
### The heart rate detection algorithms ######################################################
##################################################################################################


#########################################################################
### method swt: uses swt
#########################################################################
def computeHR_swt(srate, data):
    """
    Estimate heart rate using Stationary Wavelet Transform and peak interval analysis.

    Args:
        srate (float): Sampling rate in Hz.
        data (np.ndarray): 1D array (n_samples,) of the time series.

    Returns:
        dict: {'critVal': float, 'heartRate': float}

            - critVal: Quality metric (1 / std(IBI)) if valid, else np.nan.
            - heartRate: BPM if valid, else np.nan.

    Method:

        - SWT ("sym4") to extract cardiac component; Hilbert envelope; ~0.3 s smoothing.
        - Peak detection with prominence tied to std and minimum spacing ~150 BPM.

        - Replace contiguous outlier intervals by local median; validate IBI stability and physiologic bounds (40–150 BPM).
        - Compute heartRate = 60 / mean(IBI) and critVal = 1 / std(IBI).

    """
    returDict = {'critVal': np.nan, 'heartRate': np.nan}

    data4det = copy.deepcopy(data)

    ###########################################
    ### compute swt
    ###########################################
    
    swtLvl = np.ceil(np.log2((srate/2) / 3.5))
    if len(data4det) % (2**swtLvl) != 0:
        necessaryLen = (2**swtLvl) * np.ceil(len(data4det) / (2**swtLvl))
        startPad = int(np.floor((necessaryLen-len(data4det))/2))
        endPad = int(np.ceil((necessaryLen-len(data4det))/2))
        dataPad = np.pad(data4det, (startPad, endPad))

    else:
        dataPad = copy.deepcopy(data4det)
        startPad = 0
        endPad = 0

    coeffs = pywt.swt(dataPad,wavelet="sym4",level=swtLvl,trim_approx=True)

    #############################
    ### Extract and square the wanted coefficient
    #############################

    COI = coeffs[1] # coeff of interest
    COI = COI[startPad:len(COI)-endPad]
    dataSquare = np.abs(signal.hilbert(COI*-1))
    
    k = np.round(srate*.3).astype("int")
    dataSquare = signal.filtfilt(np.ones(k)/k,1,dataSquare)

    #############################
    ### Compute peak2peak intervals
    #############################

    peakInds = signal.find_peaks(dataSquare, prominence=1*np.std(dataSquare,ddof=1), distance=(60/150)*srate)[0]
    
    if any(peakInds) and (len(peakInds) >= 3):
        peakIntervals = np.diff(peakInds) / srate

        outUpThr  = np.mean(peakIntervals) + 1*np.std(peakIntervals,ddof=1)
        outLowThr = np.mean(peakIntervals) - 1*np.std(peakIntervals,ddof=1)
        outlierInds = np.where((peakIntervals > outUpThr) | (peakIntervals < outLowThr))[0]
        outlierIvs = outlierInds[hF.intervalExtractor(outlierInds)[0]]

        for outi in range(outlierIvs.shape[0]):
            starti = outlierIvs[outi,0]
            endi = outlierIvs[outi,1]

            if starti == 0:
                preWin = np.array([])

            else:
                preWin = peakIntervals[np.max((0,starti-2)):starti]

            if endi == (len(peakIntervals)-1):
                postWin = np.array([])

            else:
                postWin = peakIntervals[endi:np.min((len(peakIntervals),endi+3))]

            surrWin = np.concatenate((preWin,postWin))
            if len(surrWin) > 0:
                peakIntervals[starti:endi+1] = np.median(surrWin)

        #########################################
        ### Test and heart activity
        #########################################
        medPeakInterval = np.mean(peakIntervals)

        std2medianRatio = np.std(peakIntervals, ddof=1) / medPeakInterval
        badIntervalsCount = np.sum( (peakIntervals < (60/150)) | (peakIntervals > (60/40)) )

        
        if (std2medianRatio < .2) & (badIntervalsCount/len(peakIntervals) < .1) & (medPeakInterval >= (60/150)) & (medPeakInterval <= (60/40)):
            critVal = 1/np.std(peakIntervals,ddof=1)
            heartRate = 60 / medPeakInterval # transform to /min

            returDict["critVal"] = critVal
            returDict["heartRate"] = heartRate

    return returDict

#########################################################################
### method normPeaks: simply find the peaks in the normed data
#########################################################################
def computeHR_normPeaks(srate, data):
    """
    Estimate heart rate from normalized data via peak detection and IBI validation.

    Args:
        srate (float): Sampling rate in Hz.
        data (np.ndarray): 1D array (n_samples,).

    Returns:
        dict: {'critVal': float, 'heartRate': float}

            - critVal: 1 / std(IBI) if valid, else np.nan.
            - heartRate: BPM if valid, else np.nan.

    Method:

        - Bandpass filter (0.8–1.7 Hz), normalize via analytic amplitude (Hilbert).
        - Peak detection with min height=0.9 and min spacing ~150 BPM.

        - Replace contiguous outlier intervals by local median; reject if >25% IBIs are outliers.
        - Validate IBI stability and physiologic bounds (40–150 BPM); compute HR and critVal.

    """
    returDict = {'critVal': np.nan, 'heartRate': np.nan}

    b,a = signal.butter(N=2, Wn=[.8,1.7], btype="bandpass", fs=srate)
    dataFilt = signal.filtfilt(b,a,data)
    
    amps = np.abs(dataFilt + 1j*signal.hilbert(dataFilt))
    data4det = dataFilt / amps


    peakInds = signal.find_peaks(data4det, height=.9, distance=srate*60/150)[0]
    
    if not( (not any(peakInds)) or (len(peakInds) < 5) or ( np.std( np.diff(peakInds), ddof=1 )/srate > .3 ) ):
        peakIntervals = np.diff(peakInds) / srate
       
        outUpThr  = np.mean(peakIntervals) + 1*np.std(peakIntervals,ddof=1)
        outLowThr = np.mean(peakIntervals) - 1*np.std(peakIntervals,ddof=1)
        outlierInds = np.where((peakIntervals > outUpThr) | (peakIntervals < outLowThr))[0]

        if len(outlierInds)/len(peakIntervals) <= .25:
            outlierIvs = outlierInds[hF.intervalExtractor(outlierInds)[0]]

            for outi in range(outlierIvs.shape[0]):
                starti = outlierIvs[outi,0]
                endi = outlierIvs[outi,1]

                if starti == 0:
                    preWin = np.array([])

                else:
                    preWin = peakIntervals[np.max((0,starti-2)):starti]

                if endi == (len(peakIntervals)-1):
                    postWin = np.array([])

                else:
                    postWin = peakIntervals[endi:np.min((len(peakIntervals),endi+3))]

                surrWin = np.concatenate((preWin,postWin))
                if len(surrWin) > 0:
                    peakIntervals[starti:endi+1] = np.median(surrWin)
            
            #########################################
            ### Test for heart activity
            #########################################

            avgPeakInterval = np.mean(peakIntervals)

            IBIstd = np.std(peakIntervals, ddof=1)
            badIntervalsCount = np.sum( (peakIntervals < (60/150)) | (peakIntervals > (60/40)) )

            if (IBIstd < .1) & (badIntervalsCount/len(peakIntervals) < .1) & (avgPeakInterval >= (60/150)) & (avgPeakInterval <= (60/40)):

                critVal   = 1 / np.std(peakIntervals,ddof=1)
                heartRate = 60 / avgPeakInterval # transform to /min

                returDict["critVal"]   = critVal
                returDict["heartRate"] = heartRate

    return returDict