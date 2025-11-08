import numpy as np
from scipy import signal
import copy
import pywt

from helperFunctions import zeroCrossDetector, intervalExtractor

def checkBins(srate, data, bins2check, binSelMethod="TPC", doDiffTest=True):
    """
    Select the best range bin(s) and estimate heart rate.

    Args:
        srate (float): Sampling rate in Hz.
        data (np.ndarray): 2D array (n_bins, n_samples) of phase (or similar) signals. For reference/ECG, pass shape (1, n_samples).
        bins2check (array-like of int): Candidate range-bin indices to evaluate.
        binSelMethod (str | None): Bin selection method:

            - "TPC": Temporal Phase Coherency to rank bins, then estimate HR from top bins (default).
            - None: Treat input as single-channel reference data and estimate HR directly (SWT method).

        doDiffTest (bool): If True, discard bins whose max sample-to-sample difference exceeds 1 (artifact screen).

    Returns:
        dict: Keys depend on mode:

            - For TPC mode:
                - "bestBin" (int | float): Selected best bin index (np.nan if none passes).

                - "heartRate" (float): Estimated heart rate in BPM (np.nan if not available).
            - For reference mode (binSelMethod is None):

                - "heartRate" (float): Estimated heart rate in BPM (np.nan if not available).

    Notes:

        - TPC mode:
            - Bandpass filter to 0.8–1.7 Hz.

            - Normalize by instantaneous amplitude (Hilbert).
            - Compute TPC via zero-crossing-derived lag T.

            - Evaluate top-N bins via computeHR_normPeaks; choose the bin with max criterion.
        - Reference mode uses computeHR_swt on data[0, :].

    """
    outputDict = {}

    # Filtering out unexpectedly large differences in the phase data
    if doDiffTest:
        bins2check = bins2check[np.where(np.max(np.abs(np.diff(data[bins2check,:], axis=1)), axis=1) < 1)[0]]

        if len(bins2check) == 0:
            outputDict["bestBin"]  = np.nan
            outputDict["heartRate"] = np.nan

            return outputDict

    # For the reference data where there are no bins
    if binSelMethod is None:
        _,heartRate = computeHR_swt(srate, data[0,:])
        outputDict["heartRate"] = heartRate

    # Temporal Phase Coherency method to select the 5 best radar range bins
    elif binSelMethod == "TPC":
        # Bandpass filtering the input data to the expected heartbeat frequency range
        b,a = signal.butter(N=2,Wn=[.8,1.7], btype='bandpass', fs=srate)
        dataFilt = signal.filtfilt(b,a,data)

        TPCarray = np.zeros(len(bins2check))
        dataNorm = np.zeros_like(data)

        # Looping through the range bins and computing TPC
        for i,binNum in enumerate(bins2check):
            amps = np.abs(dataFilt[binNum,:] + 1j*signal.hilbert(dataFilt[binNum,:]))
            dataNorm[binNum,:] = dataFilt[binNum,:] / amps

            numZeroCross,zeroCrossInds = zeroCrossDetector(dataNorm[binNum,:])

            Tval = int(2*np.sum(np.diff(zeroCrossInds)) / (numZeroCross-1))

            TPCarray[i] = np.sum(dataNorm[binNum,0:-Tval]*dataNorm[binNum,Tval:]) / (np.std(dataNorm[binNum,0:-Tval],ddof=1) * np.std(dataNorm[binNum,Tval:],ddof=1))

        # Taking absolute value of TPC values
        TPCarray = np.abs(TPCarray)

        # Selecting the top 5 bins based on TPC value
        topN = np.min((5,len(TPCarray)))
        bestBins = bins2check[np.argpartition(TPCarray,-topN)[-topN:]]

        # Selecting the data which will be used for the HR computation
        data2use = dataNorm[bestBins,:]

        critVals   = np.full(len(bestBins), np.nan)
        heartRates = np.full(len(bestBins), np.nan)
        for i in range(len(bestBins)):
            print(f"Currently computing HR for bin #{bestBins[i]}")
            critVals[i],heartRates[i] = computeHR_normPeaks(srate, data2use[i,:])

        if np.all(np.isnan(heartRates)) or np.all(np.isnan(critVals)):
            outputDict["bestBin"]  = np.nan
            outputDict["heartRate"] = np.nan

        else:

            bestBin   = bestBins[np.nanargmax(critVals)]
            heartRate = heartRates[np.nanargmax(critVals)]

            outputDict["bestBin"] = bestBin
            outputDict["heartRate"] = heartRate


    return outputDict



##########################################################################################
### method swt: meant for the reference (ECG) data, uses the Stationary Wavelet Transform
##########################################################################################
def computeHR_swt(srate, data):
    """
    Estimate heart rate from ECG-like reference data using the Stationary Wavelet Transform.

    Args:
        srate (float): Sampling rate in Hz.
        data (np.ndarray): 1D array (n_samples,) of reference signal (e.g., ECG). No prior filtering required.

    Returns:
        tuple:

            - critVal (float): Quality metric (higher is better), defined as 1 / std(IBI) after artifact handling; np.nan if invalid.
            - heartRate (float): Estimated heart rate in BPM; np.nan if estimation fails.

    Method:

        - Choose SWT level to target cardiac content.
        - Pad signal symmetrically to a valid SWT length, apply SWT ("sym4"), and select a coefficient band of interest.

        - Compute envelope via Hilbert transform, smooth (~0.3 s moving average), and detect peaks.
        - Build inter-beat intervals (IBIs); replace contiguous outliers by local median using intervalExtractor.

        - If IBIs pass stability and physiologic tests (IBI std, fraction of out-of-range intervals, IBI range),

          compute HR as 60 / mean(IBI) and criterion as 1 / std(IBI).

    Notes:

        - Peak detection uses a prominence threshold proportional to signal std and a minimum spacing ~150 BPM.
        - Returns (np.nan, np.nan) if insufficient peaks or unstable IBIs.

    """
    # Determine which decomposition level contains the desired frequency
    swtLvl = np.ceil(np.log2((srate/2) / 3.5))
    
    # Prepare the data for SWT
    if len(data) % (2**swtLvl) != 0:
        necessaryLen = (2**swtLvl) * np.ceil(len(data) / (2**swtLvl))
        startPad = int(np.floor((necessaryLen-len(data))/2))
        endPad = int(np.ceil((necessaryLen-len(data))/2))
        dataPad = np.pad(data, (startPad, endPad))

    else:
        dataPad = copy.deepcopy(data)
        startPad = 0
        endPad = 0
    
    # Compute the SWT
    coeffs = pywt.swt(dataPad,wavelet="sym4",level=swtLvl,trim_approx=True)

    # Extract and trim the coefficients which are of interests
    COI = coeffs[1]
    COI = COI[startPad:len(COI)-endPad]

    # Compute and smooth the envelope of the coefficient vector
    dataSquare = np.abs(signal.hilbert(COI*-1))
    k = np.round(srate*.3).astype("int")    
    dataSquare = signal.filtfilt(np.ones(k)/k,1,dataSquare)

    # Search for the peaks in the transformed coefficient vector
    peakInds = signal.find_peaks(dataSquare, prominence=1*np.std(dataSquare,ddof=1), distance=(60/150)*srate)[0]
    if (not any(peakInds)) or (len(peakInds) < 3):
        critVal   = np.nan
        heartRate = np.nan

    else:
        # Compute the inter-peak-intervals and then apply a median filter to deal with artifacts
        peakIntervals = np.diff(peakInds) / srate
        peakIntervalsOG = copy.deepcopy(peakIntervals)

        outUpThr  = np.mean(peakIntervals) + 1*np.std(peakIntervals,ddof=1)
        outLowThr = np.mean(peakIntervals) - 1*np.std(peakIntervals,ddof=1)
        outlierInds = np.where((peakIntervals > outUpThr) | (peakIntervals < outLowThr))[0]
        outlierIvs = outlierInds[intervalExtractor(outlierInds)[0]]

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

        critVal   = np.nan
        heartRate = np.nan

        # Compute the average inter-peak-interval
        medPeakInterval = np.mean(peakIntervals)

        std2medianRatio = np.std(peakIntervals, ddof=1) / medPeakInterval
        badIntervalsCount = np.sum( (peakIntervals < (60/150)) | (peakIntervals > (60/40)) )

        # Some tests which ensure that the inter-peak-intervals correspond to a good heartbeat signal
        if (std2medianRatio < .2) & (badIntervalsCount/len(peakIntervals) < .1) & (medPeakInterval >= (60/150)) & (medPeakInterval <= (60/40)):
            critVal = 1/np.std(peakIntervals,ddof=1)
            # Compute the heartrate and transform it to 1/min
            heartRate = 60 / medPeakInterval

    return critVal, heartRate


#########################################################################
### method normPeaks: simply finds the peaks in the normed data
#########################################################################
def computeHR_normPeaks(srate, data):
    """
    Estimate heart rate from normalized data via peak detection and inter-beat interval analysis.

    Args:
        srate (float): Sampling rate in Hz.
        data (np.ndarray): 1D normalized signal (n_samples,), typically amplitude-normalized output used in TPC pipeline.

    Returns:
        tuple:

            - critVal (float): Quality metric (1 / std(IBI)) after artifact handling; np.nan if invalid.
            - heartRate (float): Estimated heart rate in BPM; np.nan if estimation fails.

    Method:

        - Detect peaks with high normalized amplitude (>= 0.9) and minimum spacing ~150 BPM.
        - Build inter-beat intervals (IBIs). If >25% IBIs are outliers, reject.

        - Replace contiguous outliers by local median via intervalExtractor.
        - Validate IBI stability (std < 0.1 s), physiologic bounds (40–150 BPM), and compute HR as 60 / mean(IBI).

    Notes:

        - Expects pre-normalized data; for raw/radar data use TPC pipeline in checkBins.

    """
    # Search for the peaks in the normed data
    peakInds = signal.find_peaks(data, height=.9, distance=srate*60/150)[0]
    
    # Test the found peak indices
    if (not any(peakInds)) or (len(peakInds) < 5) or ( np.std( np.diff(peakInds), ddof=1 )/srate > .3 ):
        critVal   = np.nan
        heartRate = np.nan

    else:
        # Compute inter-peak-intervals, and then apply a median filter to deal with artifacts
        peakIntervals = np.diff(peakInds) / srate

        outUpThr  = np.mean(peakIntervals) + 1*np.std(peakIntervals,ddof=1)
        outLowThr = np.mean(peakIntervals) - 1*np.std(peakIntervals,ddof=1)
        outlierInds = np.where((peakIntervals > outUpThr) | (peakIntervals < outLowThr))[0]

        if len(outlierInds)/len(peakIntervals) > .25:
            critVal   = np.nan
            heartRate = np.nan

        else:   
            outlierIvs = outlierInds[intervalExtractor(outlierInds)[0]]

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
            
            critVal   = np.nan
            heartRate = np.nan

            # Compute the average inter-peak-interval
            avgPeakInterval = np.mean(peakIntervals)

            # Compute and test some indicatory values, to find out whether the inter-peak-interval series is physiologically valid
            IBIstd = np.std(peakIntervals, ddof=1)
            badIntervalsCount = np.sum( (peakIntervals < (60/150)) | (peakIntervals > (60/40)) )
            if (IBIstd < .1) & (badIntervalsCount/len(peakIntervals) < .1) & (avgPeakInterval >= (60/150)) & (avgPeakInterval <= (60/40)):

                critVal   = 1 / np.std(peakIntervals,ddof=1)
                # Compute the heartrate and transform it to 1/min
                heartRate = 60 / avgPeakInterval


    return critVal, heartRate

