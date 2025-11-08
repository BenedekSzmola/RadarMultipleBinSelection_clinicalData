import numpy as np
from scipy import signal, stats

from helperFunctions import selectBinsOfInterest,zeroCrossDetector

def checkBins(srate, data, bins2check, magData=None, binSelMethod="TPC", doBinPreFilt=False, doDiffTest=True):
    """
    Select the best range bin(s) and estimate breathing rate from radar (or reference) data.

    Args:
        srate (float): Sampling rate in Hz.
        data (np.ndarray): 2D array of shape (n_bins, n_samples). For reference data, pass shape (1, n_samples).
        bins2check (array-like of int): Candidate range-bin indices to evaluate.
        magData (np.ndarray | None): 2D array of shape (n_bins, n_samples) of magnitude signals used for
            pre-filtering bin selection (body presence). If None, magnitude-based pre-filtering is skipped.
        binSelMethod (str | None): Method to select bins:

            - "TPC": Use a zero-crossing based time-periodicity criterion to find best bins (default).
            - None: Treat input as reference sensor (single-channel), bypass bin selection, compute BR directly.

        doBinPreFilt (bool): If True and magData is provided, bandpass-filter magnitude (.1-.6 Hz) and
            select bins with higher variance via selectBinsOfInterest.
        doDiffTest (bool): If True, discard bins whose max sample-to-sample difference exceeds 1 (artifact screen).

    Returns:
        dict: With keys:

            - "bestBin" (int | float): Selected bin index (or 0 for reference data). np.nan if none selected.
            - "respRate" (float): Estimated breathing rate in breaths per minute (BPM). np.nan if not available.

    Notes:

        - For binSelMethod="TPC", the signal is bandpass-filtered (.1-.6 Hz), normalized via analytic amplitude

          (Hilbert), and bin periodicity is scored using time-periodicity correlation (TPC).

        - The top 5 bins by TPC are evaluated via autocorrelation-based BR estimation; the bin with the

          strongest combined criterion (ranked sum of peak sum and 1/inter-peak std) is selected.

        - If no bins pass the artifact screens or pre-filtering, returns NaNs.

    """
    outputDict = {}
    
    # Screen out signals with unexpectedly large sample-to-sample differences
    if doDiffTest:
        bins2check = bins2check[np.where(np.max(np.abs(np.diff(data[bins2check,:], axis=1)), axis=1) < 1)[0]]

        if len(bins2check) == 0:
            outputDict["bestBin"]  = np.nan
            outputDict["respRate"] = np.nan

            return outputDict

    # Select the range bins corresponding to the human body based on the standard deviation of magnitude
    if (magData is not None) and doBinPreFilt:
        b,a = signal.butter(N=2,Wn=[.1,.6], btype='bandpass', fs=srate)
        
        magDataFilt = signal.filtfilt(b,a,magData)
        bins2check = bins2check[selectBinsOfInterest(magDataFilt[bins2check,:])]
        
        if len(bins2check) == 0:
            outputDict["bestBin"]  = np.nan
            outputDict["respRate"] = np.nan

            return outputDict
        
    # BR computation for reference data
    if binSelMethod is None:
        b,a = signal.butter(N=2,Wn=[.1,.6], btype='bandpass', fs=srate)
        
        dataFilt = signal.filtfilt(b,a,data[0,:])

        returDict = computeBR_acorr(srate, dataFilt)
        outputDict["bestBin"] = 0
        outputDict["respRate"] = returDict["breathRate"]

    # TPC method for selecting the 5 best range bins showing periodic signals
    elif binSelMethod == "TPC":
        # Apply bandpass filter in the physiological breathing frequency range
        b,a = signal.butter(N=2,Wn=[.1,.6], btype='bandpass', fs=srate)
        dataFilt = signal.filtfilt(b,a,data)

        # Initialize arrays for the TPC values and the normed data
        TPCarray = np.zeros(len(bins2check))
        dataNorm = np.zeros_like(data)

        for i,binNum in enumerate(bins2check):
            # Norm the filtered data for TPC computation
            amps = np.abs(dataFilt[binNum,:] + 1j*signal.hilbert(dataFilt[binNum,:]))
            dataNorm[binNum,:] = dataFilt[binNum,:] / amps

            # Locate the zero crossings for TPC computation
            numZeroCross,zeroCrossInds = zeroCrossDetector(dataNorm[binNum,:])
            Tval = int(2*np.sum(np.diff(zeroCrossInds)) / (numZeroCross-1))
            TPCarray[i] = np.sum(dataNorm[binNum,0:-Tval]*dataNorm[binNum,Tval:]) / (np.std(dataNorm[binNum,0:-Tval],ddof=1) * np.std(dataNorm[binNum,Tval:],ddof=1))

        # Taking the absolute value of the computed TPC
        TPCarray = np.abs(TPCarray)

        # Selecting the 5 highest TPC bins
        if len(TPCarray) >= 5:
            bestBins = bins2check[np.argpartition(TPCarray,-5)[-5:]]
        else:
            bestBins = bins2check[np.argpartition(TPCarray,-len(TPCarray))[-len(TPCarray):]]

        data2use = dataFilt[bestBins,:]

        # Initialize arrays for the BRs and the criterion values which will be used to select the best bin
        critVals = np.full((len(bestBins),2), np.nan)
        breathRates = np.full(len(bestBins), np.nan)
        for i in range(len(bestBins)):
            print(f"Currently computing BR for range bin #{bestBins[i]}")
            returDict = computeBR_acorr(srate, data2use[i,:])
            critVals[i,:]  = returDict['critVal']
            breathRates[i] = returDict['breathRate']
            
        # Transform the criterion values
        critVals = stats.rankdata(critVals,axis=0,nan_policy='omit')
        critVals = np.nansum(critVals,axis=1)
        
        # Select the best bin and extract the corresponding breathing rate
        bestBin    = bestBins[np.nanargmax(critVals)]
        breathRate = breathRates[np.nanargmax(critVals)]

        outputDict["bestBin"]  = bestBin
        outputDict["respRate"] = breathRate

    return outputDict


#########################################################################
### method acorr for breathing rate computation
#########################################################################
def computeBR_acorr(srate, data):
    """
    Estimate breathing rate using the autocorrelation method with artifact screening.

    Args:
        srate (float): Sampling rate in Hz.
        data (np.ndarray): 1D array of shape (n_samples,) containing a single-channel time series
            (ideally bandpass-filtered to .1–.6 Hz prior to calling).

    Returns:
        dict: With keys:

            - 'critVal' (np.ndarray): Shape (2,), criterion values [sum of autocorr peaks, 1/inter-peak std].
            - 'breathRate' (float): Breaths per minute (BPM). np.nan if screening fails or peaks are insufficient.

    Method:

        - Compute autocorrelation and normalize to unit peak.
        - Build lag vector (seconds).

        - Artifact screening based on width between the first two negative peaks of the autocorrelogram;

          acceptable width range corresponds to BPM in [5, 30].

        - Find positive peaks within physiologic lags (approximately 2–12 s), enforce minimum peak spacing.
        - If multiple peaks exist and inter-peak variability is small (std < 1 s), define criteria and

          compute BR as 60 / lag_of_first_positive_peak.

    Notes:

        - Returns NaNs if negative-peak check fails or insufficient peaks are detected.
        - Set highBPM_thr=30 to bound the upper BPM limit used in screening.

    """
    highBPM_thr = 30
    
    # Compute autocorrelation, and norm to 1
    autocorr = signal.correlate(data, data)
    autocorr = autocorr / np.max(autocorr)

    # Create array with the lag values in seconds
    autocorr_lags = signal.correlation_lags(len(data), len(data))
    autocorr_lags = autocorr_lags / srate

    # Use artifact screening method, based on width of the main autocorrelogram peak
    firstNegPeaks = signal.find_peaks(-autocorr)[0]
    if len(firstNegPeaks) < 2:
        return {'critVal': np.array([np.nan,np.nan]), 'breathRate': np.nan}
    else:
        firstNegPeaks = firstNegPeaks[np.argpartition(np.abs(autocorr_lags[firstNegPeaks]),1)[:2]]
        acorrWidth = autocorr_lags[firstNegPeaks[1]] - autocorr_lags[firstNegPeaks[0]]
        if (acorrWidth < 60/highBPM_thr) or (acorrWidth > 60/5):
            return {'critVal': np.array([np.nan,np.nan]), 'breathRate': np.nan}

    # Search for the peaks in the autocorrelogram in the physiologically sensible range
    acorrPosInds = np.where((autocorr_lags >= 60/highBPM_thr) & (autocorr_lags <= np.min((2.1*60/5,np.max(autocorr_lags)))) )[0]
    acorrPeaks,_ = signal.find_peaks(autocorr[acorrPosInds],distance=np.round((60/highBPM_thr)*srate).astype("int"),height=0)
    
    critVal    = np.array([np.nan,np.nan])
    breathRate = np.nan

    # Apply some tests and if they are passed compute the criterion value and the breathing rate
    if any(acorrPeaks) and (len(acorrPeaks) > 1):
        acorrPeaks = acorrPosInds[acorrPeaks]

        interPeakStd = np.std(np.diff(autocorr_lags[acorrPeaks],prepend=0),ddof=1)

        if (interPeakStd < 1) and (autocorr_lags[acorrPeaks[0]] >= (1/.5)) and (autocorr_lags[acorrPeaks[0]] <= (1/.1)):
            critVal    = np.array( [np.sum(autocorr[acorrPeaks]) , 1 / interPeakStd] )
            breathRate = 60 / autocorr_lags[acorrPeaks[0]]

    returDict = {'critVal': critVal,
                 'breathRate': breathRate}

    return returDict
