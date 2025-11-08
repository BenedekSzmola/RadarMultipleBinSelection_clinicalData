########################################################################
###### import python packages ##########################################
########################################################################
import numpy as np
from scipy import signal
import copy
from joblib import Parallel, delayed

from ripser import ripser

########################################################################

########################################################################
### import methods from other scripts ##################################
########################################################################
import helperFunctions as hF
from MBS_persDiagFeatureMiner import persDiagramChecker_DBSCAN,subLevelFiltPersDgmChecker_DBSCAN
########################################################################

#########################################################################
### complete epochs checkbin
#########################################################################
def checkBin_allEpochs(srate, dataEpochs, bins2checkFull, epochInds2skip=None):
    """
    Classify breathing-relevant bins across multiple epochs using VR (delay-embedding + ripser) and
    sublevel-filtration persistence heuristics.

    Args:
        srate (float): Sampling rate in Hz.
        dataEpochs (list[np.ndarray]): List of per-epoch arrays, each shape (n_bins, n_samples).
        bins2checkFull (array-like of int): Candidate bin indices to evaluate.
        epochInds2skip (array-like of int | None): Optional epoch indices to skip from processing.

    Returns:
        np.ndarray: binClasses of shape (len(bins2check), numEpochs), entries in {0,1,2} where
            0=failure, 1=possibly flawed, 2=good.

    Notes:

        - Preprocessing: per-epoch lowpass (5 Hz, N=4) and row-wise normalization to [0,1].
        - For each bin: build delay embedding (tau=10, m=3), compute diagrams via ripser,

          and classify via persDiagramChecker_DBSCAN and subLevelFiltPersDgmChecker_DBSCAN.

        - The final bin category is 2 if either method yields 2; otherwise the min of the two categories.
        - Epochs are processed in chunks (size=1000) with parallelization across epochs and bins.

    """
    bins2check = copy.deepcopy(bins2checkFull)

    numEpochs = len(dataEpochs)

    len_s = int(dataEpochs[0].shape[1] / srate)

    tau = 10
    m = 3

    def prepare_filtData(epochi):
        b,a = signal.butter(N=4, Wn=5, fs=srate, btype="lowpass")
        data2use = signal.filtfilt(b,a,dataEpochs[epochi])
        
        return hF.normDataTo_0_1(data2use)
    
    data2use_epochs = []
    for epochi in range(numEpochs):
        print(f'Prepareing filt data, epoch {epochi} / {numEpochs}')
        data2use_epochs.append(prepare_filtData(epochi))

    def prepare_bin_diagram_and_class(curr_data2use):
        embed = hF.createDelayVector(curr_data2use, tau, m)

        diagrams = ripser(embed)['dgms']
        diagrams[0] = diagrams[0][:-1, :]

        VR_output, subLvl_output = Parallel(n_jobs=2, backend='loky')([
            delayed(persDiagramChecker_DBSCAN)(diagrams, len_s, showLogs=False),
            delayed(subLevelFiltPersDgmChecker_DBSCAN)(curr_data2use, len_s, showLogs=False)
        ])

        # Determine the diagnostic category
        diagCategory = 2 if (VR_output[0] == 2) or (subLvl_output['class'] == 2) else np.min((VR_output[0], subLvl_output['class']))

        return diagCategory

    def compute_epoch_diagram_and_class(curr_data2use_epoch):
        curr_epoch_binCats = Parallel(n_jobs=3, backend='loky')(
            delayed(prepare_bin_diagram_and_class)(curr_data2use_epoch[binNum,:]) for binNum in bins2check
        )

        return curr_epoch_binCats
    
    binClasses = np.full((len(bins2check), numEpochs), np.nan)

    epoch_chunk_len = 1000
    for curr_chunk_start in range(0, numEpochs, epoch_chunk_len):
        print('Starting new epoch chunk for computing the diagrams and then classifying, starting epochind: ',curr_chunk_start)
        curr_chunk_inds = np.arange(curr_chunk_start, np.min((numEpochs, curr_chunk_start+epoch_chunk_len)))

        # if there is the epochs2skip input, knock out the indices which shouldnt be checked
        if epochInds2skip is not None:
            curr_chunk_inds = curr_chunk_inds[~np.isin(curr_chunk_inds, epochInds2skip)]

        curr_chunk_binCats = Parallel(n_jobs=4, backend='loky', verbose=50)(
            delayed(compute_epoch_diagram_and_class)(data2use_epochs[epochi]) for epochi in curr_chunk_inds
        )

        for i,epochi in enumerate(curr_chunk_inds):
            binClasses[:,epochi] = curr_chunk_binCats[i]

    return binClasses
##############################################################################################################################################################################
##############################################################################################################################################################################


#########################################################################
### method acorr
#########################################################################
def computeBR_acorr(srate, data):
    """
    Estimate breathing rate using autocorrelation with artifact screening and peak analysis.

    Args:
        srate (float): Sampling rate in Hz.
        data (np.ndarray): 1D array (n_samples,) of the time series.

    Returns:
        dict: {'critVal': np.ndarray, 'breathRate': float}

            - critVal: Shape (2,), [sum of autocorr peaks, 1/inter-peak std]; NaNs if invalid.
            - breathRate: Breaths per minute (BPM); NaN if estimation fails.

    Method:

        - Bandpass filter (.1–.6 Hz).
        - Compute normalized autocorrelation and lag array (seconds).

        - Artifact screen via width between first two negative peaks (corresponds to BPM in [5,30]).
        - Find positive peaks in physiologic lag range; enforce minimum spacing.

        - If multiple peaks and inter-peak variability is small (std < 1 s), define criteria and compute

          breathRate = 60 / lag_of_first_positive_peak.
    """
    highBPMthr = 30
    b,a = signal.butter(N=2,Wn=[.1,.6], btype='bandpass', fs=srate)
    
    dataFilt = signal.filtfilt(b,a,data)

    ##############################
    ### Compute autocorrelation
    ##############################

    autocorr = signal.correlate(dataFilt, dataFilt)
    autocorr = autocorr / np.max(autocorr)

    autocorr_lags = signal.correlation_lags(len(dataFilt), len(dataFilt))
    autocorr_lags = autocorr_lags / srate

    firstNegPeaks = signal.find_peaks(-autocorr)[0]
    if len(firstNegPeaks) < 2:
        return {'critVal': np.full(2, np.nan), 'breathRate': np.nan}
    else:
        firstNegPeaks = firstNegPeaks[np.argpartition(np.abs(autocorr_lags[firstNegPeaks]),1)[:2]]
        acorrWidth = autocorr_lags[firstNegPeaks[1]] - autocorr_lags[firstNegPeaks[0]]
        if (acorrWidth < 60/highBPMthr) or (acorrWidth > 60/5):
            return {'critVal': np.full(2, np.nan), 'breathRate': np.nan}

    #########################################
    ### Test for breathing activity
    #########################################

    acorrPosInds = np.where((autocorr_lags >= 60/highBPMthr) & (autocorr_lags <= np.min((2.1*60/5,np.max(autocorr_lags)))) )[0]

    acorrPeaks,_ = signal.find_peaks(autocorr[acorrPosInds],distance=np.round((60/highBPMthr)*srate).astype("int"),height=0)

    critVal    = np.full(2, np.nan)
    breathRate = np.nan

    if any(acorrPeaks) and (len(acorrPeaks) > 1):
        acorrPeaks = acorrPosInds[acorrPeaks]

        interPeakStd = np.std(np.diff(autocorr_lags[acorrPeaks],prepend=0),ddof=1)

        if (interPeakStd < 1) and (autocorr_lags[acorrPeaks[0]] >= (1/.5)) and (autocorr_lags[acorrPeaks[0]] <= (1/.1)):
            critVal    = np.array( [np.sum(autocorr[acorrPeaks]) , 1 / interPeakStd] )
            breathRate = 60 / autocorr_lags[acorrPeaks[0]]

    returDict = {'critVal': critVal,
                 'breathRate': breathRate}

    return returDict