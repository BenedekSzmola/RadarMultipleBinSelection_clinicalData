"""
File with various helper functions which are shared by the other scripts.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
import copy
from ripser import ripser
from persim import plot_diagrams


##############################################
### utility for loading the data
##############################################
def giveMeasFilePath(recID):
    """
    Build the absolute file path to the evaluated measurement .pkl file for a given recording ID.

    Args:
        recID (str): Recording ID (e.g., "S027"). Determines whether the path uses the pilot subdirectory.

    Returns:
        str: Absolute path to the .pkl file for the specified recording.

    """
    ### -> input the path to your measurement files here
    filePath = ""

    return filePath
##############################################

##############################################
### utility for loading the data
##############################################
def giveSaveFilePath():
    """
    Return the default directory path used to save vital-rate analysis results.

    Returns:
        str: Absolute path to the save directory.
    """
    ### -> put the path where you want your savefiles to be saved
    return f""
##############################################
##############################################

##############################################
### utility for finding saved analysis files
##############################################
def findCorrectSave(saveFiles,strs2find):
    """
    Find indices of filenames that contain all given substrings.

    Args:
        saveFiles (Sequence[str]): List/array of filenames.
        strs2find (Sequence[str]): Substrings that must all be present in a filename (case-sensitive).

    Returns:
        np.ndarray: Indices in saveFiles where all substrings are found.
    """
    out = np.full(len(saveFiles),True)
    for stri in range(len(strs2find)):
        out = out & np.array([strs2find[stri] in saveFiles[i] for i in range(len(saveFiles))])
    
    return np.where(out)[0]
##############################################
##############################################

########################################################################
### convert seconds to hhmmss and back
########################################################################
def convertTimeStamp(inputTimestamp: list):
    """
    Convert time between seconds and [hours, minutes, seconds].

    Args:
        inputTimestamp (int | float | list): Either:

            - seconds (int/float), or
            - list of length 3: [hours, minutes, seconds], or

            - list of length 1: [seconds].

    Returns:
        int | list: If input is [h, m, s], returns total seconds (int).
                    If input is seconds, returns [hours, minutes, seconds] (list of 3 ints).
    """

    if isinstance(inputTimestamp, int | float):
        inputTimestamp = [inputTimestamp]

    if len(inputTimestamp) == 3:
        outputTimestamp = inputTimestamp[0]*60*60 + inputTimestamp[1]*60 + inputTimestamp[2]

    elif len(inputTimestamp) == 1:
        recHours = inputTimestamp[0] // 3600
        minRem = inputTimestamp[0] % 3600
        recMins = minRem // 60
        recSecs = minRem % 60

        outputTimestamp = [recHours, recMins, recSecs]

    return outputTimestamp
########################################################################
########################################################################

##############################################

### utility for figure xticks (sec --> HHMMSS)

##############################################
def convAxSec2HHMMSS(ax2conv):
    """
    Convert x-axis tick labels from seconds to HH:MM:SS format on a Matplotlib Axes.

    Args:
        ax2conv (matplotlib.axes.Axes): Axes whose x-tick labels are numeric seconds.

    Returns:
        None

    Notes:

        - Reads current tick labels, interprets them as seconds, and rewrites labels as HH:MM:SS.
        - Does not change tick locations.

    """
    timeStampS = [item.get_text() for item in ax2conv.get_xticklabels()]    

    newStamps = ['']*len(timeStampS)
    for i,stampi in enumerate(timeStampS):
        if isinstance(stampi, str):
            stampi = int(stampi)
        stampHours = stampi // 3600
        minRem = stampi % 3600
        stampMins = minRem // 60
        stampSecs = minRem % 60
        newStamps[i] = f"{stampHours:02}:{stampMins:02}:{stampSecs:02}"

    ax2conv.set_xticklabels(newStamps, rotation=0, ha="center")

    return 
##############################################
##############################################

##############################################
### utility for sensor index
##############################################
def getSensorIdx(sensorList,sensorName):
    """
    Find the index of a sensor in the provided sensor list, handling common synonyms.

    Args:
        sensorList (Sequence[str]): List/array of sensor names from the measurement metadata.
        sensorName (str): Target sensor name; supports synonyms:

            - Thorax RIP: ["RIP Thorax", "RIP.Thrx", "RIP Thora"] mapped to "RIP Thora"
            - Abdomen RIP: ["RIP Abdomen", "RIP.Abdom", "RIP Abdom"] mapped to "RIP Abdom"

            - Any other sensor name is matched directly.

    Returns:
        int: Index of the matched sensor within sensorList.

    Raises:
        IndexError: If the sensor name (or mapped synonym) is not found.
    """
    if sensorName in ["RIP Thorax", "RIP.Thrx", "RIP Thora"]:
        sensorIdx = np.where("RIP Thora" == sensorList)[0][0]

    elif sensorName in ["RIP Abdomen", "RIP.Abdom", "RIP Abdom"]:
        sensorIdx = np.where("RIP Abdom" == sensorList)[0][0]

    else:
        sensorIdx = np.where(sensorName == sensorList)[0][0]

    return sensorIdx

##############################################
##############################################

#######################################
### norm data to 0-1
#######################################
def normDataTo_0_1(data2norm):
    """
    Normalize data to the range [0, 1].

    Args:
        data2norm (np.ndarray): 1D or 2D array.

            - If 1D, normalization is over the full vector.
            - If 2D, normalization is row-wise.

    Returns:
        np.ndarray: Normalized data with the same shape as input.

    Notes:

        - Preserves NaNs.
        - If the range is zero (max == min), returns a copy of the input (1D) or may produce NaNs (2D) if range is zero per row.

    """
    if len(data2norm.shape) == 1:
        if (np.nanmax(data2norm) - np.nanmin(data2norm)) == 0:
            normedData = copy.deepcopy(data2norm)
        else:
            normedData = (data2norm - np.nanmin(data2norm)) / (np.nanmax(data2norm) - np.nanmin(data2norm))
    elif len(data2norm.shape) == 2:
        normedData = (data2norm - np.nanmin(data2norm, axis=1, keepdims=True)) / (np.nanmax(data2norm, axis=1, keepdims=True) - np.nanmin(data2norm, axis=1, keepdims=True))

    return normedData
##############################################
##############################################

#########################################
### Script for detecting zero crossings
#########################################
def zeroCrossDetector(data):
    """
    Detect zero-crossings in a 1D signal.

    Args:
        data (np.ndarray): 1D array (n_samples,) of signal values.

    Returns:
        tuple:

            - numZeroCross (int): Number of sign-change points.
            - zeroCrossInds (np.ndarray): Indices where sign changes occur (np.diff(np.sign(data)) != 0).

    """
    zeroCrossInds = np.where(np.diff(np.sign(data)) != 0)[0]
    numZeroCross  = len(zeroCrossInds)

    return numZeroCross,zeroCrossInds

#######################################
### extracts contiguous intervals
#######################################
def intervalExtractor(x, indexCorr=False):
    """
    Convert a sorted list of indices into contiguous intervals and their lengths.

    Args:
        x (array-like of int): Sorted indices (e.g., output of np.where(...)[0]).
        indexCorr (bool): If True, make end indices exclusive (add +1 to the right endpoint).

    Returns:
        tuple:

            - startEndInds (np.ndarray): Shape (n_intervals, 2), pairs [start_idx, end_idx] for each contiguous run.
            - intervalLens (np.ndarray): Shape (n_intervals,), lengths of each interval.

    Notes:

        - If x is empty, returns ([], []).
        - When indexCorr=False, intervals are inclusive on both ends.

    """
    # expects indices, like the output of np.where(...)[0]

    if len(x) == 0:
        return [], []

    edges = np.where( np.diff(x) != 1 )[0]
    edges = np.concatenate(([0], edges, [len(x)-1]))

    startEndInds = np.array( [edges[:-1], edges[1:]] ).T
    startEndInds[1:,0] = startEndInds[1:,0] + 1

    intervalLens = np.squeeze(np.diff(startEndInds, axis=1), axis=1) + 1

    if indexCorr:
        startEndInds[:,1] = startEndInds[:,1] + 1

    return startEndInds, intervalLens


########################################################
### Script that selects bins with advanced thresholding
########################################################
def selectBinsOfInterest(data):
    """
    Select bins-of-interest (BOI) from a 2D data array using variance-based thresholds and local trend analysis.

    Args:
        data (np.ndarray): 2D array of shape (n_bins, n_samples). Each row corresponds to one bin’s time series.

    Returns:
        np.ndarray: Sorted unique array of bin indices considered as BOI.

    Method:

        - Compute per-bin standard deviation across time and a simple threshold (median + 1*std).
        - Identify contiguous intervals of bins above threshold.

        - For each interval, estimate lower and upper bounds using local positive/negative std-diff trends,

          and expand to [lowBin, highBin].

        - Aggregate and return unique bin indices across intervals.

    Notes:

        - Relies on intervalExtractor to group contiguous indices and to handle near-threshold gaps.
        - If trend intervals are missing, falls back to local diff heuristics.

    """
    stdVector = np.std(data, ddof=1, axis=1)
    smoothStdVector = copy.deepcopy(stdVector)
    simplThr = np.median(smoothStdVector) + 1*np.std(smoothStdVector)

    stdDiffs = np.diff(smoothStdVector)

    aboveThrInds = np.where(smoothStdVector > simplThr)[0]
    aboveThrIvs,aboveThrIvLens = intervalExtractor(aboveThrInds)
    aboveThrIvs = aboveThrInds[aboveThrIvs]

    BOI = np.array([],dtype='int')
    for iiv in range(len(aboveThrIvLens)):
        if aboveThrIvLens[iiv] < 2: 
            continue

        if (aboveThrIvs[iiv,0] in BOI) and (aboveThrIvs[iiv,1] in BOI):
            continue

        if iiv < (len(aboveThrIvLens)-1):
            i = 0
            while (iiv+i+1 < len(aboveThrIvLens)) and ((aboveThrIvs[iiv+i+1,0] - aboveThrIvs[iiv+i,1]) < 3):
                i += 1

            upperThrCross = aboveThrIvs[iiv+i,1]

        else:
            upperThrCross = aboveThrIvs[iiv,1]

        upperThrCross = int(upperThrCross)
        
        if iiv > 0:
            i = 0
            while (iiv-(i+1) >= 0) and ((aboveThrIvs[iiv-i,0] - aboveThrIvs[iiv-(i+1),1]) < 3):
                i += 1

            lowerThrCross = aboveThrIvs[iiv-i,0]

        else:
            lowerThrCross = aboveThrIvs[iiv,0]

        lowerThrCross = int(lowerThrCross)

        posDiffInds = np.where(stdDiffs > 0)[0]
        posDiffIntervals,posDiffIvLens = intervalExtractor(posDiffInds)
        posDiffIntervals = posDiffInds[posDiffIntervals]

        negDiffInds = np.where(stdDiffs < 0)[0]
        negDiffIntervals,negDiffIvLens = intervalExtractor(negDiffInds)
        negDiffIntervals = negDiffInds[negDiffIntervals]

        if (len(posDiffIvLens) == 0) or (len(negDiffIvLens) == 0):
            continue

        negIvs2check = np.where((negDiffIvLens > 1) & (negDiffIntervals[:,1] < lowerThrCross))[0]
        if any(negIvs2check):
            lowBin = negDiffIntervals[negIvs2check[np.argmin(np.abs(negDiffIntervals[negIvs2check,1] - lowerThrCross))],1]
        elif lowerThrCross >= 2:
            lowBin = np.argmax(np.diff(stdDiffs[:lowerThrCross]))
        else:
            lowBin = 0

        posIvs2check = np.where((posDiffIvLens > 1) & (posDiffIntervals[:,0] > upperThrCross))[0]
        if any(posIvs2check):
            highBin = posDiffIntervals[posIvs2check[np.argmin(np.abs(posDiffIntervals[posIvs2check,1] - upperThrCross))],0]
        elif upperThrCross <= (len(stdDiffs)-2):
            highBin = np.argmax(np.diff(stdDiffs[upperThrCross:])) + upperThrCross - 1
        else:
            highBin = 0

        BOI = np.append(BOI,np.arange(lowBin,highBin+1).astype('int'))

    return np.unique(BOI)

########################################################################
### Construct sparse matrix for sublevel filtration
########################################################################
def doSubLVLfiltration(x, delInf=True, smallLifeThr=1e-3, showPlot=False, timeVec=None):
    """
    Compute a 0-dimensional sublevel-filtration persistence diagram for a time series.

    Args:
        x (np.ndarray): 1D array of samples.
        delInf (bool): If True, drop the infinite-death point from the diagram.
        smallLifeThr (float): Remove points with lifespan <= smallLifeThr.
        showPlot (bool): If True and timeVec is provided, plot signal and its persistence diagram.
        timeVec (np.ndarray | None): Time vector aligned with x for plotting.

    Returns:
        np.ndarray: H0 persistence diagram (n_points, 2) with columns [birth, death].

    Method:

        - Construct a sparse distance matrix encoding sublevel filtration (max of adjacent values; diagonal as vertex births).
        - Use ripser(distance_matrix=True) to compute H0 diagram.

        - Optionally filter and plot results.

    """
    N = len(x)

    I = np.arange(N-1)
    J = np.arange(1, N)
    V = np.maximum(x[0:-1], x[1::])

    # Add vertex birth times along the diagonal of the distance matrix
    I = np.concatenate((I, np.arange(N)))
    J = np.concatenate((J, np.arange(N)))
    V = np.concatenate((V, x))

    #Create the sparse distance matrix
    D = sparse.coo_matrix((V, (I, J)), shape=(N, N)).tocsr()

    dgm0 = ripser(D, maxdim=0, distance_matrix=True)['dgms'][0]

    if delInf:
        dgm0 = dgm0[:-1,:]
        
    dgm0 = dgm0[dgm0[:, 1]-dgm0[:, 0] > smallLifeThr, :]

    if showPlot and (timeVec is not None):
        allgrid = np.unique(dgm0.flatten())
        allgrid = allgrid[allgrid < np.inf]
        xs = np.unique(dgm0[:, 0])
        ys = np.unique(dgm0[:, 1])
        ys = ys[ys < np.inf]

        #Plot the time series and the persistence diagram
        plt.figure(figsize=(16, 6))
        ylims = [np.min(x)-.1, np.max(x)+.1]
        plt.subplot(121)
        plt.plot(timeVec, x)
        ax = plt.gca()
        ax.set_yticks(allgrid)
        ax.set_xticks([])
        plt.ylim(ylims)
        plt.grid(linewidth=1, linestyle='--')
        plt.title("Time domain signal")
        plt.xlabel("Time [s]")

        plt.subplot(122)
        ax = plt.gca()
        ax.set_yticks(ys)
        ax.set_xticks(xs)
        plt.grid(linewidth=1, linestyle='--')
        plot_diagrams(dgm0, size=50)
        plt.ylim(ylims)
        plt.title("Persistence Diagram")


        plt.show()

    return dgm0


########################################################################
### Construct Time delay embedded vector
########################################################################
def createDelayVector(x, tau, m):
    """
    Create a time-delay embedding matrix from a 1D time series.

    Args:
        x (np.ndarray): 1D array of samples.
        tau (int): Time delay (samples) between embedding dimensions.
        m (int): Embedding dimension.

    Returns:
        np.ndarray: Delay-embedded matrix of shape (len(x) - (m-1)*tau, m), where column d is x shifted by d*tau.

    Raises:
        ValueError: If len(x) <= (m-1)*tau (insufficient length for requested embedding).
    """
    delayVec = np.zeros((len(x)-(m-1)*tau,m))

    for dimi in range(m):
        if dimi < (m-1):
            delayVec[:,dimi] = x[dimi*tau:-(m-1-dimi)*tau]
        else:
            delayVec[:,dimi] = x[dimi*tau:]

    return delayVec
########################################################################
########################################################################

########################################################################
### the start of synch markers for each subject in clock time
### (unless the starting markers arent whole, then last)
### this is from where the LSL timeaxis counts
### important for the label vector importing from DOMINO
### gives a dict where the code of each subject is the key,
###     and the value is a len=3 list [hh,mm,ss]
########################################################################
def perSubjStartClocks():
    """
    Return per-subject LSL start clocks in [hh, mm, ss] format.

    Args:
        None

    Returns:
        dict: Mapping {recID: [hours, minutes, seconds]} for subjects S027–S075.
    """
    perSubjStartClock = {
        'S027': [17,15,23],
        'S028': [17,21,14],
        'S029': [17,27,37],
        'S030': [17,45,52],
        'S031': [19,35,46],
        'S032': [19,13,46],
        'S033': [19,21,28],
        'S034': [19,33,46],
        'S035': [19,44,41],
        'S036': [18,19,35],
        'S037': [20,3,25],
        'S038': [18,22,50],
        'S039': [19,20,53],
        'S040': [19,29,15],
        'S041': [20,4,5],
        'S042': [19,16,33],
        'S043': [21,14,37],
        'S044': [19,30,30],
        'S045': [18,21,10],
        'S046': [19,17,26],
        'S047': [18,21,2],
        'S048': [18,28,7],
        'S049': [19,42,8],
        'S050': [18,46,41],
        'S051': [19,28,50],
        'S052': [18,29,29],
        'S053': [18,19,1],
        'S054': [18,26,53],
        'S055': [18,17,28],
        'S056': [18,23,26],
        'S057': [18,22,9],
        'S058': [18,8,31],
        'S059': [18,10,2],
        'S060': [18,8,50],
        'S061': [18,11,22],
        'S062': [18,32,7],
        'S063': [18,56,10],
        'S064': [18,9,21],
        'S065': [18,41,55],
        'S066': [18,15,14],
        'S067': [18,16,6],
        'S068': [18,27,9],
        'S069': [18,13,57],
        'S070': [18,26,11],
        'S071': [18,17,5],
        'S072': [18,20,55],
        'S073': [18,10,1],
        'S074': [18,12,23],
        'S075': [18,12,46]
    }

    return perSubjStartClock
########################################################################
########################################################################

########################################################################
### the borders of the data where there was sensible data
########################################################################
def perSubjCuts():
    """
    Return per-subject cut intervals (in seconds) designating the sensible/valid data range.

    Args:
        None

    Returns:
        dict: Mapping {recID: [start_sec, end_sec]} for subjects S027–S075.
    """
    perSubjCuts = {
        'S027': [17000,46000],
        'S028': [3000,49000],
        'S029': [1600,49000],
        'S030': [7600,47500],
        'S031': [2750,41000],
        'S032': [11000,41000],
        'S033': [5800,42000],
        'S034': [3200,38000],
        'S035': [6300,40000],
        'S036': [14000,45000],
        'S037': [9500, 34000],
        'S038': [13500, 42000],
        'S039': [5000, 39000],
        'S040': [3000, 35000],
        'S041': [2500, 33000],
        'S042': [19000, 40000],
        'S043': [1800, 33000],
        'S044': [1000, 40000],
        'S045': [10000, 43000],
        'S046': [10000, 39000],
        'S047': [13000, 44000],
        'S048': [10000, 32500],
        'S049': [9000,38500],
        'S050': [14500,44000],
        'S051': [6900,39000],
        'S052': [1200,44000],
        'S053': [11000,44400],
        'S054': [8500,44400],
        'S055': [16000,45000],
        'S056': [14000,44000],
        'S057': [4700,44000],
        'S058': [13500,37800],
        'S059': [7000,43000],
        'S060': [6700,45000],
        'S061': [7200,42000],
        'S062': [13000,43000],
        'S063': [15000,41000],
        'S064': [5000,45500],
        'S065': [11000,42000],
        'S066': [6500,45000],
        'S067': [6500,41500],
        'S068': [13700,39500],
        'S069': [15700,43000],
        'S070': [14300,39000],
        'S071': [12500,43400],
        'S072': [1100,45300],
        'S073': [15000,45700],
        'S074': [13300,45700],
        'S075': [30000,42500]
    }

    return perSubjCuts
########################################################################
########################################################################