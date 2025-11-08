#######################################################################################################################

### Import packages

#######################################################################################################################
import numpy as np
import copy

# import python libraries

from sklearn.cluster import DBSCAN
from ripser import ripser
from persim import persistent_entropy

#######################################################################################################################

# import from other files of the project
import helperFunctions as hF

#######################################################################################################################
#######################################################################################################################

def persDiagramChecker_DBSCAN(diagrams, timeWinLen, showLogs=False):
    """
    Classify a 2D persistence diagram (H0/H1) using DBSCAN-based heuristics and compute a score.

    Args:
        diagrams (list[np.ndarray]): [H0, H1] arrays with shape (n_points, 2) each, containing [birth, death].

            - If the last row in H0 has death=inf, it is removed.

        timeWinLen (float): Epoch/window length in seconds; used to bound expected outlier count via an upper BPM limit (30 BPM).
        showLogs (bool): If True, print intermediate diagnostics.

    Returns:
        tuple:

            - category (int): 0=failure, 1=possibly flawed, 2=good.
            - score (float): Heuristic score; np.nan if category=0.

    Method:

        - Remove infinite-death point in H0, compute lifespans, and basic validity checks.
        - Compute normalized persistent entropy of the diagram.

        - For H1: remove negative lifespans, transform to (birth, lifespan).
        - Choose DBSCAN eps from the "diagram of diagram" (H0 of the H1 lifespans) by the largest quantile gap.

        - Fit DBSCAN on H1 points; treat cluster points as noise/background, and outliers as candidate fundamental loops.
        - Filter outliers by lifespan and birth location relative to the largest background cluster.

        - Enforce a maximum outlier count given timeWinLen and an upper BPM threshold.
        - Score is a product of entropy and geometric terms; assign category by rule-based checks.

    Notes:

        - This function may slice diagrams[0] in place to drop the infinite point; pass copies if needed.
        - Returns (0, nan) for multiple failure conditions (insufficient points, no clusters, etc.).

    """
    def returnHelper(showLogs=showLogs,logMsg='Empty log'):        
        if showLogs:
            print(logMsg)
            print('-----------------------------------------------')

        return

    
    # The return values are: the assigned category, and the score given
    # The category assigned to the input diagram can be:
    # 0 ... failure
    # 1 ... possibly flawed in some way
    # 2 ... good

    if diagrams[0].shape[0] == 0:
        returnHelper(logMsg="Diagram failed because a lack of H0 points.")

        return 0,np.nan

    upperBPMthr = 30

    # extract the info from the input diagram list for easy access
    if np.isinf(diagrams[0][-1,1]):
        diagrams[0] = diagrams[0][:-1,:]
    
    lifespanH0 = np.diff(diagrams[0], axis=1).squeeze(axis=1)

    # Do length checks (if there aren't enough points on the diagram, no point in proceeding)
    if (len(lifespanH0) == 0):
        returnHelper(logMsg="Diagram failed because a lack of H0 points.")

        return 0,np.nan

    H0lifeMax = np.max(lifespanH0)

    birthsH1 = diagrams[1][:,0]
    lifespanH1 = np.diff(diagrams[1], axis=1).squeeze(axis=1)
    # Do length checks (if there aren't enough points on the diagram, no point in proceeding)
    if (len(lifespanH1) < 3):
        returnHelper(logMsg="Diagram failed because a lack of H1 points.")
        
        return 0,np.nan
    
    # compute persistence entropy
    persEntropy = persistent_entropy.persistent_entropy(diagrams,normalize=True)

    # compute DBSCAN clustering of the H1 points on the diagram
    normalH1s       = np.where(lifespanH1 >= 0)[0]#lifespanNoiseThr)[0]
    birthLifespanH1 = np.hstack((birthsH1[normalH1s].reshape(-1,1),lifespanH1[normalH1s].reshape(-1,1)))                

    
    if diagrams[1].shape[0] > 3:
        diagofdiag = ripser(birthLifespanH1,maxdim=0)['dgms']
        diagofdiag[0] = diagofdiag[0][:-1,:]
        H0ofH1life = diagofdiag[0][:,1]
    else:
        returnHelper(logMsg="Diagram failed because a lack of H1 points.")

        return 0,np.nan

    H0ofH1lifeQuants = np.array([np.quantile(H0ofH1life, q/100) for q in np.arange(1,101,1)])
    quantDiffs = np.diff(H0ofH1lifeQuants)
    dbscanEps_ = H0ofH1lifeQuants[np.argmax(quantDiffs)]

    diag_db     = DBSCAN(eps=dbscanEps_, min_samples=2).fit(birthLifespanH1)
    diag_labels = diag_db.labels_
    diag_coreInds = diag_db.core_sample_indices_

    if showLogs:
        if len(np.where(diag_labels != -1)[0]) > 0:
            print('Ratio of core samples (not counting outliers): ',len(diag_coreInds)/len(np.where(diag_labels != -1)[0]))
        else:
            print('Ratio of core samples not computed because no non-outlier points.')

    uniqueLabels = np.unique(diag_labels[diag_labels != -1])

    numClusters = len(uniqueLabels)
    if numClusters == 0:
        returnHelper(logMsg="Diagram failed because no clusters could be formed!")
        return 0,np.nan
    
    # Collect for each cluster how many member points it has
    nPerCluster = np.zeros(numClusters)
    for labeli in uniqueLabels:
        currPoints = diag_labels == labeli
        nPerCluster[labeli] = np.sum(currPoints)

    # Determine which cluster is the biggest (per number of members), and get the min max and median values for this cluster
    # This cluster is regularly "just noise", but its location on the diagram is of interest
    bigClustInd = np.argmax(nPerCluster)
    bigClustBirthLowerQ = np.quantile(birthLifespanH1[diag_labels == bigClustInd,0],.10)
    bigClustBirthUpperQ = np.quantile(birthLifespanH1[diag_labels == bigClustInd,0],.90)

    dbscanOutliers = birthLifespanH1[diag_labels == -1,:]
    dbscanNonOutliers = birthLifespanH1[diag_labels != -1,:]
    numOutliers = dbscanOutliers.shape[0]
    # If there are no outliers then its a failure
    if numOutliers == 0:
        returnHelper(logMsg="Diagram failed because there were no outliers.")

        return 0,np.nan

    # Turn outliers which have a low lifespan into clusterpoints
    outliers2use = dbscanOutliers[:,1] > 1.5*np.max(dbscanNonOutliers[:,1])
    if np.sum(outliers2use) == 0:
        returnHelper(logMsg="Diagram failed because no outliers with long enough lifespan!")
        
        return 0,np.nan

    dbscanNonOutliers = np.vstack((dbscanNonOutliers,dbscanOutliers[~outliers2use,:]))
    dbscanOutliers = dbscanOutliers[outliers2use,:]
    numOutliers = dbscanOutliers.shape[0]

    # Turn outliers which have a too high birth point into clusterpoints
    outliers2use = (dbscanOutliers[:,0] >= bigClustBirthLowerQ) & (dbscanOutliers[:,0] <= bigClustBirthUpperQ)
    if np.sum(outliers2use) == 0:
        returnHelper(logMsg="Diagram failed because no outliers with good birth values!")
        
        return 0,np.nan

    dbscanNonOutliers = np.vstack((dbscanNonOutliers,dbscanOutliers[~outliers2use,:]))
    dbscanOutliers = dbscanOutliers[outliers2use,:]
    numOutliers = dbscanOutliers.shape[0]

    # Check number of outliers, given the window length the maximum sensible number of outliers can be determined
        #       (i.e. loops corresponding to variable amplitude breaths)
    if numOutliers > (timeWinLen * (upperBPMthr/60)):
        returnHelper(logMsg="Diagram failed because of too many outlier points.")
        
        return 0,np.nan

    outliersBirthDiff2H0max = np.abs(dbscanOutliers[:,0] - H0lifeMax)

    # Check if the outliers' birth point is overlapping with the big cluster's births
    # because for heartbeats the embedding is most often not just a simple loop, but has smaller subloops, H0 max isnt always the best indicator for where the H1 max should be
    # it more often overlaps the big noise clusters births, often also right around its middle (median/mean)
    outliersWithBirthOverlap = np.full(numOutliers, False)
    for outlInd in range(numOutliers):
        outliersWithBirthOverlap[outlInd] = (bigClustBirthLowerQ <= dbscanOutliers[outlInd,0]) and (dbscanOutliers[outlInd,0] <= bigClustBirthUpperQ)

    dbscanOutliersMax     = dbscanOutliers[np.argmax(dbscanOutliers[:,1]),:]
    dbscanOutlierMax_birthDiff2H0max = np.abs(dbscanOutliersMax[0] - H0lifeMax) / H0lifeMax

    # Compute the score which the bin gets if it passes the checkpoints
    persDiagScore = -np.log( persEntropy[1] * (1 / dbscanOutliersMax[1]) * (H0lifeMax / dbscanOutliersMax[1]) * \
                             (np.max(birthLifespanH1[diag_labels != -1,1]) / dbscanOutliersMax[1]) * \
                             dbscanOutlierMax_birthDiff2H0max * H0lifeMax )
    
    class2Checks = np.vstack((
        (dbscanOutliers[:,1] > 2*H0lifeMax),
        outliersWithBirthOverlap | (outliersBirthDiff2H0max < .025),
        (dbscanOutliers[:,1] > 2*np.max(dbscanNonOutliers[:,1]))
    ))

    class1Checks = np.vstack((
        (dbscanOutliers[:,1] > np.quantile(lifespanH0, .95)),
        outliersWithBirthOverlap,
        (dbscanOutliers[:,1] > 1.5*np.max(dbscanNonOutliers[:,1]))
    ))
    
    class2PassedPerOutlier = np.sum(class2Checks, axis=0)
    class1PassedPerOutlier = np.sum(class1Checks, axis=0)

    if np.all(class2PassedPerOutlier == 3):
        returnHelper(logMsg=f"Diagram had passed with CAT2, score: {persDiagScore:.4e}, all outliers passed at least 3 of the class2 checks")
        if showLogs:
            print('num class2 checks passed per outlier: ',class2PassedPerOutlier)

        return 2,persDiagScore

    elif np.all(class1PassedPerOutlier == 3):
        returnHelper(logMsg=f"Diagram had passed with CAT1, score: {persDiagScore:.4e}, all outliers passed at least 3 of the class1 checks")
        if showLogs:
            print('num class1 checks passed per outlier: ',class1PassedPerOutlier)

        return 1,persDiagScore

    returnHelper(logMsg="Diagram not categorized, therefore failed!")

    return 0,np.nan
#######################################################################################################################
#######################################################################################################################

#######################################################################################################################

### function for analyzing persistence diagram from sublevel filtration (with only dim 0)

#######################################################################################################################
def subLevelFiltPersDgmChecker_DBSCAN(timeseries, timeWinLen, returnPoints=False, flipRerun=False, showLogs=False):
    """
    Classify a sublevel-filtration persistence diagram (dim=0) derived from a time series (breathing-optimized).

    Args:
        timeseries (np.ndarray): 1D array of samples for the window.
        timeWinLen (float): Window length in seconds; used to constrain expected number of events (6–30 BPM).
        returnPoints (bool): If True, return approximate lower/upper point-of-interest (POI) time indices.
        flipRerun (bool): Internal flag; if True, indicates this is the flipped re-run of the same timeseries.
        showLogs (bool): If True, print intermediate diagnostics.

    Returns:
        dict: With keys:

            - 'class' (int): 0=failure, 1=possibly flawed, 2=good.
            - 'score' (float): Score derived from normalized persistent entropy; np.nan if failure.

            - 'lowerPOItimeinds' (np.ndarray): Indices of minima (lower POIs) including the global minimum as first element (if returnPoints=True).
            - 'upperPOItimeinds' (np.ndarray): Indices of matched upper POIs (if returnPoints=True).

            - 'lowClustBirth' (str): 'low' | 'high' | 'mixed' | 'none' indicating the background cluster birth region (if returnPoints=True).

    Method:

        - Optionally add tiny noise when plateaus exist to avoid degenerate peak detection (when returnPoints=True).
        - Compute H0 diagram via uS.doSubLVLfiltration(timeseries, delInf=True).

        - If the diagram concentrates in the lower-right region, flip the signal and re-run once.
        - Quick pass case: all births low, lifespans high.

        - Else, compute DBSCAN parameters from the diagram-of-diagram (H0 of H0 lifespans), fit DBSCAN, and analyze clusters:
            - Identify highest-life cluster and ensure it is sufficiently separated and has a plausible count (6–30 BPM).

            - If valid, set class and score; optionally map diagram POIs back to time indices.

    Raises:
        Exception: If timeseries is None.

    Notes:

        - 'flipRerun' is handled internally to avoid infinite recursion.
        - Uses heuristics tuned for breathing (BPM range ~6–30).

    """
    def returnHelper(showLogs=showLogs,logMsg='Empty log'):
        if showLogs:
            print(logMsg)
            print('-----------------------------------------------')

        return
    
    def subLvlFiltPeakDet(returDict,dgmPOIs,lowClustBirths,timeseries):
        numPOIs = dgmPOIs.shape[0]
        lowerPOItimeinds = np.zeros(numPOIs+1,dtype=int)
        upperPOItimeinds = np.zeros(numPOIs,dtype=int)

        lowerPOItimeinds[0] = np.argmin(timeseries)
        for i in range(numPOIs):
            lowerPOItimeinds[i+1] = np.argmin(np.abs(dgmPOIs[i,0] - timeseries))
            upperPOItimeinds[i] = np.argmin(np.abs(dgmPOIs[i,1] - timeseries))

        lowerPOItimeinds = np.sort(lowerPOItimeinds)
        upperPOItimeinds = np.sort(upperPOItimeinds)
        returDict['lowerPOItimeinds'] = lowerPOItimeinds
        returDict['upperPOItimeinds'] = upperPOItimeinds

        if lowClustBirths is None:
            returDict['lowClustBirth'] = 'none'
        else:
            timeseriesRange = np.max(timeseries) - np.min(timeseries)
            if np.median(lowClustBirths) < (np.min(timeseries) + .33*timeseriesRange):
                returDict['lowClustBirth'] = 'low'
            elif np.median(lowClustBirths) > (np.min(timeseries) + .66*timeseriesRange):
                returDict['lowClustBirth'] = 'high'
            else:
                returDict['lowClustBirth'] = 'mixed'

        return returDict
    
    lowerBPMthr = 6
    upperBPMthr = 30


    returDict = {
        'class': 0,
        'score': np.nan,
        'lowerPOItimeinds': np.array([]),
        'upperPOItimeinds': np.array([]),
        'lowClustBirth': ''
    }

    # The return values are: the assigned category, and the score given
    # The category assigned to the input diagram can be:
    # 0 ... failure
    # 1 ... possibly flawed in some way
    # 2 ... good

    if timeseries is not None:
        # Test whether all points are unique (if sensors get overloaded then they can plateu on a given value --> messes up the peak detection principle)
        if returnPoints:
            if len(np.unique(timeseries)) < len(timeseries):
                timeseries = timeseries + .00001*np.random.randn(len(timeseries))

        dgm0 = hF.doSubLVLfiltration(timeseries, delInf=True, smallLifeThr=1e-3)

        timeseriesRange = np.max(timeseries) - np.min(timeseries)

        if flipRerun:
            signalFloor = np.min(timeseries) * -1
        else:
            signalFloor = np.min(timeseries)

    else:
        raise Exception("The breathing sublevel filtration diagram checker needs a timeseries!")

    if dgm0.shape[0] == 0:
        returnHelper(logMsg="Diagram failed because a lack of H0 points.")

        return returDict

    # extract the info from the input diagram list for easy access
    if np.isinf(dgm0[-1,1]):
        dgm0 = dgm0[:-1,:]
    
    birthsH0 = dgm0[:,0]
    lifespanH0 = np.diff(dgm0, axis=1).squeeze(axis=1)

    # Do length checks (if there aren't enough points on the diagram, no point in proceeding)
    if (len(lifespanH0) == 0):
        returnHelper(logMsg="Diagram failed because a lack of H0 points.")

        return returDict
    
    # compute persistence entropy
    persEntropy = persistent_entropy.persistent_entropy(dgm0,normalize=True)

    persDiagScore = copy.deepcopy(persEntropy[0])

    # compute DBSCAN clustering of the H1 points on the diagram
    normalH0s       = np.where(lifespanH0 >= 0)[0]
    birthLifespanH0 = np.hstack((birthsH0[normalH0s].reshape(-1,1),lifespanH0[normalH0s].reshape(-1,1)))

    if (not flipRerun) and (timeseries is not None):
        # criterion for flipping signal: every point is concentrated in the lower right corner of the diagram
        # what that means is that there is one downward deflection in the window, otherwise the signal is in the upper regions of the yaxis
        if (np.quantile(birthLifespanH0[:,0],.05) > (signalFloor + .5*timeseriesRange)) and np.all(birthLifespanH0[:,1] < (signalFloor + .3*timeseriesRange)):

            returDict = subLevelFiltPersDgmChecker_DBSCAN(-1*copy.deepcopy(timeseries),timeWinLen,flipRerun=True,returnPoints=returnPoints,showLogs=showLogs)
            if returnPoints:
                if returDict['lowClustBirth'] == "high":
                    returDict['lowClustBirth'] == "low"
                elif returDict['lowClustBirth'] == "low":
                    returDict['lowClustBirth'] == "high"
                
                flippedLowerInds = copy.deepcopy(returDict['lowerPOItimeinds'])
                flippedUpperInds = copy.deepcopy(returDict['upperPOItimeinds'])
                returDict['lowerPOItimeinds'] = flippedUpperInds
                returDict['upperPOItimeinds'] = flippedLowerInds

            return returDict

    # Handle the case when there are only clean large peaks in the signal
    if np.all(birthLifespanH0[:,0] < (signalFloor + .15*timeseriesRange)) and np.all(birthLifespanH0[:,1] > (signalFloor + .75*timeseriesRange)):
        returnHelper(logMsg=f"Diagram had passed with CAT2, score: {persDiagScore:.4e}")
            
        returDict['class'] = 2
        returDict['score'] = persDiagScore

        if (timeseries is not None) and returnPoints and (returDict['class'] in [1,2]):
            returDict = subLvlFiltPeakDet(returDict,dgm0,lowClustBirths=None,timeseries=timeseries)
        
        return returDict

    if dgm0.shape[0] >= 3:
        diagofdiag = ripser(birthLifespanH0,maxdim=0)['dgms']
        diagofdiag[0] = diagofdiag[0][:-1,:]
        H0ofH0life = diagofdiag[0][:,1]
    else:
        returnHelper(logMsg="Diagram failed because a lack of points.")

        return returDict

    quants = np.arange(1,101)/100
    H0ofH0lifeQuants = np.array([np.quantile(H0ofH0life, qi) for qi in quants])
    maxDiffGap = np.argmax(np.diff(H0ofH0lifeQuants))
    if showLogs:
        print('The max diff gap was at quantile: ',quants[maxDiffGap])
        
    dbscanEps_ = H0ofH0lifeQuants[maxDiffGap]

    if dbscanEps_ == 0:
        returnHelper(showPlots=False,logMsg="Diagram failed because computed DBSCAN eps parameter is 0")

        return returDict

    dbscanMinSamp_ = np.max((1,int(timeWinLen * (lowerBPMthr/60))))
    if showLogs:
        print('This is eps: ', dbscanEps_)
        print('This is min samp: ', dbscanMinSamp_)

    diag_db     = DBSCAN(eps=dbscanEps_, min_samples=dbscanMinSamp_).fit(birthLifespanH0)
    diag_labels = diag_db.labels_
    uniqueLabels = np.unique(diag_labels[diag_labels != -1])

    numClusters = len(uniqueLabels)

    if numClusters == 0:
        returnHelper(logMsg="Diagram failed because no clusters could be formed!")
        return returDict

    nPerClust = np.array([np.sum(diag_labels == lab) for lab in uniqueLabels])

    # collect the min-max of the clusters and their average
    clustersLifeMinMax = np.full((numClusters,2),np.nan)
    clustersBirthMinMax = np.full((numClusters,2),np.nan)
    clustersLifeMinMaxCoords = np.full((numClusters,4),np.nan)
    clustersBirthMinMaxCoords = np.full((numClusters,4),np.nan)
    clustersLifeAvg    = np.zeros(numClusters)
    for i,clust in enumerate(uniqueLabels):
        currClustPoints = birthLifespanH0[diag_labels == clust,:]

        clustersLifeMinMax[i,:] = [np.min(currClustPoints[:,1]), np.max(currClustPoints[:,1])]

        clustersBirthMinMax[i,:] = [np.min(currClustPoints[:,0]), np.max(currClustPoints[:,0])]

        clustersLifeMinMaxCoords[i,:2] = [currClustPoints[np.argmin(currClustPoints[:,1]),0], currClustPoints[np.argmin(currClustPoints[:,1]),1]]
        clustersLifeMinMaxCoords[i,2:] = [currClustPoints[np.argmax(currClustPoints[:,1]),0], currClustPoints[np.argmax(currClustPoints[:,1]),1]]

        clustersBirthMinMaxCoords[i,:2] = [currClustPoints[np.argmin(currClustPoints[:,0]),0], currClustPoints[np.argmin(currClustPoints[:,0]),1]]
        clustersBirthMinMaxCoords[i,2:] = [currClustPoints[np.argmax(currClustPoints[:,0]),0], currClustPoints[np.argmax(currClustPoints[:,0]),1]]
        
        clustersLifeAvg[i]    = np.mean(birthLifespanH0[diag_labels == clust,1])

    # check whether there are singleton clusters in the middle, if yes designated them as bad clusters
    badClusters = []
    for clusti in range(numClusters):
        if nPerClust[clusti] == 1:
            otherClusts = np.arange(numClusters)
            otherClusts = otherClusts[otherClusts != clusti]
            if np.any(clustersLifeAvg[otherClusts] > clustersLifeAvg[clusti]):
                badClusters.append(clusti)

    badClusters = np.array(badClusters)
    numBadClusters = len(badClusters)

    if numBadClusters > 1:
        returnHelper(logMsg="Diagram failed because too many bad singleton clusters!")
        return returDict

    highestClust = np.argmax(clustersLifeMinMax[:,1])

    if (numClusters - numBadClusters) == 1:
        if (clustersLifeMinMax[highestClust,0] > (signalFloor + .66*timeseriesRange)) and ((nPerClust[highestClust]+1) > (timeWinLen * (lowerBPMthr/60))) and ((nPerClust[highestClust]+1) < (timeWinLen * (upperBPMthr/60))):
        
            if (clustersLifeAvg[highestClust] > (signalFloor + .75*timeseriesRange)):
                returnHelper(logMsg=f"Diagram had passed with CAT2, score: {persDiagScore:.4e}")
                    
                returDict['class'] = 2
                returDict['score'] = persDiagScore

            else:
                returnHelper(logMsg=f"Diagram had passed with CAT1, score: {persDiagScore:.4e}")
                    
                returDict['class'] = 1
                returDict['score'] = persDiagScore
                
            if (timeseries is not None) and returnPoints and (returDict['class'] in [1,2]):
                POIs = np.where(diag_labels == uniqueLabels[highestClust])[0]
                dgmPOIs = dgm0[POIs,:]
                
                returDict = subLvlFiltPeakDet(returDict,dgmPOIs,lowClustBirths=None,timeseries=timeseries)
            
            return returDict
        
        else:
            returnHelper(logMsg="Diagram failed because only 1 cluster could be formed, and it is representing noise!")
            return returDict
    
    if (nPerClust[highestClust] == 1):
        if (clustersLifeAvg[highestClust] > (signalFloor + .8*timeseriesRange)) and (not flipRerun) and (timeseries is not None):
        
            returDict = subLevelFiltPersDgmChecker_DBSCAN(-1*copy.deepcopy(timeseries),timeWinLen,flipRerun=True,returnPoints=returnPoints,showLogs=showLogs)
            if returnPoints:
                if returDict['lowClustBirth'] == "high":
                    returDict['lowClustBirth'] == "low"
                elif returDict['lowClustBirth'] == "low":
                    returDict['lowClustBirth'] == "high"
                
                flippedLowerInds = copy.deepcopy(returDict['lowerPOItimeinds'])
                flippedUpperInds = copy.deepcopy(returDict['upperPOItimeinds'])
                returDict['lowerPOItimeinds'] = flippedUpperInds
                returDict['upperPOItimeinds'] = flippedLowerInds

            return returDict
        
        else:
            returnHelper(logMsg="Fail, highest cluster singular, but not high enough!")
            return returDict

    otherClusts = np.arange(numClusters)
    otherClusts = otherClusts[otherClusts != highestClust]
    otherClusts = np.setdiff1d(otherClusts,badClusters)

    if np.all(clustersLifeMinMax[highestClust,0] >= clustersLifeMinMax[otherClusts,1]) and np.all(clustersLifeAvg[highestClust] > 2*clustersLifeMinMax[otherClusts,1]) and ((nPerClust[highestClust]+1) > (timeWinLen * (lowerBPMthr/60))) and ((nPerClust[highestClust]+1) < (timeWinLen * (upperBPMthr/60))):
        if np.all(clustersLifeAvg[highestClust] > 2*clustersLifeMinMax[otherClusts,1]):
            returnHelper(logMsg=f"Diagram had passed with CAT2, score: {persDiagScore:.4e}")

            returDict['class'] = 2
            returDict['score'] = persDiagScore

        else:
            returnHelper(logMsg=f"Diagram had passed with CAT1, score: {persDiagScore:.4e}")

            returDict['class'] = 1
            returDict['score'] = persDiagScore

        if (timeseries is not None) and returnPoints and (returDict['class'] in [1,2]):
            POIs = np.where(diag_labels == uniqueLabels[highestClust])[0]
            dgmPOIs = dgm0[POIs,:]
            # Before deciding which is the "lowcluster" or noise cluster check whether there is overlap
            lowestClust = uniqueLabels[np.argmin(clustersLifeAvg)]
            if np.sum(clustersLifeMinMax[np.argmin(clustersLifeAvg),1] > clustersLifeMinMax[:,0]) > 1:
                overlappers = np.where(clustersLifeMinMax[np.argmin(clustersLifeAvg),1] > clustersLifeMinMax[:,0])[0]
                largestN = overlappers[np.argmax(nPerClust[overlappers])]
                lowestClust = uniqueLabels[largestN]

            lowClustBirths = dgm0[diag_labels == lowestClust,0]
            returDict = subLvlFiltPeakDet(returDict,dgmPOIs,lowClustBirths,timeseries)

        if returDict['class'] not in [1,2]:
            returnHelper(logMsg=f"Diagram failed, almost good but the high cluster not separated enough!")

        return returDict
        
    returnHelper(logMsg=f"Diagram failed, either too many points or no good clusters!")

    return returDict

#######################################################################################################################
#######################################################################################################################


#######################################################################################################################

### heartbeat optimized function for analyzing persistence diagram from sublevel filtration (with only dim 0)

#######################################################################################################################
def subLevelFiltPersDgmChecker_DBSCAN_HR(timeseries, timeWinLen, returnPoints=False, showLogs=False):
    """
    Classify a sublevel-filtration persistence diagram (dim=0) from a time series (heartbeat-optimized).

    Args:
        timeseries (np.ndarray): 1D array of samples for the window.
        timeWinLen (float): Window length in seconds; used to constrain expected number of beats (40–150 BPM).
        returnPoints (bool): If True, return approximate lower/upper point-of-interest (POI) time indices.
        showLogs (bool): If True, print intermediate diagnostics.

    Returns:
        dict: With keys:

            - 'class' (int): 0=failure, 1=possibly flawed, 2=good.
            - 'score' (float): Score derived from normalized persistent entropy; np.nan if failure.

            - 'lowerPOItimeinds' (np.ndarray): Indices of minima (lower POIs) including the global minimum as first element (if returnPoints=True).
            - 'upperPOItimeinds' (np.ndarray): Indices of matched upper POIs (if returnPoints=True).

            - 'lowClustBirth' (str): 'low' | 'high' | 'mixed' | 'none' indicating the background cluster birth region (if returnPoints=True).

    Method:

        - Compute H0 diagram via uS.doSubLVLfiltration(timeseries, delInf=True).
        - Use fixed DBSCAN eps (20% of signal range) and min_samples tied to 40 BPM.

        - Identify clusters and validate the highest-life cluster separation and plausible count (40–150 BPM).
        - If valid, set class/score; optionally map diagram POIs back to time indices.

    Raises:
        Exception: If timeseries is None.

    Notes:

        - Heuristics tuned for heartbeat; for breathing use subLevelFiltPersDgmChecker_DBSCAN.

    """
    def returnHelper(showLogs=showLogs,logMsg='Empty log'):
        if showLogs:
            print(logMsg)
            print('-----------------------------------------------')

        return
    
    def subLvlFiltPeakDet(returDict,dgmPOIs,lowClustBirths,timeseries):
        numPOIs = dgmPOIs.shape[0]
        lowerPOItimeinds = np.zeros(numPOIs+1,dtype=int)
        upperPOItimeinds = np.zeros(numPOIs,dtype=int)

        lowerPOItimeinds[0] = np.argmin(timeseries)
        for i in range(numPOIs):
            lowerPOItimeinds[i+1] = np.argmin(np.abs(dgmPOIs[i,0] - timeseries))
            upperPOItimeinds[i] = np.argmin(np.abs(dgmPOIs[i,1] - timeseries))

        lowerPOItimeinds = np.sort(lowerPOItimeinds)
        upperPOItimeinds = np.sort(upperPOItimeinds)
        returDict['lowerPOItimeinds'] = lowerPOItimeinds
        returDict['upperPOItimeinds'] = upperPOItimeinds

        if lowClustBirths is None:
            returDict['lowClustBirth'] = 'none'
        else:
            timeseriesRange = np.max(timeseries) - np.min(timeseries)
            if np.median(lowClustBirths) < (np.min(timeseries) + .33*timeseriesRange):
                returDict['lowClustBirth'] = 'low'
            elif np.median(lowClustBirths) > (np.min(timeseries) + .66*timeseriesRange):
                returDict['lowClustBirth'] = 'high'
            else:
                returDict['lowClustBirth'] = 'mixed'

        return returDict
    
    returDict = {
        'class': 0,
        'score': np.nan,
        'lowerPOItimeinds': np.array([]),
        'upperPOItimeinds': np.array([]),
        'lowClustBirth': ''
    }

    # The return values are: the assigned category, and the score given
    # The category assigned to the input diagram can be:
    # 0 ... failure
    # 1 ... possibly flawed in some way
    # 2 ... good

    if timeseries is not None:
        # Test whether all points are unique (if sensors get overloaded then they can plateu on a given value --> messes up the peak detection principle)
        if returnPoints:
            if len(np.unique(timeseries)) < len(timeseries):
                timeseries = timeseries + .00001*np.random.randn(len(timeseries))

        dgm0 = hF.doSubLVLfiltration(timeseries, delInf=True, smallLifeThr=1e-3)

        timeseriesRange = np.max(timeseries) - np.min(timeseries)

    else:
        raise Exception("The heartbeat sublevel filtration diagram checker needs a timeseries!")

    if dgm0.shape[0] == 0:
        returnHelper(logMsg="Diagram failed because a lack of H0 points.")

        return returDict

    # extract the info from the input diagram list for easy access
    if np.isinf(dgm0[-1,1]):
        dgm0 = dgm0[:-1,:]
    
    birthsH0 = dgm0[:,0]
    lifespanH0 = np.diff(dgm0, axis=1).squeeze(axis=1)

    # Do length checks (if there aren't enough points on the diagram, no point in proceeding)
    if (len(lifespanH0) == 0):
        returnHelper(logMsg="Diagram failed because a lack of H0 points.")

        return returDict
    
    # compute persistence entropy
    persEntropy = persistent_entropy.persistent_entropy(dgm0,normalize=True)

    persDiagScore = copy.deepcopy(persEntropy[0])

    # compute DBSCAN clustering of the H1 points on the diagram
    normalH0s       = np.where(lifespanH0 >= 0)[0]
    birthLifespanH0 = np.hstack((birthsH0[normalH0s].reshape(-1,1),lifespanH0[normalH0s].reshape(-1,1)))

    dbscanEps_ = .20*timeseriesRange

    dbscanMinSamp_ = np.max((1,int(timeWinLen * (40/60))))
    if showLogs:
        print('This is eps: ', dbscanEps_)
        print('This is min samp: ', dbscanMinSamp_)

    diag_db     = DBSCAN(eps=dbscanEps_, min_samples=dbscanMinSamp_).fit(birthLifespanH0)
    diag_labels = diag_db.labels_
    uniqueLabels = np.unique(diag_labels[diag_labels != -1])

    signalFloor = np.min(timeseries)

    numClusters = len(uniqueLabels)

    if numClusters == 0:
        returnHelper(logMsg="Diagram failed because no clusters could be formed!")
        return returDict

    nPerClust = np.array([np.sum(diag_labels == lab) for lab in uniqueLabels])

    # collect the min-max of the clusters and their average
    clustersLifeMinMax = np.full((numClusters,2),np.nan)
    clustersBirthMinMax = np.full((numClusters,2),np.nan)
    clustersLifeMinMaxCoords = np.full((numClusters,4),np.nan)
    clustersBirthMinMaxCoords = np.full((numClusters,4),np.nan)
    clustersLifeAvg    = np.zeros(numClusters)
    for i,clust in enumerate(uniqueLabels):
        currClustPoints = birthLifespanH0[diag_labels == clust,:]

        clustersLifeMinMax[i,:] = [np.min(currClustPoints[:,1]), np.max(currClustPoints[:,1])]

        clustersBirthMinMax[i,:] = [np.min(currClustPoints[:,0]), np.max(currClustPoints[:,0])]

        clustersLifeMinMaxCoords[i,:2] = [currClustPoints[np.argmin(currClustPoints[:,1]),0], currClustPoints[np.argmin(currClustPoints[:,1]),1]]
        clustersLifeMinMaxCoords[i,2:] = [currClustPoints[np.argmax(currClustPoints[:,1]),0], currClustPoints[np.argmax(currClustPoints[:,1]),1]]

        clustersBirthMinMaxCoords[i,:2] = [currClustPoints[np.argmin(currClustPoints[:,0]),0], currClustPoints[np.argmin(currClustPoints[:,0]),1]]
        clustersBirthMinMaxCoords[i,2:] = [currClustPoints[np.argmax(currClustPoints[:,0]),0], currClustPoints[np.argmax(currClustPoints[:,0]),1]]
        
        clustersLifeAvg[i]    = np.mean(birthLifespanH0[diag_labels == clust,1])


    if numClusters == 1:
        if (clustersLifeMinMax[0,0] > (signalFloor + .3*timeseriesRange)) and ((nPerClust[0]+1) > (timeWinLen * (40/60))) and ((nPerClust[0]+1) < (timeWinLen * (150/60))):
        
            returnHelper(logMsg=f"Diagram had passed with CAT2, score: {persDiagScore:.4e}")
                
            returDict['class'] = 2
            returDict['score'] = persDiagScore
            
            if (timeseries is not None) and returnPoints:
                POIs = np.where(diag_labels == uniqueLabels[0])[0]
                dgmPOIs = dgm0[POIs,:]
                
                returDict = subLvlFiltPeakDet(returDict,dgmPOIs,lowClustBirths=None,timeseries=timeseries)
            
            return returDict
        
        else:
            returnHelper(logMsg="Diagram failed because only 1 cluster could be formed, and it is representing noise!")
            return returDict
    
    highestClust = np.argmax(clustersLifeAvg)

    if (np.sum(clustersLifeMinMax[highestClust,0] <= clustersLifeMinMax[:,1]) == 1) and ((nPerClust[highestClust]+1) > (timeWinLen * (40/60))) and ((nPerClust[highestClust]+1) < (timeWinLen * (150/60))):
    
        returnHelper(logMsg=f"Diagram had passed with CAT2, score: {persDiagScore:.4e}")

        returDict['class'] = 2
        returDict['score'] = persDiagScore

        if (timeseries is not None) and returnPoints:
            POIs = np.where(diag_labels == uniqueLabels[highestClust])[0]
            dgmPOIs = dgm0[POIs,:]
            # Before deciding which is the "lowcluster" or noise cluster check whether there is overlap
            lowestClust = uniqueLabels[np.argmin(clustersLifeAvg)]
            if np.sum(clustersLifeMinMax[np.argmin(clustersLifeAvg),1] > clustersLifeMinMax[:,0]) > 1:
                overlappers = np.where(clustersLifeMinMax[np.argmin(clustersLifeAvg),1] > clustersLifeMinMax[:,0])[0]
                largestN = overlappers[np.argmax(nPerClust[overlappers])]
                lowestClust = uniqueLabels[largestN]

            lowClustBirths = dgm0[diag_labels == lowestClust,0]
            returDict = subLvlFiltPeakDet(returDict,dgmPOIs,lowClustBirths,timeseries)

        return returDict
        
    returnHelper(logMsg=f"Diagram failed, either too many points or no good clusters!")

    return returDict

#######################################################################################################################
#######################################################################################################################