'''
Script containing all the functions which are used to extract features from data

I have removed all of the stats and functions that will not be needed of used for model building
to improve the speed of the script
'''


# Import functions
import numpy as np
import scipy
from scipy import stats
from scipy.stats import kstest
import statsmodels
from statsmodels import stats
from statsmodels.stats import diagnostic


'''
Feature Extractions functions
'''

#how many values are above/below the mean
def globalStats(arr):
    """Global Statistics of an array"""
    arrMedian = np.median(arr)
    arrMean = arr.mean()
    nPosCount = arr[arr > arrMean].size
    nNegCount = arr[arr < arrMean].size #useful as some RFI have a lot of values below the 'baseline'
    nPosPct = nPosCount / float(arr.size)
    nNegPct = nNegCount / float(arr.size)
    std = arr.std()

        
    if np.isclose(arrMedian, 0.): meanMedianRatio = 0.
    else: meanMedianRatio = np.abs(arrMean / arrMedian)
    #return a dictionary full of statistics
    return { 'mean': arrMean, 'median': arrMedian, 'std': std, 'min': arr.min(), 'max': arr.max(),
             'meanMedianRatio': meanMedianRatio, 'maxMinRatio': np.abs(arr.max() / arr.min()),
             'posCount': nPosCount, 'negCount': nNegCount, 'posPct': nPosPct, 'negPct': nNegPct}



def GaussianTests(arr):
    """Test for values that indicate how Gaussian the data is"""
    kurtosis = scipy.stats.kurtosis(arr)
    skew = scipy.stats.skew(arr)
     
    #KS test uses 0 mean and 1 variance, data is already roughly centered around 0, so need to recenter distribution and make variance 1. Is this not using a circular argument as i must assume normal dist to calculate variance??
    arrnorm = (arr-arr.mean())/arr.std()
    ks = scipy.stats.kstest(arrnorm,'norm')
                              
    return { 'kurtosis': kurtosis, 'skew': skew, 'ks': ks  }
     
    
    
def windowedStats(arr, nseg=16):
    """Statistics on segments of an array"""
    #splits array into nseg segments and creates empty arrays for each value, each array has nseg elements
    segSize = arr.shape[0] / nseg #how many elements in each segment
    minVals = np.zeros(nseg)
    maxVals = np.zeros(nseg)
    meanVals = np.zeros(nseg)
    stdVals = np.zeros(nseg)
    snrVals = np.zeros(nseg)

    #takes sidth segment and assigns value for that segment to sidth element of value array
    #put KS testing in here too?
    for sid in np.arange(nseg):
        minVals[sid] = arr[segSize*sid:segSize*(sid+1)].min()
        maxVals[sid] = arr[segSize*sid:segSize*(sid+1)].max()
        meanVals[sid] = arr[segSize*sid:segSize*(sid+1)].mean()
        stdVals[sid] = np.std(arr[segSize*sid:segSize*(sid+1)])
        if np.isclose(stdVals[sid], 0): snrVals[sid] = 0.
        else: snrVals[sid] = maxVals[sid] / stdVals[sid]
        
    return { 'min': minVals, 'max': maxVals, 'mean': meanVals, 'std': stdVals, 'snr': snrVals  }


 
def countValOverflows(arr, threshold=1e20):
    """Return a count of the number of values which are above a given threshold"""
    nCount = arr[np.abs(arr)>threshold].size
    
    return { 'ncount': nCount, 'pct': nCount / float(arr.size) }



def pixelizeSpectrogram(arr, nTime=16, nChan=4):
    """Coarsely pixelize a spectrogram"""
    timeSize = arr.shape[0] / nTime
    chanSize = arr.shape[1] / nChan
    #empty value arrays
    minVals = np.zeros((nTime, nChan))
    maxVals = np.zeros((nTime, nChan))
    meanVals = np.zeros((nTime, nChan))

    #cycles over different nTime x nChan segments of arr and saves max/min/mean in tidth element of value arrays
    for tid in np.arange(nTime):
        for cid in np.arange(nChan):
            minVals[tid,cid] = arr[timeSize*tid:timeSize*(tid+1), chanSize*cid:chanSize*(cid+1)].min()
            maxVals[tid,cid] = arr[timeSize*tid:timeSize*(tid+1), chanSize*cid:chanSize*(cid+1)].max()
            meanVals[tid,cid] = arr[timeSize*tid:timeSize*(tid+1), chanSize*cid:chanSize*(cid+1)].mean()
                            
    return { 'min': minVals, 'max': maxVals, 'mean': meanVals }



def countPeaks(arr):
    """Count the number of points that exceed X*std centred on the median"""
    arrMedian = arr.median()
    std = arr.std()
    X = [2,3,5]
    for i in X:
        posThreshold[i] = arrMedian + i*std
        negThreshold[i] = arrMedian - i*std
        posPeaks[i] = arr[arr > posThreshold[i]].size
        negPeaks[i] = arr[arr < negThreshold[i]].size
    
    return {'posPeaks': posPeaks, 'negPeaks': negPeaks}
    
    
    