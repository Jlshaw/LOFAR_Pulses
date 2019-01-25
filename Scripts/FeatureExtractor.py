"""
Feature Extraction Script
"""

# This script extracts features for plots which have already been created and are saved in TrainingPlots directory

# Import Functions
import sys,os
import numpy as np
import pickle
import dedispersion
import filterbankio_headerless
import FeatureExtractionFunctions as fef
import pandas as pd
from io import StringIO

# Set directory from which to retrieve raw data
BASE_DATA_PATH = '/oxford_data/FRBSurvey/Archive/'

# Get list of plots we want to get stats for 
plotlist = os.listdir('/home/jshaw/TrainingPlots/')
print('Extracting features from', len(plotlist), 'plots in TrainingPlots directory')

# Create an empty dictionary in which we will save the dictionaries of features 
FeaturesDict = {}

# Get a print out of how many plots have had features extracted
n = 0
benchmarks = [i for i in range(0,2000,25)]


'''
Main loop for going to raw data files and extracting features for each of the files in TrainingPlots directory
'''

for plot_filename in plotlist:
    
    # Create empty dictionary in which to save features
    metaData = {}
    
    # Get .fil and .dat filenames from plot filename along with block number
    dm_filename = plot_filename[:25] + '.dat'
    fb_filename = plot_filename[:6] + 'fb' + plot_filename[8:25] + '.fil'
    BlockNumber = int(plot_filename.split('block')[1].split('.png')[0])
    BeamID = int(plot_filename.split('Beam')[1][0])
    
    # Set the path to the .fil & .dat file
    lofarFil = BASE_DATA_PATH + fb_filename
    lofarDM = BASE_DATA_PATH + dm_filename
    
    # Get best dm and binfactor from .dat file
    content = open(lofarDM).read()
    content = content.split('# ------')[-1].split('Done')[:-1]
    for x in content:
        if content.index(x) == BlockNumber - 1:
            events = x.split('#')
            bufStr = events[-1]
            events = events[0] 
            df = pd.read_csv(StringIO(events), sep=',', names=['MJD', 'DM', 'SNR', 'BinFactor']).dropna()
            binFactor = df['BinFactor'].median()
            BestDM = float(bufStr.split('|')[2].split(' ')[3])
            MaxSNR = float(bufStr.split(':')[-1])
            mjd0 = bufStr.split(' ')[6]
        else:
            pass    
        
        
    # Add some of the relevant data to metaData dictionary
    metaData['Beam'] = BeamID
    metaData['Block'] = BlockNumber
    metaData['Events'] = len(df.index)
    metaData['MJDStart'] = mjd0
    metaData['binFactor'] = int(binFactor)
    metaData['BestDM'] = BestDM
    metaData['MaxSNR'] = MaxSNR
    metaData['DMmax'] = df['DM'].max()
    metaData['DMmin'] = df['DM'].min()
    metaData['DMmean'] = df['DM'].mean()
    metaData['DMmedian'] = df['DM'].median()
    metaData['SNRmean'] = df['SNR'].mean()
    metaData['SNRmin'] = df['SNR'].min()
    metaData['SNRmedian'] = df['SNR'].median()
    metaData['SNRstd'] = df['SNR'].std()
    metaData['MJDmax'] = df['MJD'].max()
    metaData['MJDmin'] = df['MJD'].min()
    metaData['MJDmean'] = df['MJD'].mean()
    metaData['MJDmedian'] = df['MJD'].median()
    metaData['MJDstd'] = df['MJD'].std()
    
    
        
    '''
    This section creates the waterfall, ddwaterfall, timeseries and ddtimeseries     
    '''
    
    # Read in data using Filterbank class
    fil = filterbankio_headerless.Filterbank(lofarFil, BlockNumber)
    timeFactor = int(binFactor)
    freqFactor = 8
    
    tInt = fil.my_header[5] # set tInt
    freqsHz = fil.freqs * 1e6 # generate array of freqs in Hz

    waterfall = np.reshape(fil.data, (fil.data.shape[0], fil.data.shape[2])) # reshape to (n integrations, n freqs)
    
    #waterfall shape must be divisible be time factor
    if waterfall.shape[0] % timeFactor == 0:
        waterfall = waterfall.reshape(int(waterfall.shape[0]/timeFactor), timeFactor, waterfall.shape[1]).sum(axis=1)
        tInt *= timeFactor
    else:
        #add extra dimensions so that waterfall shape is divisible by timeFactor
        zeros = np.zeros((timeFactor - (waterfall.shape[0] % timeFactor), waterfall.shape[1])) 
        waterfall = np.concatenate((waterfall, zeros))
        
        #sum elements in 1st dimension
        waterfall = waterfall.reshape(int(waterfall.shape[0]/timeFactor), timeFactor, waterfall.shape[1]).mean(axis=1) 
        tInt *= timeFactor
        
    dm = BestDM
    
    # Apply dedispersion 
    ddwaterfall = dedispersion.incoherent(freqsHz, waterfall, tInt, dm, boundary='wrap') 

    waterfall = waterfall.reshape(waterfall.shape[0], int(waterfall.shape[1]/freqFactor), freqFactor).sum(axis=2)
    ddwaterfall = ddwaterfall.reshape(ddwaterfall.shape[0], int(ddwaterfall.shape[1]/freqFactor), freqFactor).sum(axis=2)
    freqsHz = freqsHz[::freqFactor]
    
    # Take 8 seconds of data
    start_time = 0.0
    time_window = 8.

    if start_time is None:
        startIdx = 0
    else:
        startIdx = int(start_time / tInt)

    if time_window is None:
        endIdx = waterfall.shape[0]
    else:
        endIdx = startIdx + int(time_window / tInt)
        if endIdx > waterfall.shape[0]:
            endIdx = waterfall.shape[0]

    timeSeries = np.sum(waterfall, axis=1)
    ddTimeSeries = np.sum(ddwaterfall, axis=1)

    # Get the time series and waterfalls for dedispersed and non-dedispersed
    timeSeries = timeSeries[startIdx:endIdx]
    ddTimeSeries = ddTimeSeries[startIdx:endIdx]
    waterfall = waterfall[startIdx:endIdx,:]
    ddwaterfall = ddwaterfall[startIdx:endIdx,:]

    
    ############################################
    # Save ddTimeSeries and metaData for Griffin
    outputDict = {'metaData' : metaData,
                  'ddTimeSeries' : ddTimeSeries }
    
    output = open('ddTimeSeriesData/' + plot_filename + '_ddts.pkl', 'wb' )
    pickle.dump(outputDict, output)
    output.close()
    ############################################
    
    
    '''
    This section contains fucntions which extract the features and save the data to a dictionary
    '''      

    # Add globalTimeStats to dictionary
    globalTimeStats = fef.globalStats(timeSeries)
    metaData['globalTimeStatsmean'] = globalTimeStats['mean']
    metaData['globalTimeStatsmedian'] = globalTimeStats['median']
    metaData['globalTimeStatsstd'] = globalTimeStats['std']
    metaData['globalTimeStatsmin'] = globalTimeStats['min']
    metaData['globalTimeStatsmax'] = globalTimeStats['max']
    metaData['globalTimeStatsmeanMedianRatio'] = globalTimeStats['meanMedianRatio']
    metaData['globalTimeStatsminMaxRatio'] = globalTimeStats['maxMinRatio']
    metaData['globalTimeStatsposCount'] = globalTimeStats['posCount']
    metaData['globalTimeStatsnegCount'] = globalTimeStats['negCount']
    metaData['globalTimeStatsposPct'] = globalTimeStats['posPct']
    metaData['globalTimeStatsnegPct'] = globalTimeStats['negPct']
    
    # Add globalDedispTimeSeries to dictionary
    globalDedispTimeStats = fef.globalStats(ddTimeSeries)
    metaData['globalDedispTimeStatsmean'] = globalDedispTimeStats['mean']
    metaData['globalDedispTimeStatsmedian'] = globalDedispTimeStats['median']
    metaData['globalDedispTimeStatsstd'] = globalDedispTimeStats['std']
    metaData['globalDedispTimeStatsmin'] = globalDedispTimeStats['min']
    metaData['globalDedispTimeStatsmax'] = globalDedispTimeStats['max']
    metaData['globalDedispTimeStatsmeanMedianRatio'] = globalDedispTimeStats['meanMedianRatio']
    metaData['globalDedispTimeStatsminMaxRatio'] = globalDedispTimeStats['maxMinRatio']
    metaData['globalDedispTimeStatsposCount'] = globalDedispTimeStats['posCount']
    metaData['globalDedispTimeStatsnegCount'] = globalDedispTimeStats['negCount']
    metaData['globalDedispTimeStatsposPct'] = globalDedispTimeStats['posPct']
    metaData['globalDedispTimeStatsnegPct'] = globalDedispTimeStats['negPct']
    
    # Add overflows to dictionary
    overflows = fef.countValOverflows(waterfall)
    metaData['overflowCounts'] = overflows['ncount']
    metaData['overflowPct'] = overflows['pct']
    
    # Add windTimeSeries to dictionary
    windTimeStats = fef.windowedStats(timeSeries)
    for i in range(16):
        metaData['windTimeSeriesmean' + str(i)] = windTimeStats['mean'][i]
        metaData['windTimeSeriesmax' + str(i)] = windTimeStats['max'][i]
        metaData['windTimeSeriesmin' + str(i)] = windTimeStats['min'][i]
        metaData['windTimeSeriesstd' + str(i)] = windTimeStats['std'][i]
        metaData['windTimeSeriessnr' + str(i)] = windTimeStats['snr'][i]
    
    # Add windDedispTimeSeries to dictionary
    windDedispTimeStats = fef.windowedStats(ddTimeSeries)
    for i in range(16):
        metaData['windDedispTimeSeriesmean' + str(i)] = windDedispTimeStats['mean'][i]
        metaData['windDedispTimeSeriesmax' + str(i)] = windDedispTimeStats['max'][i]
        metaData['windDedispTimeSeriesmin' + str(i)] = windDedispTimeStats['min'][i]        
        metaData['windDedispTimeSeriesstd' + str(i)] = windDedispTimeStats['std'][i]
        metaData['windDedispTimeSeriessnr' + str(i)] = windDedispTimeStats['snr'][i]
    
    # Add pixels to dictionary
    pixels = fef.pixelizeSpectrogram(waterfall)
    for i in range(16):
        for j in range(4):
            metaData['pixelsmax_%i_%i'%(i,j)] = pixels['max'][i][j]
            metaData['pixelsmin_%i_%i'%(i,j)] = pixels['min'][i][j]
            metaData['pixelsmean_%i_%i'%(i,j)] = pixels['mean'][i][j]
            
    # Add peakCounts to dictionary
    nPeaks = fef.countPeaks(ddTimeSeries)
    X = [2,3,5]
    for i in X:
        x = X.index(i)
        metaData['posPeaks' + str(i) + 'std'] = nPeaks['posPeaks'][x]
        metaData['negPeaks' + str(i) + 'std'] = nPeaks['negPeaks'][x]
        
    # Add GaussianTests to dictionary
    GaussianTests = fef.GaussianTests(timeSeries)
    metaData['GaussianTestskurtosis'] = GaussianTests['kurtosis']
    metaData['GaussianTestsskew'] = GaussianTests['skew']
    metaData['GaussianTestsksD'] = GaussianTests['ks'][0]
    metaData['GaussianTestskspvalue'] = GaussianTests['ks'][1]
    
    
    # Save the metaData to the features dictionary with the plot filename
    FeaturesDict[plot_filename] = metaData
    
    # Print how many plots have had features extracted
    n += 1
    if n in benchmarks:
        print('Features extracted from', n, 'plots')
  

'''
Convert the dictionary of features to a dataframe and save it as a pickle file
'''
  
FeaturesDF = pd.DataFrame(FeaturesDict)
FeaturesDF = FeaturesDF.transpose()
FeaturesDF = FeaturesDF.apply(pd.to_numeric)
FeaturesDF.to_pickle('FeaturesDataFrame.pkl')


