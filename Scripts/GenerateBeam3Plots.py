'''
Test plot generation
'''

# Import Functions
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import dedispersion
import filterbankio_headerless
import matplotlib
import sys, os, random
import pandas as pd
from io import StringIO
import FeatureExtractionFunctions as fef
import pickle

# Determine how many plots have been made
plotlist = os.listdir('TestPlots/Beam3Plots/')
NumberOfPlots = len(plotlist)

# Set directory from which to retrieve files
BASE_DATA_PATH = '/oxford_data/FRBSurvey/Archive/'

# get list of all data files
plotsData = [i for i in os.listdir(BASE_DATA_PATH) if i.endswith('.fil')]
print(len(plotsData), 'total fil files')
Beam3Data = []
saveDir = 'TestPlots/Beam3Plots/'

benchmarks = [i for i in range(0,100000,200)]

# Create lists of already existing plots
trainPlots1 = [i for i in os.listdir('TrainingPlotsUsed/MoreTrainingPlots/') if i.endswith('.png')]
trainPlots2 = [i for i in os.listdir('TrainingPlotsUsed/TrainingPlotsUpdate/') if i.endswith('.png')]
testPlots2 = [i for i in os.listdir('TestPlots/Tests2/') if i.endswith('.png')]
plotsCreated = trainPlots1 + trainPlots2 + testPlots2

# Create an empty dictionary in which we will save the dictionaries of features 
FeaturesDict = {}




'''
Select out only the beam 3 data from the archive 
'''

for fb_filename in plotsData:
    
    # get beamID from fb filename
    BeamID = int(fb_filename.split('Beam')[1][0])
    
    if BeamID == 3:
        Beam3Data.append(fb_filename)
    else:
        pass
        
# print how many blocks in Beam3
print(len(Beam3Data),'fil files found in Beam3')




'''
Retrieve the raw data from archive and create waterfall
'''
        
for fb_filename in Beam3Data:
    
    # Create empty dictionary in which to save features
    metaData = {}        
        
    # Name the filterbank and data file
    dm_filename = fb_filename[:6] + 'dm' + fb_filename[8:25] + '.dat'    
    lofarFil = BASE_DATA_PATH + fb_filename
    lofarDM = BASE_DATA_PATH + dm_filename
    BeamID = int(fb_filename.split('Beam')[1][0])
    
    # Check for errors with dat file
    try:
        content = open(lofarDM).read()
        content = content.split('# ------')[-1].split('Done')[:-1]
        NumberOfBlocks = len(content)
        
    except:
        print('Error with dat file for', dm_filename, 'choosing different file to plot')
        print(NumberOfPlots,'have been created')
        continue
    
    # run through all blocks in the file
    print(NumberOfBlocks,'blocks found in',dm_filename)
    blockList = list(range(1,NumberOfBlocks + 1))
    for BlockNumber in blockList:
            
        # Check that the plot for the selected block does not already exist
        saveName = dm_filename[:25] + '_block' + str(BlockNumber) + '.png'
        savePath = saveDir + saveName
    
        if saveName in plotlist:
            continue
        
        # Check that plot is not in original batch
        if saveName in plotsCreated:
            continue
        
        # Get the best dm and the bin factor from the .dat file
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
        
        
        # check for errors with fil file
        try:
            fil = filterbankio_headerless.Filterbank(lofarFil, BlockNumber)
            timeFactor = int(binFactor)
            freqFactor = 8
        
        except:
            print('Error with filterbank file for',fb_filename,'choosing different file to plot')
            print(NumberOfPlots,'have been created')
            continue
        
    
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
        metaData['DMstd'] = df['DM'].std()
        metaData['DMrange'] = df['DM'].max() - df['DM'].min()
        metaData['DMstdMaxratio'] = df['DM'].std()/df['DM'].max()
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
        This section retrieves and generatese the waterfall data for an individual block
        '''
        
        tInt = fil.my_header[5] # set tInt

        freqsHz = fil.freqs * 1e6 # generate array of freqs in Hz

        waterfall = np.reshape(fil.data, (fil.data.shape[0], fil.data.shape[2])) # reshape to (n integrations, n freqs)
    
    
        #waterfall shape must be divisible be time factor
        if waterfall.shape[0] % timeFactor == 0:
            waterfall = waterfall.reshape(int(waterfall.shape[0]/timeFactor), timeFactor, waterfall.shape[1]).sum(axis=1)
            tInt *= timeFactor
        else:
            #print('WARNING, %i time samples is not divisible by the time factor %i'%(waterfall.shape[0], timeFactor))
        
            #add extra dimensions so that waterfall shape is divisible by timeFactor
            zeros = np.zeros((timeFactor - (waterfall.shape[0] % timeFactor), waterfall.shape[1])) 
            waterfall = np.concatenate((waterfall, zeros))
        
            #sum elements in 1st dimension
            waterfall = waterfall.reshape(int(waterfall.shape[0]/timeFactor), timeFactor, waterfall.shape[1]).mean(axis=1) 
            tInt *= timeFactor
            
        dm = BestDM
    
    
        # Apply dedispersion 
        ddwaterfall = dedispersion.incoherent(freqsHz, waterfall, tInt, dm, boundary='wrap') # apply dedispersion

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

        timeSeries = timeSeries[startIdx:endIdx]
        ddTimeSeries = ddTimeSeries[startIdx:endIdx]
        waterfall = waterfall[startIdx:endIdx,:]
        ddwaterfall = ddwaterfall[startIdx:endIdx,:]

        normTimeSeries = timeSeries / (waterfall.shape[1] * timeFactor)
        normDDTimeSeries = ddTimeSeries / (waterfall.shape[1] * timeFactor)
    
        waterfallNorm = waterfall
    
    
    
    
        '''
        This section generates and saves the plots
        '''
    
        # Create the plot
        cmap = 'magma'

        fig = plt.figure(figsize=(8,8)) # (width, height)

        gs = matplotlib.gridspec.GridSpec(3,3)
        gs.update(hspace=0.0, wspace=0.0)
     
        # Dedispersed Frequency plot
        ax1 = plt.subplot(gs[0:2,0:3])
        imRawdd = plt.imshow(np.flipud(ddwaterfall.T), extent=(0, tInt*ddwaterfall.shape[0], fil.freqs[0], fil.freqs[-1]), aspect='auto',           cmap=plt.get_cmap(cmap), interpolation='nearest')
        plt.ylabel('Freq. (MHz)')
        plt.title('LOFAR Event (timeFactor: %i,  freqFactor: %i,  MaxSNR: %0.f,  DM: %0.f)'%(timeFactor, freqFactor, MaxSNR, dm), fontdict={'fontsize':'small'})
        ax1.get_xaxis().set_visible(False)
    
        # Time Series Plot
        ax2 = plt.subplot(gs[2:3,0:3])
        lineColor = 'k'
        plt.plot(tInt*np.arange(waterfall.shape[0]), normDDTimeSeries, lineColor, alpha=0.8)
        plt.xlim(0, tInt*timeSeries.shape[0])
        plt.text(0.02, 0.85, 'DM=%0.f'%dm, transform=ax2.transAxes)
        plt.xlabel('Time (s)')
        plt.ylabel('Flux (uncalibrated units)')
    
        plt.ioff()
        # Save plots to TrainingPlots directory
        plt.savefig(savePath)
        plt.close()
    
    
    
        ############################################
        # Save ddTimeSeries and metaData for Griffin
        outputDict = {'metaData' : metaData,
                  'ddTimeSeries' : ddTimeSeries }
    
        output = open('ddTimeSeriesData/' + saveName + '_ddts.pkl', 'wb' )
        pickle.dump(outputDict, output)
        output.close()
        ############################################
    
    
        '''
        This section extracts features from the data
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
        FeaturesDict[saveName] = metaData
    
    
        # Update plotlist and display how many plots created
        plotlist = os.listdir('TestPlots/Beam3Plots/')
        NumberOfPlots = len(plotlist)
        if NumberOfPlots == 10:
            print('10 plots created')
        if NumberOfPlots in benchmarks:
            print(NumberOfPlots, 'plots created')
    
    
    
    
'''
Convert the dictionary of features to a dataframe and save it as a pickle file
'''

print('Features extracted from',NumberOfPlots,'plots in Beam 3, saving data to pkl file')
  
FeaturesDF = pd.DataFrame(FeaturesDict)
FeaturesDF = FeaturesDF.transpose()
FeaturesDF = FeaturesDF.apply(pd.to_numeric)
FeaturesDF.to_pickle('DataFrames/Beam3FeatsDataFrame.pkl')

