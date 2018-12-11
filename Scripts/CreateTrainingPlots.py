'''
Training Plots Script
'''

# This script randomly selects 1000 (10 currently) blocks of data, plots them and saves the plots such that they can be inspected

# Import Functions
import numpy as np
import matplotlib.pyplot as plt
import dedispersion
import filterbankio_headerless
import matplotlib
import os, random
import pandas as pd
from io import StringIO

# Determine how many plots have been made
plotlist = os.listdir('TrainingPlots/')        
NumberOfPlots = len(plotlist) - 1

# Set directory from which to retrieve files
BASE_DATA_PATH = '/oxford_data/FRBSurvey/Archive/'



'''
Main loop for generating a certain number of plots 
'''

while NumberOfPlots < 1000:
    
    if NumberOfPlots == 400:
        print('400 plots created')
        
    if NumberOfPlots == 600:
        print('600 plots created')
        
    if NumberOfPlots == 800:
        print('800 plots created')
    
    # Randomly select a filterbank file from the base directory   
    randomFile = random.choice(os.listdir(BASE_DATA_PATH))
    
    while randomFile[-4:] == '.dat':
        randomFile = random.choice(os.listdir(BASE_DATA_PATH))
    else:
        fb_filename = randomFile
        
        
    # Name the filterbank and data file
    dm_filename = fb_filename[:6] + 'dm' + fb_filename[8:25] + '.dat'    
    lofarFil = BASE_DATA_PATH + fb_filename
    lofarDM = BASE_DATA_PATH + dm_filename
    
    
    # Determine how many blocks of data are in the file
    content = open(lofarDM).read()
    content = content.split('# ------')[-1].split('Done')[:-1]
    NumberOfBlocks = len(content)
        
        
    # Randomly choose a block number for inspection    
    BlockNumber = random.randint(1, NumberOfBlocks)
    
    
    # Check that the plot for the selected block does not already exist
    saveDir = 'TrainingPlots/'
    saveName = dm_filename[:25] + '_block' + str(BlockNumber) + '.png'
    savePath = saveDir + saveName
    
    if saveName in plotlist:
        continue
        
     # Get the best dm and the bin factor from the .dat file
    for x in content:
        events = x.split('#')
        bufStr = events[-1]
        events = events[0] 
        df = pd.read_csv(StringIO(events), sep=',', names=['MJD', 'DM', 'SNR', 'BinFactor']).dropna()
        binFactor = df['BinFactor'].median()
        if content.index(x) == BlockNumber - 1:
            BestDM = float(bufStr.split('|')[2].split(' ')[3])
            MaxSNR = float(bufStr.split(':')[-1])
        else:
            pass    
        
    
    '''
    This section contains the plotting functions from the notebook to plot an individual block
    '''
   
    # Read in the data using Filterbank class
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
        print('WARNING, %i time samples is not divisible by the time factor %i'%(waterfall.shape[0], timeFactor))
        
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
    
    
    # Save plots to TrainingPlots directory
    plt.savefig(savePath)
    
    
    # Update number of saved plots
    plotlist = os.listdir('TrainingPlots/')        
    NumberOfPlots = len(plotlist) - 1
    
    
    