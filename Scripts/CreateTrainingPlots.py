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

# Determine how many plots have been made
plotlist = os.listdir('TrainingPlots/')        
NumberOfPlots = len(plotlist) - 1

# Set directory from which to retrieve files
BASE_DATA_PATH = '/oxford_data/FRBSurvey/Archive/'



'''
Main loop for generating a certain number of plots 
'''

while NumberOfPlots < 10:
    
    
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
    BlockNumber = random.randint(1, NumberOfBlocks + 1)
    
    
    # Check that the plot for the selected block does not already exist
    saveDir = 'TrainingPlots/'
    saveName = dm_filename[:25] + '_block' + str(BlockNumber) + '.png'
    savePath = saveDir + saveName
    
    if saveName in plotlist:
        continue
        
        
        
    
    '''
    This section contains the plotting functions from the notebook to plot an individual block
    '''
   
    # Read in the data using Filterbank class
    fil = filterbankio_headerless.Filterbank(lofarFil, BlockNumber)
    timeFactor = 64
    freqFactor = 8

    tInt = fil.my_header[5] # set tInt

    freqsHz = fil.freqs * 1e6 # generate array of freqs in Hz

    waterfall = np.reshape(fil.data, (fil.data.shape[0], fil.data.shape[2])) # reshape to (n integrations, n freqs)
    waterfall = waterfall.reshape(int(waterfall.shape[0]/timeFactor), timeFactor, waterfall.shape[1]).sum(axis=1)
    tInt *= timeFactor
    
    
    # Get the best dm from the .dat file
    for x in content:
        if content.index(x) == BlockNumber - 1:
            events = x.split('#')
            bufStr = events[-1]
            BestDM = float(bufStr.split('|')[2].split(' ')[-2])
        else:
            pass
        
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
    plt.title('LOFAR Event - Dedispersed', fontdict={'fontsize':'small'})
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
    
    
    