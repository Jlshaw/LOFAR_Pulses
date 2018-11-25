'''
Training Plots Script
'''

# This script randomly selects 1000 blocks of data, plots them and saves the plots such that they can be inspected

# Import Functions
import numpy as np
import matplotlib.pyplot as plt
import dedispersion
import filterbankio_headerless
import matplotlib
import os, random

# Determine how many plots have been made
plotlist = os.listdir('TrainingPlots/')        ##### NEED TO PUT THIS AGAIN AFTER NEW PLOT CREATED #####
NumberOfPlots = len(plotlist) - 1

# Set directory from which to retrieve files
BASE_DATA_PATH = '/oxford_data/FRBSurvey/Archive/'

# Main loop for generating plots
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
    
    
    