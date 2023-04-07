#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#
#   processData.py
#       takes in a series of sound recordings and uses parselmouth, 
#       Librosa, and Pandas to create praat annotations for each
#       recording using the corresponding english transcription
#       to prepare the data for insertion to a Recurrent Neural Network
#
#   Preconditions:
#
#   Postconditions: 
#
#   Author: Jacob Haapoja
#   ©2023
#
#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import parselmouth as pm
import librosa as lib
import pandas
from noisify import *
from adaptiveFilter import *
from extract import *

def noisifyDir(sentenceDir):
    return True
    #validate directory
    
    #for each file in the "_0riginal" folder:
        #noisify at 5% and store in 5%/noisy folder
        #noisify at 25% and store in 25%/noisy folder
        #noisify at 50% and store in 50%/noisy folder
        

def filterDir(sentenceDir):
    return True
    #validate directory
    
    #for each of 5%, 25%, and 50%:
        #for each file in the noisy directory:
            #apply LMS filter and store in filtered directory

def extractDir(sentenceDir):
    return True
    #validate directory
    
    #for each of 0riginal, 5%, 25%, 50%:
        #for each file in filtered directory:
            #extract numpy array and store in npy_array directory

def ProcessData(dataDir):
    return True
    #validate directory
    
    #validate that each sentence has proper folders
    
    #for each sentence [1-6]:
        #noisifyDir
        #filterDir
        #extractDir
        
    #report success