#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#
#   extract.py
#       extracts praat data from an audio file
#
#   Preconditions:
#
#   Postconditions: 
#
#   Author: Jacob Haapoja
#   ©2023
#
#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

#import dependencies
import parselmouth as pm
from praatio import tgio
import numpy as np
import librosa as lib
import math
import pandas as pd
import torch
import keras
import os.path as path

def extract(inFile, transcriptFile, outFile):
    if path.isfile(inFile) and inFile.endswith(".wav"):
        #load audio and transcript
        snd = pm.Sound(inFile)
        audio, sr = lib.load(inFile, sr = 16000)
        df = pd.read_csv(transcriptFile)

        #pull data from audio file
        pitch = snd.to_pitch()
        formants = snd.to_formant_burg()
        mfcc = snd.to_mfcc()

        tg = tgio.Textgrid()
        trans_tier = tgio.IntervalTier('transcription', [], 0, pairedWaveFN=inFile)

        #for i, row in df.iterrows():
            




        ##extract pitch
            #extract formants
            #extract mfcc
            
            #align data
            #convert to numpy arrays
            #save numpy arrays in outFile

def main():
    return True
    #tester script
    
if __name__ == "__main__":
    main()