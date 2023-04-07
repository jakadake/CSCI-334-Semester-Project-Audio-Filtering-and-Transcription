#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#
#   processTranscripts.py
#
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
import nltk
from nltk.corpus import cmudict

def word_to_phonemes(word, pronDict):
    # Look up the pronunciation in the CMU Pronouncing Dictionary
    try:
        phonemes = pronDict[word.lower()][0]
    except KeyError:
        # If the word is not in the dictionary, return an empty list
        phonemes = []
    return phonemes

def writeToFile(phonemes, outFile):
    oFile = open(outFile + "_phonemes.txt")
    for phoneme in phonemes:
        oFile.write(phoneme + "\n")
def prepTranscripts(inFile, outFile):
    pronDict = cmudict.dict();

    if path.isfile(inFile) and inFile.endswith(".txt"):
        with open(inFile) as f:
            text = f.read().strip()
    else:
        print("Error reading file. exiting...")
        return False

    phonemes = []
    words = nltk.work_tokenize(text)
    for word in words:
        phonemes.append(word_to_phonemes(word, pronDict))



