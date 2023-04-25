#######################################################################
#
#   processTranscripts.py
#
#   Desc: a script to take in a text file containing an english sentence
#
#   Preconditions:
#
#   Postconditions:
#
#   Author: Jacob Haapoja
#   ©2023
#
#######################################################################

import os.path as path          # for file type checking
import nltk                     # natural language processing toolkit NOTE: requires installing nltk corpus data in advance
from nltk.corpus import cmudict # provides a word to phoneme translation dictionary

def word_to_phonemes(word, pronDict):
    # Look up the pronunciation in the CMU Pronouncing Dictionary
    try:
        phonemes = pronDict[word.lower()][0]
    except KeyError:
        # If the word is not in the dictionary, return an empty list
        phonemes = []
    return phonemes

def writeToFile(phonemes, outFile):
    # open the file for writing
    oFile = open(outFile, "w")
    # write one phoneme per line to oFile
    for phoneme in phonemes:
        oFile.write(phoneme + "\n")

def prepTranscript(inFile, outFile):
    #initialize the pronounciation dictionary
    pronDict = cmudict.dict()
    #verify file exists and is the correct type
    if path.isfile(inFile) and inFile.endswith(".txt"):
        with open(inFile) as f:
            # read the entire contents of the target file into text
            text = f.read()
    else:
        #if file doesn't exist or is the wrong type exit failure
        print("Error reading file. exiting...")
        return False
    # close file after retrieving contents
    f.close()
    # initialize processing bins
    phoneme_sets = [] # a list of lists of phonemes
    phonemes = [] # complete aggregate of all phonemes
    # isolate words from the string
    words = nltk.word_tokenize(text)
    # turn each word into a list of phonemes
    for word in words:
        phoneme_sets.append(word_to_phonemes(word, pronDict))
    # condense phonemes to a 1-D list
    for seq in phoneme_sets:
        # for each word:
        for seg in seq:
            # split word into individual phonemes
            phonemes.append(seg)
    # store in outFile
    writeToFile(phonemes, outFile)

def readFromFile(inFile):
    # type checking
    if path.isfile(inFile) and inFile.endswith(".ipa"):
        #initialize return list
        phonemeSequence = []
        #open file for reading
        file = open(inFile, "r")
        #for each phoneme
        for line in file:
            # add phoneme string to the list w/o the newline
            phonemeSequence.append(line.strip())
        return phonemeSequence


def main():
    # define source files
    s1_path = "Data/S1/transcript.txt"
    s2_path = "Data/S2/transcript.txt"
    s3_path = "Data/S3/transcript.txt"
    s4_path = "Data/S4/transcript.txt"
    s5_path = "Data/S5/transcript.txt"
    s6_path = "Data/S6/transcript.txt"
    # condense for ease of processing
    sentences = [s1_path, s2_path, s3_path, s4_path, s5_path, s6_path]
    # for file in sentences
    for s in sentences:
        # preserve file name
        inFile = s
        # construct outFile name
        outFile = s.replace(".txt", ".ipa")
        # call translator
        prepTranscript(inFile, outFile)
    return True


if __name__ == "__main__":
    main()
