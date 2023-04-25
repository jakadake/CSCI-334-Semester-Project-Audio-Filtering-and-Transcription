#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#
#   processAudio.py
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
import numpy as np
import scipy.io.wavfile as wf
import os
import os.path as path
import parselmouth as pm
import praatio as pio
import librosa as lib
# from processTranscripts import *
# import math
# import pandas as pd
# import torch
# import keras

def noisify(inFile: str, outFile: str, amp = 0.1):
    if path.isfile(inFile) and inFile.endswith(".wav"):
        outFile = outFile.replace(".wav", f"_{int(amp*100)}_noisy.wav")
        sRate, audioData = wf.read(inFile)
        noise = np.random.normal(0, 1, len(audioData))
        noise = noise * amp * max(abs(audioData))
        noisyAudio = audioData + noise
        wf.write(outFile, sRate, noisyAudio.astype(np.int16))

        if path.exists(outFile):
            return True
        else:
            print('ERROR: problem writing to file, exiting noisify...')
            return False
    else:
        print('ERROR: problem reading from file or incorrect file type, exiting noisify...')
        return False


# def extract(inFile, transcriptFile, outFile):
#     if path.isfile(inFile) and inFile.endswith(".wav"):
#         #load audio and transcript
#         snd = pm.Sound(inFile)
#         audio, sr = lib.load(inFile, sr=16000)
#         phonemes = readFromFile(transcriptFile)
#
#         #pull data from audio file
#         pitch = snd.to_pitch()
#         formants = snd.to_formant_burg()
#         mfcc = snd.to_mfcc()
#
#         tg = pio.textgrid()
#         trans_tier = pio.data_classes.IntervalTier('transcription', [], 0, pairedWaveFN=inFile)
#
#         for i, row in phonemes.iterrows():
#             trans_tier.addInterval(pio.Interval(row['start'], row['end'], row['word']))
#
#         tg.addTier(trans_tier)
#         tg.save(outFile + '_alignment.TextGrid')

def LMS(inFile: str, outFile: str, ref_out: str, amp = 0.1, lRate=0.01, fOrder=100):
    sRate, audioData = wf.read(inFile)

    reference = np.random.normal(0, 1, len(audioData))

    filtCoef = np.zeros(fOrder)
    filteredAudio = np.zeros(len(audioData))

    for n in range(fOrder, len(audioData) - 100):
        noiseInput = reference[n - fOrder:n]
        filtOutput = np.dot(filtCoef, noiseInput)
        error = audioData[n] - filtOutput
        filtCoef += lRate * error * noiseInput
        filteredAudio[n] = audioData[n] - filtOutput

    wf.write(ref_out.replace(".wav", f"_{int(amp * 100)}_noise_ref.wav"), sRate, np.int16(reference))
    wf.write(outFile.replace(".wav", f"_{int(amp * 100)}_filtered.wav"), sRate, np.int16(filteredAudio))
    return True


def main():
    # declare amplitudes to generate
    noiseAmplitudes = [0.05, 0.25, 0.5]

    # get list of sentence files from 'Data' directory
    #   sentence folder names can be arbitrary but must have subfolders structured as below
    #       [sentenceFileName]
    #           | _0riginal
    #           |   |audio
    #           | _[%noiseAmplitude]_percent
    #           |   |filtered
    #           |   |noisy
    #           | ... (repeat for number of amplitudes to study)
    senFolders = os.listdir("Data")

    # For each Sentence Folder
    for sentence in senFolders:
        # define orginal data directory path
        origin = "Data\\" + sentence + "\\_0riginal\\audio"
        # get list of recording files
        recordings = os.listdir(origin)
        # for each recording file do:
        for rec in recordings:
            if rec.endswith(".wav"):
                path = origin + "\\" + rec
                for amp in noiseAmplitudes:
                    outPath = origin.replace("_0riginal\\audio", f"_{int(amp * 100)}_percent\\noisy\\") + rec
                    noisify(path, outPath, amp)
                    path_noisy = outPath.replace(".wav", f"_{int(amp*100)}_noisy.wav")
                    outPath = origin.replace("_0riginal\\audio", f"_{int(amp * 100)}_percent\\filtered\\") + rec
                    ref = outPath.replace("filtered", "noise_references")
                    LMS(path_noisy, outPath, ref, amp)
    return True
    
if __name__ == "__main__":
    main()