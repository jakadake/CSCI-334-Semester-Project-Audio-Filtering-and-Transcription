#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#
#   adaptiveFilter.py
#       Filters noise from an audio file dynamically using the
#       Least-Mean-Squares algorithm. How quickly the filter adapts to
#       shifts in the noise qualities is controlled by the learning rate
#       specified lRate, and how many samples considered when making
#       such changes is controlled by the filter order, specified fOrder
#
#   Preconditions:
#       a source audio file in .wav format
#       a learning rate to adjust how quickly the filter adapts to changes
#       a filter order which controls the number of prior samples 
#           considered when adapting the filter    
#       an empty .wav file for output
#
#   Postconditions: 
#       source file is unaffected
#       output file has been filtered of noise
#
#   Author: Jacob Haapoja
#   ©2023
#
#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import numpy as np
import scipy.io.wavfile as wf
import os.path as path

def LMS(inFile: str, outFile: str, ref_out: str, lRate=0.01, fOrder=100):
    sRate, audioData = wf.read(inFile)
    
    reference = np.random.normal(0, 1, len(audioData))
    
    filtCoef = np.zeros(fOrder)
    filteredAudio = np.zeros(len(audioData))
    
    for n in range(fOrder, len(audioData)-100):
        noiseInput = reference[n-fOrder:n]
        filtOutput = np.dot(filtCoef, noiseInput)
        error = audioData[n] - filtOutput
        filtCoef += lRate * error * noiseInput
        filteredAudio[n] = audioData[n] - filtOutput

    wf.write(ref_out.replace(".wav", "_noise_ref.wav"), sRate, np.int16(reference))
    wf.write(outFile.replace(".wav", "_filtered.wav"), sRate, np.int16(filteredAudio))
    return True

def main():
    validInputFlag = False
    while validInputFlag == False:
        inFile = input('enter full wav file path: ')
        if path.exists(inFile) and path.isFile(inFile) and inFile.endswith('.wav'):
            validInputFlag = True
        else:
            print('ERROR: invalid file path, please try again')
    
    validInputFlag = False
    while validInputFlag == False:
        try:
            lRate = float(input('enter the learning rate as a decimal between 0 and 1: '))
            if 0.0<lRate<1.0:
                validInputFlag = True
            else:
                print('ERROR: invalid input type, please try again.')
        except:
            print('ERROR: invalid input type, please try again.')

    validInputFlag = False
    while validInputFlag == False:
        try:
            fOrder = int(input('enter an integer number of samples to use when adjusting the filter: '))
            validInputFlag = True
        except:
            print('ERROR: invalid input type, please try again.')
    
    validInputFlag = False
    while validInputFlag == False:
        outFile = input('enter full wav file path: ')
        if path.exists(outFile) and path.isFile(outFile) and outFile.endswith('.wav'):
            validInputFlag = True
        else:
            print('ERROR: invalid file path, please try again')
    
    if AF_LMS(inFile, lRate, fOrder, outFile):
        print('noisified successfully')
        return True
    else:
        print('There was a problem, cancelling operation')
        return False

if __name__ == "__main__":
    main()