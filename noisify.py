#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#
#   Noisify.py
#       Generates random noise at a specified amplitude and
#       adds it to the given audio file, which is then saved
#       in the specified output file.
#
#   Preconditions:
#       a source audio file in .wav format
#       an amplitude percentage as a number between 0.0 and 1.0
#       an empty .wav file for output
#   Postconditions: 
#       source file is unaffected
#       output file is populated with the sum of the source and noise
#
#   Author: Jacob Haapoja
#   ©2023
#
#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


import numpy as np
import os.path as path
import scipy.io.wavfile as wf

def noisify(inFile, amp, outFile):
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
    
def main():
    validInputFlag = False
    while validInputFlag == False:
        inFile = input('enter full wav file path: ')
        if path.exists(inFile) and path.isfile(inFile) and inFile.endswith('.wav'):
            validInputFlag = True
        else:
            print('ERROR: invalid file path, please try again')

    validInputFlag = False
    while validInputFlag == False:
        try:
            amplitude = float(input('enter percent amplitude of noise as a decimal between 0 and 1: '))
            if 0.0<amplitude<1.0:
                validInputFlag = True
            else:
                print('ERROR: invalid input type, please try again.')
        except:
            print('ERROR: invalid input type, please try again.')

    validInputFlag = False
    while validInputFlag == False:
        outFile = input('enter the output filepath without the file extension: ')
        if path.exists(outFile) and path.isfile(outFile) and outFile.endswith('.wav'):
            validInputFlag = True
        else:
            print('ERROR: invalid file path, please try again')

    if noisify(inFile, amplitude, outFile):
        print('noisified successfully')
        return True
    else:
        print('There was a problem, cancelling operation')
        return False
    
if __name__ == "__main__":
    main()