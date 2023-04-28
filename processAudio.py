#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#
#   processAudio.py
#       noisifies, then filters the audio samples, saving the noisy,
#           filtered, and reference noise signals as sound files
#
#   Preconditions: Data folder is formatted as shown below:
#       sentence folder names can be arbitrary but must have subfolders
#       structured as below
#
#       Data
#           |[sentenceFileName]
#           |   |_0riginal
#           |   |   |audio
#           |   |_[%noiseAmplitude]_percent
#           |   |   |filtered
#           |   |   |noisy
#           |   |... (repeat for number of amplitudes to study)
#
#   Postconditions: all audio files in all sentence folders will have
#       been noisified, filtered, and then stored in their respective
#       folders
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

# applies a least mean squares filter to the sound indicated by inFile
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