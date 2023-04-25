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
import numpy as np
import matplotlib.pyplot as plt
import os.path as path
import os

def draw_spectrograms(snd: pm.Sound, amp: list, dynamic_range = 70):
    spectrogram = snd.to_spectrogram()
    fig = plt.figure(figsize=(12.8, 7.2))
    X, Y = spectrogram.x_grid(), spectrogram.y_grid()
    sg_db = 10*np.log10(spectrogram.values)
    plt.pcolormesh(X, Y, sg_db, vmin=sg_db.max() - dynamic_range,label="spectrogram", cmap='binary', alpha=0.7)
    plt.ylim([spectrogram.ymin, 5000])
    plt.xlabel("time [s]")
    plt.ylabel("frequency [Hz]")
    return fig

def analyze_waveforms(fileName: str, orig_path: str, amplitudes: list):
    analysis = [[], []]
    for j, amp in enumerate(amplitudes):
        original = pm.Sound(orig_path)
        data = orig_path.replace("_0riginal\\" + fileName, f"_{int(amp * 100)}_percent\\" + fileName)
        noisy = pm.Sound(data.replace(".wav", f"_{int(amp * 100)}_noisy.wav"))
        filtered = pm.Sound(data.replace(".wav", f"_{int(amp * 100)}_filtered.wav"))
        reference = pm.Sound(data.replace(".wav", f"_{int(amp * 100)}_noise_ref.wav"))
        sounds = [original, noisy, filtered, reference]

        plt.figure(figsize=(19.2, 10.8))

        for i, snd in enumerate(sounds):
            plt.subplot(2, 2, i+1)
            plt.plot(snd.xs(), snd.values.T)
            plt.xlim([snd.xmin, snd.xmax])
            plt.xlabel("time [s]")
            plt.ylabel("amplitude")

        figPath = f"_{int(amp * 100)}_percent" + fileName.replace(".wav", "_waveforms.png")
        plt.savefig(orig_path.replace("_0riginal\\" + fileName, figPath))
        noisy_err = calc_avg_error(sounds[0], sounds[1])
        filtered_err = calc_avg_error(sounds[0], sounds)
        analysis[0][j] = noisy_err
        analysis[1][j] = filtered_err

    return

def calc_avg_error(acc: pm.Sound, exp: pm.Sound):
    err = np.zeros(len(acc))
    for i in range(len(acc)):
        if np.isnan(acc[i]):
            acc[i] = 0
        if np.isnan(exp[i]):
            exp[i] = 0

        err[i] = abs(acc[i] - exp[i]) / acc[i]
    return sum(err)/len(acc)

def process_data(sentence_folder: str, amp = [0.05, 0.25, 0.5]):
    if path.exists(sentence_folder):
        files = os.listdir(sentence_folder + "\\_0riginal\\audio")
        for f in files:
            analysis = "recording\t|\tnoise amp\t|\tavg error noisy\t|\tavg error filtered\n"
            original_path = sentence_folder + "\\_0riginal\\audio\\" + f

        return True
    else:
        return False

def main():
    sentences = os.listdir("Data")
    for sen in sentences:
        process_data("Data\\" + sen)

    return True