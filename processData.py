#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#
#   processData.py
#       takes in a series of sound recordings and uses parselmouth, 
#       Librosa, and Pandas to create praat annotations for each
#       recording using the corresponding english transcription
#       to prepare the data for insertion to a Recurrent Neural Network
#
#   Preconditions: processAudio has been run and data files are in
#       the correct tree format
#
#   Postconditions: a 2x2 grid for the waveforms and spectrograms
#       have been generated in the figures folder, along with an
#       analysis file listing average percent error rounded to the
#       nearest whole number
#
#   Author: Jacob Haapoja
#   ©2023
#
#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# import dependencies
import parselmouth as pm
import numpy as np
import matplotlib.pyplot as plt
import os.path as path
import os

# draws the spectrograms for one sound file
def draw_spectrograms(fileName: str, orig_path: str, amplitudes=[0.05, 0.25, 0.5], dynamic_range = 70):
    # get the original sound
    original = pm.Sound(orig_path)
    # set figure save location
    figPath = orig_path.replace("_0riginal\\audio\\" + fileName, "figures\\")

    for i, amp in enumerate(amplitudes):
        # set the location of the files for amplitude amp
        data_path = orig_path.replace("_0riginal\\audio\\" + fileName, f"_{int(amp * 100)}_percent\\")
        noisy_path = data_path + "noisy\\" + fileName.replace(".wav", f"_{int(amp * 100)}_noisy.wav")
        filt_path = data_path + "filtered\\" + fileName.replace(".wav", f"_{int(amp * 100)}_filtered.wav")
        ref_path = data_path + "noise_references\\" + fileName.replace(".wav", f"_{int(amp * 100)}_noise_ref.wav")
        # import processed audio as Sound objects
        noisy = pm.Sound(noisy_path)
        filtered = pm.Sound(filt_path)
        reference = pm.Sound(ref_path)
        # create array containing all sound objects to study
        sounds = [original, noisy, filtered, reference]
        # set figure size to 1930x1080
        plt.figure(figsize=(19.2, 10.8))
        # for each sound
        for j, snd in enumerate(sounds):
            # specify subplot
            plt.subplot(2, 2, j+1)
            # extract spectrogram object
            spectrogram = snd.to_spectrogram()
            # draw spectrogram
            X, Y = spectrogram.x_grid(), spectrogram.y_grid()
            sg_db = 10 * np.log10(spectrogram.values)
            plt.pcolormesh(X, Y, sg_db, vmin=sg_db.max() - dynamic_range, label="spectrogram", cmap='binary', alpha=0.7)
            plt.ylim([spectrogram.ymin, 5000])
            # determine title
            if i == 0:
                plt.title("Original", fontsize=40)
            elif i == 1:
                plt.title("Noisy", fontsize=40)
            elif i == 2:
                plt.title("Filtered", fontsize=40)
            else:
                plt.title("Noise Reference", fontsize=40)
            # set labels and font sizes
            plt.xlabel("time [s]", fontsize=40)
            plt.xticks(fontsize=20)
            plt.ylabel("frequency [Hz]", fontsize=40)
            plt.yticks(fontsize=20)
        # end loop
        # adjust layout so labels don't overlap
        plt.tight_layout()
        # save the waveform plots to a png file corresponding to the noise amplitude
        plt.savefig(figPath + fileName.replace(".wav", f"_{int(amp * 100)}_spectrograms.png"))
        # close the window to free memory space
        plt.close()
    return True

# draws the waveforms for one file, calculates the avg error, and returns
#   these quantities as a list
def analyze_waveforms(fileName: str, orig_path: str, amplitudes: list):
    # initialize error% list for the file
    analysis = [[], []]
    # get the original sound
    original = pm.Sound(orig_path)
    # set figure save location
    figPath = orig_path.replace("_0riginal\\audio\\" + fileName, "figures\\")

    # repeat the following for each amplitude at index j
    for j, amp in enumerate(amplitudes):
        # set the location of the files for amplitude amp
        data_path = orig_path.replace("_0riginal\\audio\\" + fileName, f"_{int(amp * 100)}_percent\\")
        noisy_path = data_path + "noisy\\" + fileName.replace(".wav", f"_{int(amp * 100)}_noisy.wav")
        filt_path = data_path + "filtered\\" + fileName.replace(".wav", f"_{int(amp * 100)}_filtered.wav")
        ref_path = data_path + "noise_references\\" + fileName.replace(".wav", f"_{int(amp * 100)}_noise_ref.wav")
        # import processed audio as Sound objects
        noisy = pm.Sound(noisy_path)
        filtered = pm.Sound(filt_path)
        reference = pm.Sound(ref_path)
        # create array containing all sound objects to study
        sounds = [original, noisy, filtered, reference]
        # set figure size to 1930x1080
        plt.figure(figsize=(19.2, 10.8))
        # plot the waveform of each sound to one quadrant of the figure
        for i, snd in enumerate(sounds):
            plt.subplot(2, 2, i+1)
            plt.plot(snd.xs(), snd.values.T)
            plt.xlim([snd.xmin, snd.xmax])
            if i == 0:
                plt.title("Original", fontsize=40)
            elif i == 1:
                plt.title("Noisy", fontsize=40)
            elif i == 2:
                plt.title("Filtered", fontsize=40)
            else:
                plt.title("Noise Reference", fontsize=40)
            plt.xlabel("time [s]", fontsize=20)
            plt.xticks(fontsize=20)
            plt.ylabel("amplitude [Hz]", fontsize=20)
            plt.yticks(fontsize=20)
        # end loop
        # adjust layout so labels don't overlap
        plt.tight_layout()
        # save the waveform plots to a png file corresponding to the noise amplitude
        plt.savefig(figPath + fileName.replace(".wav", f"_{int(amp * 100)}_waveforms.png"))
        # close the window to free memory space
        plt.close()
        # calculate the average error between original vs noisy and original vs filtered
        noisy_err = calc_avg_error(sounds[0], sounds[1])
        filtered_err = calc_avg_error(sounds[0], sounds[2])
        # save the errors on the row corresponding to the noise amplitude
        analysis[0].append(noisy_err)
        analysis[1].append(filtered_err)
    # end loop
    return analysis

# calculates the average error between two sounds
def calc_avg_error(acc: pm.Sound, exp: pm.Sound): # acc -> accepted values (original audio) exp -> experimental values (filtered/noisy)
    err = np.zeros(acc.values.size)
    for i in range(acc.values.size):
        if np.isnan(acc.values[0][i]):
            acc.values[0][i] = 0
        if np.isnan(exp.values[0][i]):
            exp.values[0][i] = 0
        if acc.values[0][i] != 0:
            err[i] = (abs(acc.values[0][i] - exp.values[0][i]) / abs(acc.values[0][i]))*100
    return round(sum(err)/acc.values.size)

# processes all audio files in a single sentence folder
def process_data(sentence_folder: str, amp=[0.05, 0.25, 0.5]):
    if path.exists(sentence_folder):
        # get filenames of all sounds to analyze
        files = os.listdir(sentence_folder + "\\_0riginal\\audio")
        # initialize errors list for noisy / filtered
        n_errors = []
        f_errors = []
        # create a sublist for each amplitude
        for a in amp:
            n_errors.append([])
            f_errors.append([])
        # for each file to analyze
        for f in files:
            # initialize analysis string to print to file
            analysis = "recording\t\t|\t" \
                       "noise amp\t\t|\t" \
                       "avg error noisy\t\t|\t" \
                       "avg error filtered\n"
            # create the full relative path for audio file of name f
            original_path = sentence_folder + "\\_0riginal\\audio\\" + f
            # generate waveform images and return array of errors [[noisy_err],[filt_err]]
            errors = analyze_waveforms(f,
                                       original_path,
                                       amp,
                                       )
            draw_spectrograms(f,
                              original_path,
                              amp)

            # for each amplitude value
            for i, a in enumerate(amp):
                # add a line for that amplitude to analysis string
                analysis += f"{f}\t\t|\t" \
                            f"{amp[i]} Hz\t\t|\t" \
                            f"{errors[0][i]} %\t\t|\t" \
                            f"{errors[1][i]} %\n"
                # append error values to relevant errors sublist
                n_errors[i].append(errors[0][i])
                f_errors[i].append(errors[1][i])
            # write analysis string to file
            an_file = open(sentence_folder + "\\analysis\\" + f.replace(".wav", "_analysis.txt"), "a+")
            an_file.write(analysis)
            an_file.close()
        # write avg errors to another file
        avgs = "Average Errors by Amplitude\n" \
               "Amp\t\t|\tnoisy\t\t|\tfiltered\n"
        for i, a in enumerate(amp):
            avgs += f"{a}\t\t|\t" \
                    f"{np.mean(n_errors[i]) * 100}%\t\t|\t" \
                    f"{np.mean(f_errors[i]) * 100}" \
                    f"\n"
        avgs_file = open(sentence_folder + "\\analysis\\averages.txt", "a+")
        avgs_file.write(avgs)
        avgs_file.close()



        return True
    else:
        return False

## main
def main():

    sentences = os.listdir("Data")
    for sen in sentences:
        process_data("Data\\" + sen)


if __name__ == "__main__":
    main()