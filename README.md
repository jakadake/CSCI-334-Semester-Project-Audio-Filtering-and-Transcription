# CSCI 334 Semester Project Audio Filtering and Transcription
 A program that uses python to noisify speech recordings, filter by the Least-Mean-Squares (LMS) adaptive method, then transcribe the audio to Arpabet symbols

processAudio.py will noisify and filter all audio files placed in the [sentence name]\_0riginal\audio file directory. 

processData.py will not run unless processAudio has been run first. It will take the audio files generated by processAudio.py and graph all related sounds, producing a 2x2 grid of spectrograms, another of waveforms, and an analysis file that lists percent error from original to noisy and filtered. if multiple audio files are tested, an averages.txt file will be placed in the analysis directory for that sentence that contains the average percent error among all tested sound files listed by amplitude level.

Note: the transcription portion is incomplete and has been moved to deprecated.