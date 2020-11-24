import numpy as np
import pandas as pd
import os
from glob import glob
import collections
import contextlib
import wave
from external.pypevoc.speech.glottal import iaif_ola
from matplotlib import pyplot as plt
import librosa
import librosa.display

from scipy.io import wavfile
import webrtcvad
import sys 

if __name__ == '__main__':
    """
    Playground for new method exploration
    """

    data_dir = 'data/vowel_a/'
    wav_files = [f for f in glob(data_dir + '**', recursive = True) if os.path.isfile(f)]
    sample_rate, wav = wavfile.read(wav_files[20])

    # wav = (wav / pow(2, 15)).astype("float32")

    # Extract glottal flow
    g, d_g, vt_coef, g_coef = iaif_ola(
        wav,
        Fs=sample_rate,
        tract_order=2 * int(np.round(sample_rate / 2000)) + 4,
        glottal_order=2 * int(np.round(sample_rate / 4000)),
    )

    n_fft = 2048
    hop_length = 1024
    n_mels = 256

    # plt.xlabel('time')
    # plt.ylabel('Amplitude')
    # plt.title('Glottal waveform')
    # plt.tight_layout()
    # plt.plot(g)
    # plt.show()

    S = librosa.feature.melspectrogram(g, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    S_DB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_DB, sr=sample_rate, hop_length=hop_length, x_axis='time', y_axis='mel')
    plt.title('Mel Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.show()

    print("works!")