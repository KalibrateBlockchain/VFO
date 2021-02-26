import importlib
import os
import sys

import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import glob
from math import floor, ceil

from external.pypevoc.speech.glottal import iaif_ola

#CISCO def glottal_flow_extractor(wav_file, save_dir, section = 1):
def glottal_flow_extractor(wav_file_path,wav, sample_rate, section = 1):
    # Read wav
    ### sample_rate, wav = wavfile.read(wav_file)

    if section == 1:
        wav = wav[floor(len(wav)/2): ceil(len(wav)/2 + sample_rate)]
    # if wav.dtype.name == "int16":
    # Convert from to 16-bit int to 32-bit float
    wav = (wav / pow(2, 15)).astype("float32")

    # Extract glottal flow
    g, d_g, vt_coef, g_coef = iaif_ola(
        wav,
        Fs=sample_rate,
        tract_order=2 * int(np.round(sample_rate / 2000)) + 4,
        glottal_order=2 * int(np.round(sample_rate / 4000)),
    )

    # NOTE: If you want to plot glottal flow or save them
    #Plot
    # t = np.arange(len(wav)) / sample_rate
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(t, wav, "c")
    # ax.plot(t, np.linalg.norm(wav) * g / np.linalg.norm(g), "r")
    # plt.show()

    # Save
    #CISCO  np.save(save_dir + "/" + wav_file.replace('/', '_')  + ".npy", g)
    np.save(wav_file_path+os.path.splitext(os.path.basename(wav_file_path))[0] + ".npy", g)
    return g
