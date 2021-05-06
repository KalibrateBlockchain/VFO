import IPython
from scipy.io import wavfile
import noisereduce as nr
import soundfile as sf
from noisereduce.generate_noise import band_limited_noise
import matplotlib.pyplot as plt
import urllib.request
import numpy as np
import io
# %matplotlib inline   

from __future__ import unicode_literals
import os, sys
import numpy as np
import librosa as lr
import librosa.display
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from math import pi, sin, sqrt, pow

#from peakdetect import peakdetect
#from GCI import SE_VQ_varF0, IAIF, get_vq_params

from scipy.integrate import cumtrapz

import numpy as np
import argparse
import logging
import pwd
import grp
import os
import time
#import audio_parser
from glob import glob
import shutil
#from utils import *
#from glottal_flow_extractor import glottal_flow_extractor
#from vocal_fold_estimator import vocal_fold_estimator
#from plot import plot_phasor, plot_mel
#from tqdm import tqdm
import pandas as pd
from scipy.io import wavfile
from math import floor, ceil
import json
from pydub import AudioSegment

import matplotlib.pyplot as plt

import pylab
from scipy.fftpack import fft
#import utils_odes


def load_audio(args):

    s_rate, rw_audio = wavfile.read(args.data_dir + os.path.sep + args.user_id + os.path.sep + args.audio_file)
    fig, ax = plt.subplots(figsize=(20,3)) #display raw_audio entire
    plt.title('Raw Audio')
    ax.plot(rw_audio)
    
    # Here's the noise clip extracted from the raw_audio beginning at 0.3 seconds, and ending at 1.0 seconds
    ns_audio = rw_audio[int(s_rate*.3):int(s_rate*1.0)]

    fig, ax = plt.subplots(figsize=(20,3)) #display noise
    plt.title('Noise')
    ax.plot(ns_audio)

    # Here's the call to the noise reduction routine
    nr_audio = nr.reduce_noise(rw_audio, ns_audio, prop_decrease=1.0)
    
    fig, ax = plt.subplots(figsize=(20,3)) #display raw_audio entire
    plt.title('Noise Reduced Audio')
    ax.plot(rw_audio)

    # remove the silence part
    nrt_audio , ix = lr.effects.trim(nr_audio)

    # trim 0.1 seconds from beginning and end
    nrt_audio = nrt_audio[int(s_rate * 0.1): int(len(nrt_audio)-(s_rate * 0.1))]    
    
    fig, ax = plt.subplots(figsize=(20,3)) #display noise reduced trimmed audio
    plt.title('Noise Reduced Trimmed Audio')
    ax.plot(rrt_audio)
    
    # filter glotal signal
    gl_audio, dg, vt, gf = iaif_ola(nrt_audio, Fs=s_rate , tract_order=2 * int(np.round(s_rate / 2000)) + 4 , glottal_order=2 * int(np.round(s_rate / 4000)))

    fig, ax = plt.subplots(figsize=(20,3)) #display glottal audio
    plt.title('Glottal Audio')
    ax.plot(gl_audio)

    return nrt_audio, gl_audio


if __name__ == '__main__':
    ####os.system('cd /home/cisco/VFO')
    # logging.basicConfig(filename='healthdrop_audio_processor.log', level=logging.DEBUG,format='%(asctime)s %>
    np.random.seed(123)
    logger = logging.getLogger('my_logger')
    logging.basicConfig(level=logging.DEBUG)
    logging.debug('Process Started')
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--user_id", type=str,required=True)
    parser.add_argument("--test_id", type=str,required=True)
    parser.add_argument("--audio_file", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True)
    parser.add_argument("--verbose_mode", type=str, required=False)
    args = parser.parse_args()
    logging.debug(args)
    nrt_audio, gl_audio = load_audio(args)
