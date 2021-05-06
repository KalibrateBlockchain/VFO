from __future__ import unicode_literals
import IPython
from scipy.io import wavfile
import noisereduce as nr
import soundfile as sf
from noisereduce.generate_noise import band_limited_noise
import matplotlib.pyplot as plt
import urllib.request
import numpy as np
import io   

import os, sys
import numpy as np
import librosa as lr
import librosa.display
import time

#import matplotlib as mpl
import matplotlib.pyplot as plt
#import matplotlib.mlab as mlab
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
from external.pypevoc.speech.glottal import iaif_ola, lpcc2pole

import pylab
from scipy.fftpack import fft
#import utils_odes


def load_audio(args):
#    %matplotlib inline
    
    working_filepath = args.data_dir+os.path.sep+args.user_id+os.path.sep+args.audio_file
    logging.debug(working_filepath)
    rw_audio, s_rate = librosa.load(working_filepath)
    #s_rate, rw_audio = wavfile.read(working_filepath)
    fig, ax = plt.subplots(figsize=(20,3)) #display raw_audio entire
    plt.title('Raw Audio')
    ax.plot(rw_audio)
    plt.savefig("/VFO/Sample_files/FigRawAudio")
    
    # Here's the noise clip extracted from the raw_audio beginning at 0.3 seconds, and ending at 1.0 seconds
    ns_audio = rw_audio[int(s_rate*.3):int(s_rate*1.0)]

    fig, ax = plt.subplots(figsize=(20,3)) #display noise
    plt.title('Noise')
    ax.plot(ns_audio)
    plt.savefig("/VFO/Sample_files/Noise")

    # Here's the call to the noise reduction routine
    nr_audio = nr.reduce_noise(rw_audio, ns_audio, prop_decrease=1.0)
    
    fig, ax = plt.subplots(figsize=(20,3)) #display noise reduced
    plt.title('Noise Reduced Audio')
    ax.plot(nr_audio)
    plt.savefig("/VFO/Sample_files/NoiseReducedAudio")

    # remove the silence part
    #nrt_audio = lr.effects.trim(nr_audio, top_db=5)
    nrt_audio = detect_leading_silence(nr_audio)

    # trim 0.1 seconds from beginning and end
    nrt_audio = nrt_audio[int(s_rate * 0.1): int(len(nrt_audio)-(s_rate * 0.1))]    
    
    fig, ax = plt.subplots(figsize=(20,3)) #display noise reduced trimmed audio
    plt.title('Noise Reduced Trimmed Audio')
    ax.plot(nrt_audio)
    plt.savefig("/VFO/Sample_files/NoiseReducedTrimmedAudio")
    
    # filter glotal signal
    gl_audio, dg, vt, gf = iaif_ola(nrt_audio, Fs=s_rate , tract_order=2 * int(np.round(s_rate / 2000)) + 4 , glottal_order=2 * int(np.round(s_rate / 4000)))

    fig, ax = plt.subplots(figsize=(20,3)) #display glottal audio
    plt.title('Glottal Audio')
    ax.plot(gl_audio)
    plt.savefig("/VFO/Sample_files/GlottalAudio")

    return nrt_audio, gl_audio

def detect_leading_silence(sound, silence_threshold=-50.0, chunk_size=10):
    '''
    sound is a pydub.AudioSegment
    silence_threshold in dB
    chunk_size in ms
    '''
    trim_ms = 0  # ms
    while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold:
        trim_ms += chunk_size

    return trim_ms        



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
