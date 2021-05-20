# ****************************************************************************
# Name       : utils_audio.py
# Author     : Andres Valdez
# Version    : 1.0
# Description: Several scripts to parse audio files
# References : https://librosa.org/doc/latest/index.html
# Data	 : 16-02-2021
# ****************************************************************************

from __future__ import unicode_literals
import os, sys
import numpy as np
import scipy as scp
import pandas as pd
import librosa as lr
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from math import pi, sin, sqrt, pow , floor, ceil

from scipy import fftpack

from peakdetect import peakdetect
from GCI import SE_VQ_varF0, IAIF, get_vq_params

from scipy.integrate import cumtrapz

#from PyPeVoc.pypevoc.speech.glottal import iaif_ola, lpcc2pole
from external.pypevoc.speech.glottal import iaif_ola, lpcc2pole

#For 0.45\textwidth figs works ok
mpl.rcParams['axes.labelsize']  = 17
mpl.rcParams['xtick.labelsize'] = 17
mpl.rcParams['ytick.labelsize'] = 17
mpl.rcParams['legend.fontsize'] = 17
mpl.rcParams['text.usetex']     = True
mpl.rcParams['font.family']     = 'serif'

import warnings
warnings.filterwarnings("ignore")

np.random.seed(123)

########################################################################
# Routines that open and process audio [*.wav,*.WAV] files
########################################################################

def extract_glottal_features(y, sr,implementation='PyPeVoc'):
    """
    This function extracts the Glottal Flow from a given audio signal.
    Returns Glottal Flow, Glottal Closure Instants, and time array
    y       :: Audio signal
    glottal :: Glottal flow
    sr      :: Sampling rate
    GCIs    :: Glottal Closure instants
    amGCIs  :: Time between two Glottal Closures array
    pos     :: GCIs time indexes
    
    """
    winlen   = int(0.025*sr)
    winshift = int(0.005*sr)
    y        = y-np.mean(y)
    y        = y/float(np.max(np.abs(y)))
    GCIs     = SE_VQ_varF0(y,sr)
    
    g_iaif   = np.zeros(len(y)) # Here store the Glottal flow derivative
    glottal  = np.zeros(len(y)) # Here store the Glottal flow
    wins     = np.zeros(len(y)) 

    start    = 0
    stop     = int(start+winlen)
    win      = np.hanning(winlen)

    while(stop <= len(y)):

        y_frame = y[start:stop]
        pGCIt   = np.where((GCIs>start) & (GCIs<stop))[0]
        GCIt    = GCIs[pGCIt]-start

        g_iaif_f            = IAIF(y_frame,sr,GCIt)
        glottal_f           = cumtrapz(g_iaif_f, dx=1/sr)
        glottal_f           = np.hstack((glottal[start], glottal_f))
        g_iaif[start:stop]  = g_iaif[start:stop] + g_iaif_f*win
        glottal[start:stop] = glottal[start:stop] + glottal_f*win
        start               = start + winshift
        stop                = start + winlen
    
    g_iaif = g_iaif-np.mean(g_iaif)
    g_iaif = g_iaif/max(abs(g_iaif))

    glottal = glottal-np.mean(glottal)
    glottal = glottal/max(abs(glottal))
    glottal = glottal-np.mean(glottal)
    glottal = glottal/max(abs(glottal))
    
    t       = np.arange(0, float(len(y))/sr, 1.0/sr)
    GCIs    = GCIs / sr

    amGCIs  = [glottal[int(k-2)] for k in GCIs]
    
    list_t  = t.tolist()
    pos = []
    for j in GCIs:
        if(j in list_t):
           pos.append(list_t.index(j))
    
    #print('GCI Lenght: ',len(pos))
    #print('GCIs: ',pos)
    
    start , end = pos[0] , pos[-3] # Default settings
    
    amGCIs  = np.array(amGCIs)
    
    new_start = initial_glot_flow(start , glottal)
    
    if(implementation == 'PyPeVoc'):
        return new_start , end
    elif(implementation == 'DisVoice'):
        return t, glottal, GCIs, amGCIs, pos, new_start , end
    else:
        print('Only valid for:',['PyPeVoc','DisVoice'])
        sys.exit()

def initial_glot_flow(start, glottal):
    """
    This function returns the initial position that makes glottal flow rising up
    after a glottal closure instant
    """
    val = glottal[start]
    for k in range( len(glottal) - 10 ):
        val_test = glottal[start + k]
        if(val_test > val):
            return start + k
        else:
            val = val_test

def load_audio_pypevoc(filename):
    """
    This routine opens the *.wav file, and loads it in the memory.
    t       :: Time array
    y       :: Audio signal
    glottal :: Glottal flow
    sr      :: Sampling rate
    GCIs    :: Glottal Closure instants
    amGCIs  :: Time between two Glottal Closures array
    """
    
    # upload the audio file
    y , sr = lr.load(filename)
    
    # remove the silence part
    y , ix = lr.effects.trim(y)
    
    y = y[floor(len(y)/2) - 5000: ceil(len(y)/2) + sr - 5000] # This is sth. to reduce the length of the signal
     
    audio_signal = y
    
    glottal, dg, vt, gf = iaif_ola(y, Fs=sr , tract_order=2 * int(np.round(sr / 2000)) + 4 , glottal_order=2 * int(np.round(sr / 4000)))
    t             = np.arange(len(y))/sr
    
    glottal_signal = glottal
    
    start , end = extract_glottal_features(y, sr)
    
    end = start + 2000 # For Calvin's audio files & other files as well (len 2k also good)
    
    t       = t[start:end]
    glottal = glottal[start:end]
    y       = y[start:end]
    t       = np.linspace(0,lr.get_duration(y=y, sr=sr * 1.0e-03),len(y)) # Reduce sampling rate 1k times
    
    # Normalize audio signal and glottal flow
    y       = y / np.linalg.norm(y)
    glottal = glottal / np.linalg.norm(glottal)
    
    print('Processed audio file  :',filename)
    print('Processed Signal size :',end-start)
    print('Analyzed data frames  :',start,end)
    
    return t,y,glottal,sr,audio_signal,glottal_signal

########################################################################
# Deprecated functions
########################################################################
def glottal_flow_parser(t,glot_flow,kval):
    """
    This function will return the first 5 vibration modes of the glottal flow signal
    t         :: Time array
    glot_flow :: Glottal flow
    kval      :: Number of modes adopted
    """
    glot_flow_fft  = scp.fftpack.fft(glot_flow)
    glot_flow_amp  = 2 / t.size * np.abs(glot_flow_fft)
    glot_flow_freq = np.abs(scp.fftpack.fftfreq(t.size, 3/1000)) # Re-scale the freqs. 1k3 times smaller
    
    # Get the amplitudes
    sa             = pd.Series(glot_flow_amp).nlargest(kval).round(3).astype(float).tolist()
    
    # Get the frequencies
    magnitudes     = abs(glot_flow_fft[np.where(glot_flow_freq >= 0)])
    sf             = np.sort((np.argpartition(magnitudes, -kval)[-kval:])/t[-1])
    
    print('First freqs.',sf[:10])
    print('First amps. ',sa[:10])
    
    gf_new = 0
    for k in range(kval):
        gf_new = gf_new + sa[k] * np.sin(sf[k]*t)
    
    # Return Normalized signal
    gf_new = gf_new / np.linalg.norm(gf_new)
    return t,gf_new

def load_audio(filename):
    """
    This routine opens the *.wav file, and loads it in the memory.
    t       :: Time array
    y       :: Audio signal
    glottal :: Glottal flow
    sr      :: Sampling rate
    GCIs    :: Glottal Closure instants
    amGCIs  :: Time between two Glottal Closures array
    """
    
    # upload the audio file
    y , sr = lr.load(filename)
    
    # remove the silence part
    y , ix = lr.effects.trim(y)
    
    y = y[floor(len(y)/2): ceil(len(y)/2 + sr)]
    
    t, glottal, GCIs, amGCIs, pos, start, end = extract_glottal_features(y, sr, 'DisVoice')
    
    end   = start + 2000
    
    t       = t[start:end]
    glottal = glottal[start:end]
    y       = y[start:end]
    t       = np.linspace(0,lr.get_duration(y=y, sr=sr * 1.0e-03),len(y)) # Reduce sampling rate 1k times
    
    # Normalize audio signal and glottal flow
    y       = y / np.linalg.norm(y)
    glottal = glottal / np.linalg.norm(glottal)
    
    print('Processed audio file  :',filename)
    print('Processed Signal size :',end-start)
    print('Analyzed data frames  :',start,end)
    
    #dummy , dummy = glottal_flow_parser(t,glottal,100)
    
    return t,y,glottal
    
########################################################################
# Here comes the main function
########################################################################

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print('Usage: [utils_audio.py] [audio_file]')
        print('audio_file :: *.WAV, *.wav')
        print(' ')
        sys.exit()
    
    t0 = time.process_time() # Here start count time
    
    dataset     = sys.argv[1]

    t, signal , glot_flow = load_audio_pypevoc(dataset)
    #t, signal , glot_flow = load_audio(dataset)
    
    plt.plot(t, signal, 'r-', alpha=1, label="audio")
    plt.plot(t, glot_flow, 'b-', alpha=1, label="Glottal Flow")

    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.legend(loc='best')
    plt.ylabel('audio signal')
    plt.xlabel('$t$')
    plt.tight_layout()
    plt.savefig(dataset[:-4] + '_data.png')
    #plt.show()
    plt.clf()
    
    t1 = time.process_time() # Here end counting time
    
    print("Elapsed time to solve: ",t1-t0)
