# ****************************************************************************
# Name       : audio_parser.py
# Author     : Andres Valdez
# Version    : 1.0
# Description: A script to parse audio file
# References : https://librosa.org/doc/latest/index.html
# Data	 : 16-02-2021
# ****************************************************************************

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

from peakdetect import peakdetect
from GCI import SE_VQ_varF0, IAIF, get_vq_params

from scipy.integrate import cumtrapz

#For 0.45\textwidth figs works ok
mpl.rcParams['axes.labelsize']  = 17
mpl.rcParams['xtick.labelsize'] = 17
mpl.rcParams['ytick.labelsize'] = 17
mpl.rcParams['legend.fontsize'] = 17
mpl.rcParams['text.usetex']     = True
mpl.rcParams['font.family']     = 'serif'

#np.random.seed(123)

########################################################################
# Routines that open *.wav files
########################################################################

def load_audio(filename):
    """
    This routine opens the *.wav file, and loads it in the memory.
    """
    y, sr =lr.load(filename)
    
    t = np.linspace(0,lr.get_duration(y=y, sr=sr),len(y))
    
    GCI    = SE_VQ_varF0(y,sr)
    g_iaif = IAIF(y,sr,GCI)
    g_iaif = g_iaif - np.mean(g_iaif)
    g_iaif = g_iaif/max(abs(g_iaif))
    
    glottal = cumtrapz(g_iaif)
    glottal = glottal-np.mean(glottal)
    glottal = glottal/max(abs(glottal))
    
    return t,y,glottal,sr

########################################################################
# Here comes the main function
########################################################################

#if __name__ == "__main__":

#    t0 = time.process_time() # Here start count time
#    
#    t, signal , glot_flow, sr = load_audio('target.WAV')
#    
#    plt.plot(t, signal, 'r-', alpha=0.5, label="audio")
#    plt.plot(t[:-1], glot_flow, 'b-', alpha=0.9, label="Glottal flow")
#    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
#    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
#    plt.legend(loc='best')
#    plt.ylabel('audio signal')
#    plt.xlabel('$t$')
#    plt.tight_layout()
#    plt.show()
#    
#    print('audio signal',type(signal),signal.shape)
#    print('audio time',type(t),t.shape)
#    
#    t1 = time.process_time() # Here end counting time
#    
#    print("Elapsed time to solve: ",t1-t0)


