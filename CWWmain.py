from __future__ import unicode_literals
import IPython 
from IPython import embed 
%matplotlib inline 
import random 

import os, sys
import io 
import shutil
import numpy as np
import pandas as pd
import scipy as scp
from scipy.io import wavfile
from scipy.integrate import cumtrapz
from scipy.fftpack import fft
import lmfit as lmf
import argparse
import logging
import copy
import pwd
import grp
import librosa as lr
import librosa.display
import soundfile as sf
import noisereduce as nr
import time
import datetime
import json
import matplotlib.pyplot as plt
from math import pi, sin, sqrt, pow, floor, ceil
from pypevoc.speech.glottal import iaif_ola, lpcc2pole
import pylab
from PIL import Image
from utils_odes import residual_ode, ode_solver, ode_sys, physical_props
from utils_odes import foo_main, sys_eigenvals, plot_solution
from models.vocal_fold.vocal_fold_model_displacement import vdp_coupled, vdp_jacobian
from solvers.ode_solvers.ode_solver import ode_solver
from fitter import vfo_fitter, vfo_vocal_fold_estimator
from vocal_fold_estimator import vocal_fold_estimator

#mode_of_processing=1 # for console
mode_of_processing=2 # for production

if mode_of_processing==1:

  #fname = "/VFO/Sample_files/F70A4800-2487-D70C-E93B-8F9199D75BB7/CBW-aaaaa.wav"
  #fname = "/VFO/Sample_files/F70A4800-2487-D70C-E93B-8F9199D75BB7/CBW-CaaaT-lower-noise.wav"
  #fname = "/VFO/Sample_files/F70A4800-2487-D70C-E93B-8F9199D75BB7/TLW-BAAAT.wav"
  #fname = "/VFO/Sample_files/F70A4800-2487-D70C-E93B-8F9199D75BB7/TLW-CAAAT.wav"
  #fname = "/VFO/Sample_files/F70A4800-2487-D70C-E93B-8F9199D75BB7/TLW-HAAAT.wav"
  #fname = "/VFO/Sample_files/F70A4800-2487-D70C-E93B-8F9199D75BB7/TomFlowers-2.wav"
  #fname = "/VFO/Sample_files/F70A4800-2487-D70C-E93B-8F9199D75BB7/CWW-CAAAAT-moderate-noise.WAV"
  #fname = "/VFO/Sample_files/F70A4800-2487-D70C-E93B-8F9199D75BB7/CWW-CAAAT-heavy-noise.wav"
  #fname = "/VFO/Sample_files/F70A4800-2487-D70C-E93B-8F9199D75BB7/CWW-CAAAAT-low-noise-sample.WAV"
  #fname = "/VFO/Sample_files/F70A4800-2487-D70C-E93B-8F9199D75BB7/TomFlowers.wav"
  #fname = "/VFO/Sample_files/PhoneAppSampleFiles/Sample_files_PhoneAppSampleFiles_VowelA210520160707_01.wav"
  #fname = "/VFO/Sample_files/PhoneAppSampleFiles/PhoneAppSampleFiles_VowelA210520161309_01.wav"
  #fname = "/VFO/Sample_files/PhoneAppSampleFiles/TWW-CaaaT-5-20.wav"
  #fname = "/VFO/Sample_files/PhoneAppSampleFiles/VowelA210520161309.caf"
  #fname = "/VFO/Sample_files/ArchiveSamples/DE8083E0-A109-FD70-2300-6BF1AEF3B3E7/VowelA210521113006.caf"
  fname = "/VFO/Sample_files/ArchiveSamples/76A16E11-2409-404D-7EA7-EDD8875561F7/VowelA210421181533.caf"
  #fname = "/VFO/Sample_files/ArchiveSamples/76A16E11-2409-404D-7EA7-EDD8875561F7/VowelA210421083129.caf"


  print(fname)
  from google.colab import drive
  drive.mount('/content/drive')
  CantAnalyze=plt.imread("/VFO/Sample_files/CantAnalyze.png")
  TooNoisy=plt.imread("/VFO/Sample_files/TooNoisy.png")

if mode_of_processing==2:
  working_filepath=""
  audio_file=""
  fname=(working_filepath+os.path.splitext(args.audio_file)[0] + "-sample.WAV")
  CantAnalyze=plt.imread(working_filepath+os.path+"CantAnalyze.png")
  TooNoisy=plt.imread(working_filepath+os.path+"TooNoisy.png")


f_audio, s_rate = sf.read(fname, always_2d=True)
f1_audio=f_audio[:, 0]
file_audio=librosa.resample(f1_audio, s_rate, 22050)
s_rate=22050
file_audio = file_audio / np.linalg.norm(file_audio)

if mode_of_processing==1:
  fig, ax = plt.subplots(figsize=(20,3)) #display raw_audio entire
  plt.title('File Audio')
  ax.plot(file_audio)

rw_audio=file_audio
# filter glotal signal
gl_audio, dg, vt, gf = iaif_ola(rw_audio, Fs=s_rate , tract_order=2 * int(np.round(s_rate / 2000)) + 4 , glottal_order=2 * int(np.round(s_rate / 4000)))


# Here's the noise clip extracted from the raw_audio beginning at 0.3 seconds, and ending at 1.0 seconds
ns_audio = rw_audio[int(s_rate*.3):int(s_rate*0.5)]
mean_noise=np.mean(np.abs(ns_audio))

if mode_of_processing==1:
  fig, ax = plt.subplots(figsize=(20,3)) #display noise
  plt.title('Noise')
  ax.plot(ns_audio)
  print("mean noise = ",mean_noise)

abs_audio = np.abs(rw_audio)

if mode_of_processing==1:
  fig, ax = plt.subplots(figsize=(20,3)) #display noise reduced
  plt.title('Abs Audio')
  ax.plot(abs_audio)

chunk = int(1000)
avg_audio=[]
r_sum=sum(abs_audio[:(chunk-1)])
for index, value in enumerate(abs_audio[: len(abs_audio)-chunk]):
  r_sum=r_sum+abs_audio[index+chunk]
  avg_audio.append(r_sum)
  r_sum=r_sum-abs_audio[index]

if mode_of_processing==1:
  fig, ax = plt.subplots(figsize=(20,3)) 
  plt.title('Average Audio')
  ax.plot(avg_audio)

threshold = max(avg_audio)
index=0
while avg_audio[index] < (threshold*.6):
  start = index
  index=index+1
while (avg_audio[index] > (threshold*.2)) and (index<len(avg_audio)-3):
  end=index
  index=index+1

start=start+int(s_rate*.3)
end=start+int(s_rate*1.0)

rwt_audio = rw_audio[start:end]
gl_audio = gl_audio[start:end]

if mode_of_processing==1:
  fig, ax = plt.subplots(figsize=(20,3)) 
  plt.title('Trimmed Audio')
  ax.plot(rwt_audio)


if mode_of_processing==1:
  fig, ax = plt.subplots(figsize=(20,3)) #display glottal audio
  plt.title('Glottal Audio')
  ax.plot(gl_audio)

rwt_audio = rwt_audio / np.linalg.norm(rwt_audio)

if mode_of_processing==1:
  fig, ax = plt.subplots(figsize=(20,3)) #display noise reduced trimmed audio
  plt.title('Normalized Third Trimmed Audio')
  ax.plot(rwt_audio)

gl_audio = gl_audio / np.linalg.norm(gl_audio)

if mode_of_processing==1:
  fig, ax = plt.subplots(figsize=(20,3)) #display noise reduced trimmed audio
  plt.title('Normalized Third Trimmed Glottal Audio')
  ax.plot(gl_audio)

fig, ax = plt.subplots(figsize=(4,.8)) #display raw_audio entire
plt.title('Audio Signal')
ax.plot(rw_audio, color='k', linewidth=0.05,markersize=0.05)
ax.axvspan(start, end, facecolor='r')
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)


run=1
verbose=1
res=vfo_vocal_fold_estimator(gl_audio,rwt_audio,s_rate,alpha=0.30,beta=0.20,delta=0.50,verbose=1,t_patience = 200, f_delta=0, cut_off=0.25, section = -1)

res2=vfo_vocal_fold_estimator(gl_audio,rwt_audio,s_rate,alpha=0.30,beta=0.20,delta=0.50,verbose=1,t_patience = 200, f_delta=0, cut_off=0.25, section = -1)
res3=vfo_vocal_fold_estimator(gl_audio,rwt_audio,s_rate,alpha=0.30,beta=0.20,delta=0.50,verbose=1,t_patience = 200, f_delta=0, cut_off=0.25, section = -1)
#res4=vfo_vocal_fold_estimator(gl_audio,rwt_audio,s_rate,alpha=0.30,beta=0.20,delta=0.50,verbose=1,t_patience = 200, f_delta=0, cut_off=0.25, section = -1)

if res2['Rk']<res['Rk']:
  res=res2
  run=2
if res3['Rk']<res['Rk']:
  res=res3
  run=3
#if res4['Rk']<res['Rk']:
  #res=res4
  #run=4


res.update({'noise':mean_noise})

t_max = 500
vdp_init_t = 0.0
vdp_init_state = [0.0, 0.1, 0.0, 0.1]  # (xr, dxr, xl, dxl), xl=xr=0
vdp_params = [res['alpha'], res['beta'], res['delta']]

# Solve vocal fold displacement model
sol = ode_solver(
    vdp_coupled,
    vdp_jacobian,
    vdp_params,
    vdp_init_state,
    vdp_init_t,
    solver="lsoda",
    ixpr=0,
    dt=1,
    tmax=t_max,
    )

# Get steady state
Sr = sol[int(t_max / 2) :, [1, 2]]  # right states, (xr, dxr)
Sl = sol[int(t_max / 2) :, [3, 4]]  # left states, (xl, dxl)

if mode_of_processing==1:
  print("Run number: ",run,"Residual: ",res['Rk'])
  print("mean noise = ",mean_noise)

color='w'
if mode_of_processing==1:
  color='k'

#plt.subplots_adjust(hspace = -1.0)
fig = plt.figure(figsize=(8, 18))
ax1= fig.add_subplot(9,9,1,frameon=False)
ax1.axis('off')
ax2= fig.add_subplot(9,9,10,frameon=False)
ax2.axis('off')
ax3= fig.add_subplot(3,2,3,frameon=True)
ax4= fig.add_subplot(3,2,4,frameon=True)
ax5= fig.add_subplot(6,1,5,frameon=False)
ax1.plot(Sl[:, 0], Sl[:, 1], color, linewidth=0.5,markersize=0.5)
ax2.plot(Sr[:, 0], Sr[:, 1], color, linewidth=0.5,markersize=0.5)
ax3.plot(Sl[:, 0], Sl[:, 1], color)
ax3.axes.yaxis.set_ticks([])
ax3.set_ylabel('Left Vocal Fold, λ = {:.9f}'.format(res['eigenreal1']), fontsize=10)
ax3.yaxis.label.set_color(color)
ax3.xaxis.label.set_color(color)
ax3.axes.xaxis.set_ticks([])
ax3.set_xlabel("α = {:.3f} , β = {:.3f} , δ = {:.3f} \nFit 1 = {:.2f}, Fit 2 = {:.2f}, Noise = {:.2f}".format(res['alpha'], res['beta'], res['delta'],res['Rk'],res['distanceRatio'],res['noise']*10000), wrap=True, fontsize=10)
ax4.plot(Sr[:, 0], Sr[:, 1], color)
ax4.axes.yaxis.set_ticks([])
ax4.set_ylabel('Right  Vocal Fold, λ = {:.9f}'.format(res['eigenreal2']), fontsize=10)
ax4.xaxis.label.set_color(color)
ax4.yaxis.label.set_color(color)
ax4.axes.xaxis.set_ticks([])
ax4.set_xlabel("{}".format(res['timestamp']), wrap=True, fontsize=10)
ax5.axes.yaxis.set_ticks([])
ax5.axes.xaxis.set_ticks([])
ax5.plot(rw_audio, color, linewidth=0.05,markersize=0.05)
ax5.axvspan(start, end, facecolor='r')
ax5.set_xlabel("Audio Sample Submission",fontsize=10)
ax5.xaxis.label.set_color(color)
plt.savefig("resplot.png", bbox_inches='tight',pad_inches = 0.05, transparent=True, edgecolor='none')


results_file = open("results.json", "w")
json.dump(res, results_file)
results_file.close()
