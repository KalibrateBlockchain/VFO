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
import pwd
import grp
import librosa as lr
import librosa.display
import soundfile as sf
import time
import datetime
import matplotlib.pyplot as plt
from math import pi, sin, sqrt, pow, floor, ceil
from pypevoc.speech.glottal import iaif_ola, lpcc2pole
import pylab
from utils_odes import residual_ode, ode_solver, ode_sys, physical_props
from utils_odes import foo_main, sys_eigenvals, plot_solution
from models.vocal_fold.vocal_fold_model_displacement import vdp_coupled, vdp_jacobian
from solvers.ode_solvers.ode_solver import ode_solver_1
from fitter import vfo_fitter


fname = "/VFO/Sample_files/F70A4800-2487-D70C-E93B-8F9199D75BB7/TLW-BAAAT.wav"
#fname = "/VFO/Sample_files/F70A4800-2487-D70C-E93B-8F9199D75BB7/TLW-CAAAT.wav"
#fname = "/VFO/Sample_files/F70A4800-2487-D70C-E93B-8F9199D75BB7/TLW-HAAAT.wav"
#fname = "/VFO/Sample_files/F70A4800-2487-D70C-E93B-8F9199D75BB7/TomFlowers-2.wav"
#fname = "/VFO/Sample_files/F70A4800-2487-D70C-E93B-8F9199D75BB7/CWW-CAAAAT-moderate-noise.WAV"
#fname = "/VFO/Sample_files/F70A4800-2487-D70C-E93B-8F9199D75BB7/CWW-CAAAT-heavy-noise.wav"
#fname = "/VFO/Sample_files/F70A4800-2487-D70C-E93B-8F9199D75BB7/CWW-CAAAAT-low-noise-sample.WAV"
#fname = "/VFO/Sample_files/F70A4800-2487-D70C-E93B-8F9199D75BB7/TomFlowers.wav"

print(fname)
from google.colab import drive
drive.mount('/content/drive')
file_audio, s_rate = sf.read(fname)
file_audio = (file_audio / pow(2, 15)).astype("float32") # Convert from to 16-bit int to 32-bit float

fig, ax = plt.subplots(figsize=(20,3)) #display raw_audio entire
plt.title('File Audio')
ax.plot(file_audio)

'''
#parse raw audio into 3 clips that start at StartClip (seconds from start of file) and extend for ClipLength seconds
StartClip1=0.1
StartClip2=35
StartClip3=75
ClipLength=7

Clip1=file_audio[int(StartClip1*s_rate):int((StartClip1+ClipLength)*s_rate)]
Clip2=file_audio[int(StartClip2*s_rate):int((StartClip2+ClipLength)*s_rate)]
Clip3=file_audio[int(StartClip3*s_rate):int((StartClip3+ClipLength)*s_rate)]

rw_audio=Clip1
'''

rw_audio=file_audio

'''
#Here's the code to extract noise clip from animated coached video
ns_audio=numpy.concatenate(([int(s_rate*1.6):int(s_rate*1.9)], rw_audio[int(s_rate*2.45):int(s_rate*2.75)], rw_audio[int(s_rate*3.05):int(s_rate*3.35)])
'''

# Here's the noise clip extracted from the raw_audio beginning at 0.3 seconds, and ending at 1.0 seconds
ns_audio = rw_audio[int(s_rate*.3):int(s_rate*0.5)]
mean_noise=np.mean(np.abs(ns_audio))
max_noise=0.000001
print('mean noise = ',mean_noise, max_noise)

if mean_noise>max_noise:
  print('sample is too noisy')
  sys.exit()

fig, ax = plt.subplots(figsize=(20,3)) #display noise
plt.title('Noise')
ax.plot(ns_audio)


  
abs_audio = np.abs(rw_audio)
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
fig, ax = plt.subplots(figsize=(20,3)) 
plt.title('Average Audio')
ax.plot(avg_audio)


threshold = max(avg_audio)
index=0
while avg_audio[index] < (threshold*.6):
  start = index
  index=index+1
while avg_audio[index] > (threshold*.2):
  end=index
  index=index+1

rwt_audio = rw_audio[start:end]
fig, ax = plt.subplots(figsize=(20,3)) 
plt.title('Trimmed Audio')
ax.plot(rwt_audio)

# trim 0.1 seconds from beginning and end
rwt_audio = rwt_audio[int(s_rate * 0.1): int(len(rwt_audio)-(s_rate * 0.1))]    
fig, ax = plt.subplots(figsize=(20,3)) #display trimmed audio
plt.title('Second Trimmed Audio')
ax.plot(rwt_audio)

t_trimmed = np.arange(len(rwt_audio))/s_rate
   
# filter glotal signal
gl_audio, dg, vt, gf = iaif_ola(rwt_audio, Fs=s_rate , tract_order=2 * int(np.round(s_rate / 2000)) + 4 , glottal_order=2 * int(np.round(s_rate / 4000)))
fig, ax = plt.subplots(figsize=(20,3)) #display glottal audio
plt.title('Glottal Audio')
ax.plot(gl_audio)

# trim audio signal
rwt_audio=rwt_audio[int(s_rate*.3): int(s_rate*0.6)]
fig, ax = plt.subplots(figsize=(20,3)) #display trimmed glottal audio
plt.title('Third Trimmed Audio')
ax.plot(gl_audio)

# trim glotal signal
gl_audio=gl_audio[int(s_rate*.3): int(s_rate*0.6)]
fig, ax = plt.subplots(figsize=(20,3)) #display trimmed glottal audio
plt.title('Third Trimmed Glottal Audio')
ax.plot(gl_audio)


rwt_audio = rwt_audio / np.linalg.norm(rwt_audio)
fig, ax = plt.subplots(figsize=(20,3)) #display noise reduced trimmed audio
plt.title('Normalized Third Trimmed Audio')
ax.plot(rwt_audio)

gl_audio = gl_audio / np.linalg.norm(gl_audio)
fig, ax = plt.subplots(figsize=(20,3)) #display noise reduced trimmed audio
plt.title('Normalized Third Trimmed Glottal Audio')
ax.plot(gl_audio)

#find period points for each period
period = []
i = int(100)
while i < (len(gl_audio) - 100): 
  min_signal = min(gl_audio[(i-100): (i+100)]) 
  max_signal = max(gl_audio[(i-100): (i+100)])
  mid_signal = (min_signal+max_signal)/2
  if (gl_audio[(i-1)] < mid_signal) & (gl_audio[i] > mid_signal) & (gl_audio[i+1] > gl_audio[i]):
    if(gl_audio[i+2] > gl_audio[i+1]) & (gl_audio[i+3] > gl_audio[i+2]) & (gl_audio[i+4] > gl_audio[i+3]):
      if (gl_audio[i+5] > gl_audio[i+4]) & (gl_audio[i+6] > gl_audio[i+5]) & (gl_audio[i+7] > gl_audio[i+6]):
        period.append(i)
  i=i+1


numberOfPeriods = int(8)

if len(period)<numberOfPeriods:
  print("sample can't be analyzed; periods = ",len(period))
  sys.exit()

n=int(len(period))-numberOfPeriods-1
print ("number of available periods", n)

res1=vfo_fitter(gl_audio, rwt_audio, s_rate, period, numberOfPeriods)
res2=vfo_fitter(gl_audio, rwt_audio, s_rate, period, numberOfPeriods)
res3=vfo_fitter(gl_audio, rwt_audio, s_rate, period, numberOfPeriods)

res=res1

if (res2['eigensign'] == res3['eigensign']):
  res=res2

#start of plotting code    

# Initial conditions
t_max = 1000
params = {'text.usetex': False, 'mathtext.fontset': 'stixsans'}
plt.rcParams.update(params)
    
vdp_init_t = 0.0
vdp_init_state = [0.0, 0.1, 0.0, 0.1]  # (xr, dxr, xl, dxl), xl=xr=0

#vdp_params = [A, B, D]
vdp_params = [res['alpha'], res['beta'], res['delta']]

# Solve vocal fold displacement model
sol = ode_solver_1(
    vdp_coupled,
    vdp_jacobian,
    vdp_params,
    vdp_init_state,
    vdp_init_t,
    solver="lsoda",
    ixpr=0,
    dt=0.5,
    tmax=t_max,
    )

# Get steady state
Sr = sol[int(t_max / 2) :, [1, 2]]  # right states, (xr, dxr)
Sl = sol[int(t_max / 2) :, [3, 4]]  # left states, (xl, dxl)



# Plot states
plt.figure()
plt.subplot(121)
plt.plot(Sl[:, 0], Sl[:, 1], 'k.-')
plt.ylabel('Left Vocal Fold, λ = {:.9f}'.format(res['eigenreal1']), fontsize=10)
plt.figtext(0.08, 0.01, "α = {:.3f} , β = {:.3f} , δ = {:.3f}".format(res['alpha'], res['beta'], res['delta']), wrap=True, horizontalalignment='left')


#Plot hide it all
ax = plt.gca()
ax.axes.xaxis.set_visible(True)
ax.axes.yaxis.set_visible(True)
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([])
ax.set_facecolor('none')
plt.grid(False)


plt.subplot(122)
plt.plot(Sr[:, 0], Sr[:, 1], 'k.-')
plt.ylabel('Right  Vocal Fold, λ = {:.9f}'.format(res['eigenreal2']), fontsize=10)
#plt.figtext(0.6, 0.01, eigen)
plt.figtext(0.59, 0.01, datetime.datetime.now())
    
#Plot hide it all
ax = plt.gca()
ax.axes.xaxis.set_visible(True)
ax.axes.yaxis.set_visible(True)
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
ax.axes.xaxis.set_ticklabels([])
ax.axes.yaxis.set_ticklabels([])
ax.set_facecolor('none')
plt.grid(False)



plt.tight_layout()
    
plt.show() #calvin added

plt.clf()
plt.cla()
plt.close()

#end of plotting code

noise_threshold = 0.00000003
if ((res['eigensign'])==1) and (np.mean(np.abs(ns_audio)) > noise_threshold):
  print("sample is too noisy")
  sys.exit()
