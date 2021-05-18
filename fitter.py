import random 
import numpy as np
import pandas as pd
import scipy as scp
from scipy.io import wavfile
from scipy.integrate import cumtrapz
from scipy.fftpack import fft
import lmfit as lmf
import librosa as lr
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
import time
from utils_odes import residual_ode, ode_solver, ode_sys, physical_props
from utils_odes import foo_main, sys_eigenvals, plot_solution
from models.vocal_fold.vocal_fold_model_displacement import vdp_coupled, vdp_jacobian
from solvers.ode_solvers.ode_solver import ode_solver_1

def vfo_fitter(gl_audio, rwt_audio, s_rate, period, numberOfPeriods):  
  n=int(len(period))-numberOfPeriods-1
  startPeriod = random.randint(1, n)
  print (startPeriod)

  rwt_audio_all=rwt_audio[period[startPeriod]: ]
  rwt_audio_analyze = rwt_audio[period[startPeriod]: period[(startPeriod + numberOfPeriods)]]
  t_analyze = np.arange(len(rwt_audio_analyze))/s_rate

  gl_audio_all = gl_audio[period[startPeriod]:]
  gl_audio_analyze = gl_audio[period[startPeriod]: period[(startPeriod + numberOfPeriods)]]
  fig, ax = plt.subplots(figsize=(20,3)) #display gl_audio_analyze
  plt.title('Glottal Audio Analyze')
  ax.plot(gl_audio_analyze)

  fig, ax = plt.subplots(figsize=(20,3)) #Audio Analyze
  plt.title('Audio Analyze')
  ax.plot(rwt_audio_analyze)

  t = np.linspace(0,lr.get_duration(y=gl_audio_analyze, sr=s_rate * 1.0e-03),len(gl_audio_analyze)) # Reduce sampling rate 1k times

  #start fitting code
  # Define initial guess and ranges for each parameter to fit (stage least squares classic)
  t0 = time.process_time() # Here start counting time
  ID      = ['x0','u0','y0','v0','A' ,'B' ,'D']
  vi , vf = [0,0,0,0,0,0,0] , [0.1,0.1,0.1,0.1,1,1,1] # According to code, "A,B,D" they are always between (0,1)
    
  # First optimization stage
  # lmfit parameter dictionary
  params = lmf.Parameters()
  for it in range(len(ID)):
    params.add(ID[it], min=vi[it], max=vf[it])

  # lmfit minimizer 
  foo     = lmf.Minimizer(residual_ode, params,fcn_kws={'t': t, 'data':gl_audio_analyze,'ID':ID})
  result  = foo.minimize(method='differential_evolution')

  # Define initial guess and ranges for each parameter to fit (stage Differential Evolution)
  i0      = [result.params[ID[0]].value , result.params[ID[1]].value , result.params[ID[2]].value , result.params[ID[3]].value , 
           result.params[ID[4]].value , result.params[ID[5]].value , result.params[ID[6]].value]
    
  # Second optimization stage
  # lmfit parameter dictionary
  params = lmf.Parameters()
  for it in range(len(ID)):
    params.add(ID[it],value=i0[it], min=vi[it], max=vf[it])

  # lmfit minimizer 
  foo     = lmf.Minimizer(residual_ode, params,fcn_kws={'t': t, 'data':gl_audio_analyze,'ID':ID})
  result = foo.minimize(method='Nelder-Mead')
  lmf.report_fit(result)
    
  # Pack results
  sol_0   = [result.params[ID[0]].value , result.params[ID[1]].value , result.params[ID[2]].value , result.params[ID[3]].value]
  A,B,D   =  result.params[ID[4]].value , result.params[ID[5]].value , result.params[ID[6]].value
  dt      = 1.0e-01
  N_steps = int((t[-1] - t[0]) / dt)
  t_model = np.linspace(t[0],t[-1],N_steps)
  u0,sol  = ode_solver(A,B,D,sol_0,t)

  x,u,y,v = sol
  
  # Plot data to fit vs model evaluation

  fig, ax = plt.subplots(figsize=(20,3)) 
  plt.title('Fitting')
  ax.plot(u0[:len(gl_audio_analyze)], 'b-', label='model fit')
  ax.plot(gl_audio_analyze, 'r--', label='target')
  #plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
  #plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
  plt.legend(loc='best')
  #plt.xlabel('$t$')
  ax.set_xlabel('λ')
  plt.show()
  #plt.savefig(filename + '_model_fit.png')

    
  # Analyze the equilibrium of the system
  l = np.linspace(-5,5,100)
  p,r1,i1,r2,i2 = sys_eigenvals(l,A,B,D)
    
  t1 = time.process_time() # Here end counting time
  print("Elapsed time to solve: ",(t1-t0) / 60,"minutes")
  print("r1 = ", r1," r2 = ",r2)
  
  res = {
    'alpha':A,
    'beta':B,
    'delta':D,
    'eigenreal1':r1,
    'eigenreal2': r2,
    'eigensign':np.sign(r1*r2),
    'chisquared':result.chisqr(),
    'gl_audio_analyze':gl_audio_analyze,
    'rwt_audio_analyze':rwt_audio_analyze,
  }
  
  return res
