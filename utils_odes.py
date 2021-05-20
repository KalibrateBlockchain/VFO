# ****************************************************************************
# Name       : utils_odes.py
# Author     : Andres Valdez
# Version    : 1.0
# Description: A script to solve the Van der Pol oscillators from Singh2020.pdf: "DETECTION OF COVID-19 THROUGH THE ANALYSIS OF VOCAL FOLD OSCILLATIONS"
#              The script includes many other utils, plot, analyze stability, etc.
# References : https://physics.nyu.edu/pine/pymanual/html/chap9/chap9_scipy.html (ODEs' integration with python)
#              https://people.duke.edu/~ccc14/sta-663/CalibratingODEs.html    (ODEs' parameters fitting)
# Data	 : 15-02-2021
# ****************************************************************************

from __future__ import unicode_literals
import os, sys
import numpy as np
import scipy as scp
import time
import lmfit as lmf

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from math import pi, sin, sqrt, pow

from utils_audio import *
#from plot import plot_phasor

#For 0.45\textwidth figs works ok
mpl.rcParams['axes.labelsize']  = 17
mpl.rcParams['xtick.labelsize'] = 17
mpl.rcParams['ytick.labelsize'] = 17
mpl.rcParams['legend.fontsize'] = 17
#mpl.rcParams['text.usetex']     = True
mpl.rcParams['font.family']     = 'serif'

mpl.rcParams['text.usetex'] = False
mpl.rcParams['mathtext.fontset'] = 'stixsans'


import warnings
warnings.filterwarnings("ignore")

np.random.seed(123)

########################################################################
# Routines that solve the ODEs
########################################################################
def physical_props():
    """
    This function returns physical constants
    """
    # Here I define constants of the model (physiological props.)
    c_til = 50      # air particle velocity, m/s
    d_len = 0.0175  # length of vocal folds, m
    x_0   = 0.001   # half glottal width at rest position, m
    
    # Set constants
    M     = 0.5  # mass, g/cm^2
    Bd    = 100  # damping, dyne s/cm^3
    tau   = 1e-3 # time delay for surface wave to travel half glottal height, ms
    eta   = 1.0  # nonlinear factor for energy dissipation at large amplitude
    
    return c_til,d_len,x_0,M,Bd,tau,eta

########################################################################
# Functions to Integrate ODEs and fit
########################################################################
def ode_sys(sol,t,A,B,D):
    """
    This function returns the order reduction for the ODE's (1) and (2)
    from Singh2020.pdf
    A , B, D      :: are (known) scalar parameters
    x_sol , y_sol :: are the vocal fold displacements
    u_sol , v_sol :: are the vocal fold velocities
    You need to convert the 2-second order ODEs into 4-first order ODEs
    """
    
    x_sol , u_sol , y_sol , v_sol = sol
    
    x_der = u_sol
    u_der = (A - B * (1 + np.power(x_sol,2))) * u_sol + (0.5 *D-1) * x_sol + A * v_sol
    y_der = v_sol
    v_der = (A - B * (1 + np.power(y_sol,2))) * v_sol - (0.5 *D+1) * y_sol + A * u_sol
    
    return [x_der,u_der,y_der,v_der]

def ode_solver(A,B,D,sol_0,t):
    """
    This function integrates the coupled ODEs that represent the Van der Pol oscillators
    A     :: alpha
    B     :: beta
    D     :: delta
    sol_0 :: initial condition [x(0),x'(0),y(0),y'(0)]
    t     :: time array
    u0    :: computed glottal flow
    sol   :: displacements of both vocal folds vs time
    """
    # Here I define constants of the model (physiological props.)
    c_til,d_len,x_0,M,Bd,tau,eta = physical_props()
    
    sol   = scp.integrate.odeint(ode_sys,sol_0,t,args=(A,B,D))
    x     = sol[:,0] / np.linalg.norm(sol[:,0])
    u     = sol[:,1] / np.linalg.norm(sol[:,1])
    y     = sol[:,2] / np.linalg.norm(sol[:,2])
    v     = sol[:,3] / np.linalg.norm(sol[:,3])
    u0    = c_til * d_len * (2 * x_0 + x + y)

    # Return Normalized signal
    u0    = u0 / np.linalg.norm(u0)
    return u0,[x,u,y,v]

def synthetic_dataset():
    """
    This function creates a synthetic dataset to benchmark & debug the code
    A , B, D      :: are (known) scalar parameters
    sol_0         :: Initial conditions (x(0),x'(0),y(0),y'(0))
    t             :: Time array
    data          :: Glottal flow
    """
    # Here I define a synthetic dataset just to test the fit
    A     = 0.50 # Reference value
    B     = 0.32 # Reference value
    D     = 0.00 # Reference value

    sol_0 = [0,0.1,0,0.1] # Initial state from papers (reference) (x(0),x'(0),y(0),y'(0))
    t     = np.linspace(0,10*pi,1000)
    data  = ode_solver(A,B,D,sol_0,t)[0]
    #data  = data + 0.01*np.random.normal(size=t.shape) # Add a noise to see the robustness.

    return t,data

def residual_ode(params,t,data,ID):
    """
    Here I define the residual function to minimize. It's the same as Eq. (3) from Singh2020.pdf
    """
    sol_0  = [params[ID[0]].value , params[ID[1]].value , params[ID[2]].value , params[ID[3]].value] 
    A,B,D  =  params[ID[4]].value , params[ID[5]].value , params[ID[6]].value
    u0     = ode_solver(A,B,D,sol_0,t)[0]
    
    K      = np.abs(np.amax(data) - np.amin(data)) / np.abs(np.amax(u0) - np.amin(u0))
    
    if(np.isnan(K)):
        return data * 1.0e02
    else:
        return (K * u0 - data) + 1.0e-03*K*u0

    
    
########################################################################
# The MAIN function. Receives audio file name or nothing and do the analysis.
########################################################################
def foo_main(t, audio_signal, data, audio_file, sampling_rate):
#def foo_main(audio_file=None):
    """
    This is the Main function, Wrapper for fitting and post-process.
    audio_file :: Optional audio file. [*.wav, *.WAV]. If none solves synthetic.
    """

    t0 = time.process_time() # Here start count time
    
    print('Fitting ODEs running on lmfit v{}'.format(lmf.__version__))
    print('Integrating ODEs with scipy v{}'.format(scp.__version__))
    """
    if(audio_file == None):
        print('Processed audio file: ','None-Syntethic')
        t , data   = synthetic_dataset()
        audio_file = 'None.txt'
    else:
        t, signal, data, sampling_rate, audio_signal, glottal_signal = load_audio_pypevoc(audio_file)      # PyPeVoc implementation
     """   
       
    # Define initial guess and ranges for each parameter to fit (stage least squares classic)
    ID      = ['x0','u0','y0','v0','A' ,'B' ,'D']
    #i0      = [0,0.1,0,0.1,0.90,0.90,0.90]
    vi , vf = [0,0,0,0,0,0,0] , [0.1,0.1,0.1,0.1,1,1,1] # According to code, "A,B,D" they are always between (0,1)
    
    # First optimization stage
    # lmfit parameter dictionary
    params = lmf.Parameters()
    for it in range(len(ID)):
        params.add(ID[it], min=vi[it], max=vf[it])

    # lmfit minimizer 
    foo     = lmf.Minimizer(residual_ode, params,fcn_kws={'t': t, 'data':data,'ID':ID})
    result  = foo.minimize(method='differential_evolution')
    #result = foo.minimize(method='Nelder-Mead')
    #result = foo.minimize(method='leastsq')

    # Define initial guess and ranges for each parameter to fit (stage Differential Evolution)
    i0      = [result.params[ID[0]].value , result.params[ID[1]].value , result.params[ID[2]].value , result.params[ID[3]].value ,
               result.params[ID[4]].value , result.params[ID[5]].value , result.params[ID[6]].value]
    
    # Second optimization stage
    # lmfit parameter dictionary
    params = lmf.Parameters()
    for it in range(len(ID)):
        params.add(ID[it],value=i0[it], min=vi[it], max=vf[it])

    # lmfit minimizer 
    foo     = lmf.Minimizer(residual_ode, params,fcn_kws={'t': t, 'data':data,'ID':ID})
    #result  = foo.minimize(method='differential_evolution')
    result = foo.minimize(method='Nelder-Mead')
    #result = foo.minimize(method='leastsq')

    lmf.report_fit(result)
    
    # Pack results
    sol_0   = [result.params[ID[0]].value , result.params[ID[1]].value , result.params[ID[2]].value , result.params[ID[3]].value]
    A,B,D   =  result.params[ID[4]].value , result.params[ID[5]].value , result.params[ID[6]].value
    dt      = 1.0e-01
    N_steps = int((t[-1] - t[0]) / dt)
    t_model = np.linspace(t[0],t[-1],N_steps)
    u0,sol  = ode_solver(A,B,D,sol_0,t_model)
    x,u,y,v = sol
    
    K = np.abs(np.amax(data) - np.amin(data)) / np.abs(np.amax(u0) - np.amin(u0))
    print('Amplitude Factor',K)
    
    title = '$\\alpha$: ' + '{:10.3e}'.format(A) + ' $\\beta$: ' + '{:10.3e}'.format(B) + ' $\delta$: ' + '{:10.3e}'.format(D)
    
    # Analyze the equilibrium of the system
    l = np.linspace(-5,5,100)
    p,r1,i1,r2,i2 = sys_eigenvals(l,A,B,D)
    
    t1 = time.process_time() # Here end counting time
    print("Elapsed time to solve: ",(t1-t0) / 60,"minutes")
    print('')
    
    Sr, Sl = plot_phasor(audio_file, audio_signal, A, B, D, "", data, sampling_rate*1000)
    
    return t,data,t_model,K*u0,x,u,y,v,title

########################################################################
# Functions to analyze, post-process
########################################################################
def sys_eigenvals(l,a,b,d):
    """
    Here we obtain the characteristic equation for the model, and the eigenvalues.
    Helps to study the stability.
    """
    #P = -2.0*a*b*l**2 - 2.0*a*l**3 - 2.0*a*l + 1.0*b**2*l**2 + 2.0*b*l**3 + 2.0*b*l - 0.25*d**2 + 1.0*l**4 + 2.0*l**2 + 1.0
    p4 = 1.0
    p3 = -2.0*a + 2.0*b
    p2 = -2.0*a*b + b**2 + 2.0
    p1 = -2.0*a + 2.0*b
    p0 = -0.25*d**2 + 1.0
    
    P = p4*np.power(l,4) + p3*np.power(l,3) + p2*np.power(l,2) + p1*l + p0
    
    coeff = np.array([p4,p3,p2,p1,p0])
    
    l1,l2,l3,l4 = np.roots(coeff)[0],np.roots(coeff)[1],np.roots(coeff)[2],np.roots(coeff)[3]
    
    r1 , i1 = l1.real , l1.imag
    r2 , i2 = l3.real , l3.imag

    """
    print('#############################################################')
    print('Analyzing the Eigenvalues of the system.')
    print('Real negative is asympt. stable')
    print('Real positive is unstable')
    print('Complex part not zero is a spiral')
    print('Real part equal to zero is a center')
    print('#############################################################')
    print('Real part mode 1:',r1,'Complex part mode 1:',i1)
    print('Real part mode 2:',r2,'Complex part mode 2:',i2)
    """
    
    return P,r1,i1,r2,i2

def plot_solution(t_tgt,u0_tgt,t_ode,u0,x,u,y,v,filename,title):
    """
    This function plots fitted model vs data
    t_tgt,u0_tgt     :: Experimental time, and experimental glottal flow
    t_ode,u0,x,u,y,v :: Model evaluations
    strs             :: Strings for alpha, beta, delta
    """
    
    # Plot data to fit vs model evaluation
    plt.plot(t_ode, u0, 'b-', label='model fit')
    plt.plot(t_tgt,u0_tgt, 'r--', label='target')
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.legend(loc='best')
    plt.xlabel('$t$')
    plt.tight_layout()
    plt.savefig(filename + '_model_fit.png')
    #plt.show()
    plt.clf()
    
    pos = int(0.5 * len(x))
    # plot phase space
    plt.plot(x[pos:], u[pos:], 'b-', label='Right')
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.xlabel('$x(t)$')
    plt.ylabel('$\dot{x}(t)$')
    #plt.legend(loc='best')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename + '_x_phase.png')
    #plt.show()
    plt.clf()

    plt.plot(y[pos:], v[pos:], 'b-', label='Left')
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.xlabel('$y(t)$')
    plt.ylabel('$\dot{y}(t)$')
    #plt.legend(loc='best')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename + '_y_phase.png')
    #plt.show()
    plt.clf()

########################################################################
# Here comes the main function
########################################################################

if __name__ == "__main__":

    t0 = time.process_time() # Here start count time
    
    # Here I call constants of the model (physiological props.)
    c_til,d_len,x_0,M,Bd,tau,eta = physical_props()
    sol_0 = [0,0.1,0,0.1] # Initial state from papers (reference) (x(0),x'(0),y(0),y'(0))
    
    # Here I define a synthetic dataset just to test the integration
    A     = 0.50 # Reference value
    B     = 0.32 # Reference value
    D     = 0.00 # Reference value
    
    D     = 0.3  #np.random.random() # Reference value
    A     = 0.1 * D # Stable if factor lower than 0.5
    B     = 0.2

    A     = 0.5313284693632837
    B     = 0.23036597364638373
    D     = 0.7731024307045707
    
    # Here I solve the Vocal Fold's ODEs
    t             = np.linspace(0,60*pi,1000)
    data,voc_sol  = ode_solver(A,B,D,sol_0,t)
    
    # Plot the reference solution
    plt.plot(t, data, 'r-', label='Glottal flow')
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.legend(loc='best')
    plt.xlabel('$t$')
    plt.tight_layout()
    plt.show()
    plt.clf()

    # Analyze the equilibrium of the system
    l = np.linspace(-10,10,1000)
    p,r1,i1,r2,i2 = sys_eigenvals(l,A,B,D)

    plt.plot(l, p, 'b-', label='$\mathcal{P}(\lambda)$')
    plt.axvline(i1,linestyle='--',color='red',label='$\lambda_{1}$')
    plt.axvline(i2,linestyle='--',color='green',label='$\lambda_{2}$')
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.xlabel('$\lambda$')
    plt.ylabel('characteristic polynomial')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    plt.clf()
    
    t1 = time.process_time() # Here end counting time
    
    print("Elapsed time to solve: ",t1-t0)
