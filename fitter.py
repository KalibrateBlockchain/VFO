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
import datetime
from utils_odes import residual_ode, ode_solver, ode_sys, physical_props
from utils_odes import foo_main, sys_eigenvals, plot_solution
#from models.vocal_fold.vocal_fold_model_displacement import vdp_coupled, vdp_jacobian
from solvers.ode_solvers.ode_solver import ode_solver_1


from models.vocal_fold.adjoint_model_displacement import adjoint_model
from models.vocal_fold.vocal_fold_model_displacement import (
    vdp_coupled,
    vdp_jacobian,
)
from solvers.ode_solvers.dae_solver import dae_solver
from solvers.ode_solvers.ode_solver import ode_solver
from solvers.optimization import optim_adapt_step, optim_grad_step
from math import floor, ceil
import logging
#np.random.seed(123)




def vfo_fitter(gl_audio, rwt_audio, s_rate, period, numberOfPeriods):  
  
<<<<<<< HEAD
  verbose=1 # for console
=======
  verbose=0 # for console
>>>>>>> 015aaac3d09192f1835a0386a42cbbca9f2f7c7a
  
  n=int(len(period))-numberOfPeriods-1
  startPeriod = random.randint(1, n)
  if verbose==1:
    print (startPeriod)

  rwt_audio_all=rwt_audio[period[startPeriod]: ]
  rwt_audio_analyze = rwt_audio[period[startPeriod]: period[(startPeriod + numberOfPeriods)]]
  t_analyze = np.arange(len(rwt_audio_analyze))/s_rate

  gl_audio_all = gl_audio[period[startPeriod]:]
  gl_audio_analyze = gl_audio[period[startPeriod]: period[(startPeriod + numberOfPeriods)]]
  
  if verbose==1:
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
  if verbose==1:
    fig, ax = plt.subplots(figsize=(20,3)) 
    plt.title('Fitting')
    ax.plot(u0[:len(gl_audio_analyze)], 'b-', label='model fit')
    ax.plot(gl_audio_analyze, 'r--', label='target')
    plt.legend(loc='best')
    ax.set_xlabel('λ')
    plt.show()
    
  # Analyze the equilibrium of the system
  l = np.linspace(-5,5,100)
  p,r1,i1,r2,i2 = sys_eigenvals(l,A,B,D)
    
  t1 = time.process_time() # Here end counting time
  
  if verbose==1:
    print("Elapsed time to solve: ",(t1-t0) / 60,"minutes")
    print("r1 = ", r1," r2 = ",r2)
  
  res = {
    'alpha':float(A),
    'beta':float(B),
    'delta':float(D),
    'eigenreal1':float(r1),
    'eigenreal2':float(r2),
    'eigensign':int(np.sign(r1*r2)),
    #'chisquared':float(result.chisqr),
    #'gl_audio_analyze':gl_audio_analyze,
    #'rwt_audio_analyze':rwt_audio_analyze,
    'timestamp': datetime.datetime.now().isoformat(),
  }
  
  return res

def vfo_vocal_fold_estimator(glottal_flow,wav_samples,sample_rate,alpha=0.3,beta=0.2,delta=0.5,verbose=-1,t_patience = 5,f_delta=0, cut_off=0.0, section = 1):
    i_delta=delta
    """
    Inputs: wav_samples: audio wavfile
            glottal_flow: numpy array of glottal flow from IAIF
    returns: dictionary best_results:
    ["iteration", "R", "Rk", "alpha", "beta", "delta", "sol", "u0"]
    """
    
    # Set constants
    M = 0.5  # mass, g/cm^2
    B = 100  # damping, dyne s/cm^3
    d = 1.75  # length of vocal folds, cm
    x0 = 0.1  # half glottal width at rest position, cm
    tau = 1e-3  # time delay for surface wave to travel half glottal height, ms
    c = 5000  # air particle velocity, cm/s
    eta = 1.0  # nonlinear factor for energy dissipation at large amplitude
<<<<<<< HEAD
    
    #compute d_1; distance of glottal_flow signal
    i_1=1
    g_1=0
=======
     
    #compute d_1; distance of glottal_flow signal
    i_1=1
    g_1=0
    
    #set timers for analysis of processing resources
    dae_t=0
    ode_t=0
    
>>>>>>> 015aaac3d09192f1835a0386a42cbbca9f2f7c7a
    while i_1<(len(glottal_flow)-1):
        i_1=i_1+1
        g_1=g_1+np.abs(glottal_flow[i_1-1]-glottal_flow[i_1])
 

    """CISCO
    sample_rate, wav_samples = wavfile.read(wav_file_path)
    if section == 1:
        wav_samples = wav_samples[floor(len(wav_samples)/2): ceil(len(wav_samples)/2 + sample_rate)]
    """
    
    # NOTE: If you want to plot glottal flow and original waveform together
    # fig = plt.figure()
    # plt.plot(np.linspace(0, len(wav_samples) / sample_rate, len(wav_samples)), wav_samples)
    # plt.plot(np.linspace(0, len(glottal_flow) / sample_rate, len(glottal_flow)), glottal_flow)
    # plt.legend(["speech sample", "glottal flow"])
    # plt.show()

    # Set model initial conditions
    #delta = np.random.random()  # asymmetry parameter
    #alpha = 0.6 * delta  # if > 0.5 delta, stable-like oscillator
    #beta = 0.2

    vdp_init_t = 0.0
    vdp_init_state = [0.0, 0.1, 0.0, 0.1]  # (xr, dxr, xl, dxl), xl=xr=0
    num_tsteps = len(wav_samples)  # total number of time steps
    T = len(wav_samples) / float(sample_rate)  # total time, s
    if verbose==1:
        print("Initial parameters: alpha = ",alpha," beta = ",beta," delta = ",delta)

    # Optimize
    best_results: Dict[str, List[float]] = {  # store best results over iterations
        "iteration": [],  # optimize iter
        "R": [],  # estimation residual @ k
        "Rk": [],  # estimation residual w.r.t L2 norm @ k
        "alpha": [],
        "beta": [],
        "delta": [],
        "sol": [],  # model ouputs
        "u0": [],  # estimated glottal flow
    }
    iteration = 0
    Rk = 1e16
    Rk_best = 1e16
    patience = 0  # number of patient iterations of no improvement before stopping optimization
    if_adjust = 0

    while patience < t_patience: # this was 400 default
        if f_delta==1:
            delta=i_delta
        # Solve vocal fold displacement model
        # logger.info("Solving vocal fold displacement model")
        K = B ** 2 / (beta ** 2 * M)
        Ps = (alpha * x0 * np.sqrt(M * K)) / tau
        time_scaling = np.sqrt(K / float(M))  # t -> s

        x_scaling = np.sqrt(eta)
        vdp_params = [alpha, beta, delta]
<<<<<<< HEAD
=======
        ode_t=ode_t-time.process_time() #cacluate ode time
>>>>>>> 015aaac3d09192f1835a0386a42cbbca9f2f7c7a
        sol = ode_solver(
            vdp_coupled,
            vdp_jacobian,
            vdp_params,
            vdp_init_state,
            (time_scaling * vdp_init_t),
            solver="lsoda",
            ixpr=0,
            dt=(time_scaling / float(sample_rate)),  # dt -> ds
            tmax=(time_scaling * T),
        )
<<<<<<< HEAD
=======
        ode_t=ode_t+time.process_time() #cacluate ode time
>>>>>>> 015aaac3d09192f1835a0386a42cbbca9f2f7c7a

        if len(sol) > len(wav_samples):
            sol = sol[:-1]
        ##assert len(sol) == len(wav_samples)
          ##  if verbose==1:
            ##    print("Inconsistent length: ODE sol;",len(sol),len(wav_samples))

        # Calculate glottal flow
        try:
            assert sol.size > 0
            X = sol[:, [1, 3]]  # vocal fold displacement (right, left), cm
            dX = sol[:, [2, 4]]  # cm/s
            u0 = c * d * (np.sum(X, axis=1) + 2 * x0)  # volume velocity flow, cm^3/s
            u0 = u0 / np.linalg.norm(u0) * np.linalg.norm(glottal_flow)  # normalize
        except AssertionError as e:
            logger.error(e)
            logger.warning("Skip")
            break

        # Estimation residual
        R = u0 - glottal_flow

        # Solve adjoint model
        # logger.info("Solving adjoint model")

        residual, jac = adjoint_model(alpha, beta, delta, X, dX, R, sample_rate, 0, T)
        M_T = [0.0, 0.0, 0.0, 0.0]  # initial states of adjoint model at T
        dM_T = [0.0, -R[-1], 0.0, -R[-1]]  # initial ddL = ddE = -R(T)
        try:
<<<<<<< HEAD
=======
            dae_t=dae_t-time.process_time() #cacluate dae time
>>>>>>> 015aaac3d09192f1835a0386a42cbbca9f2f7c7a
            adjoint_sol = dae_solver(
                residual,
                M_T,
                dM_T,
                T,
                tfinal=0,  # simulate (tfinal-->t0)s backward
                backward=True,
                ncp=len(wav_samples),
                solver="IDA",
                algvar=[0, 1, 0, 1],
                suppress_alg=True,
                atol=1e-6,
                rtol=1e-6,
                usejac=True,
                jac=jac,
                usesens=False,
                display_progress=True,
                report_continuously=False,  # NOTE: report_continuously should be False
                verbosity=50,
            )
<<<<<<< HEAD
=======
            dae_t=dae_t+time.process_time() #cacluate dae time
>>>>>>> 015aaac3d09192f1835a0386a42cbbca9f2f7c7a
        except Exception as e:
            if verbose==1:
                print("exception: ",e)
            break

        # Compute adjoint lagrange multipliers
        L = adjoint_sol[1][:, 0][::-1]  # reverse time 0 --> T
        E = adjoint_sol[1][:, 2][::-1]
        assert (len(L) == num_tsteps) and (len(E) == num_tsteps), "Size mismatch"
        L = L / np.linalg.norm(L)  # normalize
        E = E / np.linalg.norm(E)

        # Update parameters
        # logger.info("Updating parameters")

        # Record parameters @ current step
        alpha_k = alpha
        beta_k = beta
        delta_k = delta
        #Rk = np.sqrt(np.sum(R ** 2))
        Rs=R[int(len(R)/5) :]
        Rk = np.sqrt(np.sum(Rs ** 2))  
        
        #compute d_1; distance of u0 signal
        i_1=1
        d_1=0
        while i_1<(len(u0)-1):
            i_1=i_1+1
            d_1=d_1+np.abs(u0[i_1-1]-u0[i_1])
  
        
        if verbose==1:
            print("")
            print("")
            print("New solution:")

            print(f"[{patience:d}:{iteration:d}] L2 Residual = {Rk:.4f} | alpha = {alpha_k:.4f}   "
            f"beta = {beta_k:.4f}   delta = {delta_k:.4f}")
            
            print(f"stiffness K = {K:.4f} dyne/cm^3    subglottal Ps = {Ps:.4f} dyne/cm^2   time_scaling = {time_scaling:.4f}")
            print("len(R)=",len(R)," len(u0)=",len(u0)," len(glottal_flow)=",len(glottal_flow))
            f_sum=np.sum(np.abs(u0[int(len(R)/5):]))
            l_sum=np.sum(np.abs(u0[:int(len(R)/5)]))
          
            print("f_sum = ",f_sum," l_sum = ",l_sum, "factor: = ",l_sum/f_sum,"d = ",d_1," d/len(u0) = ",d_1/len(u0)," g =",g_1," d/g =",d_1/g_1)
            
               
            plt.figure()
            fig, ax = plt.subplots(figsize=(20,3)) 
            plt.plot(sol[:, 0], glottal_flow, "k.-")
            plt.plot(sol[:, 0], u0, "b.-")
            #plt.plot(sol[:, 0], R, "r.-")
            plt.xlabel("t")
            plt.legend(["glottal flow", "estimated glottal flow", "residual"])
            plt.show()     
            
            t_max_1 = 500

            vdp_init_t_1 = 0.0
            vdp_init_state_1 = [0.0, 0.1, 0.0, 0.1]  # (xr, dxr, xl, dxl), xl=xr=0

            # vdp_params = [0.64, 0.32, 0.16]  # normal
            # vdp_params = [0.64, 0.32, 1.6]  # torus
            # vdp_params = [0.7, 0.32, 1.6]  # two cycle
            # vdp_params = [0.8, 0.32, 1.6]  # one cycle
            vdp_params_1 = alpha_k, beta_k, delta_k

            # Solve vocal fold displacement model
<<<<<<< HEAD
=======
            ode_t=ode_t-time.process_time() #cacluate ode time
>>>>>>> 015aaac3d09192f1835a0386a42cbbca9f2f7c7a
            sol_1 = ode_solver(
                vdp_coupled,
                vdp_jacobian,
                vdp_params_1,
                vdp_init_state_1,
                vdp_init_t_1,
                solver="lsoda",
                ixpr=0,
                dt=1,
                tmax=t_max_1,
            )
<<<<<<< HEAD
=======
            ode_t=ode_t+time.process_time() #cacluate ode time
>>>>>>> 015aaac3d09192f1835a0386a42cbbca9f2f7c7a

            # Get steady state
            Sr_1 = sol_1[int(t_max_1 / 2) :, [1, 2]]  # right states, (xr, dxr)
            Sl_1 = sol_1[int(t_max_1 / 2) :, [3, 4]]  # left states, (xl, dxl)

            # Plot states
            plt.figure()
            plt.subplot(121)
            plt.plot(Sl_1[:, 0], Sl_1[:, 1], 'k.-')
            #plt.xlabel(r'$\xi_r$')
            #plt.ylabel(r'$\dot{\xi}_r$')
            plt.tight_layout()
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
            plt.plot(Sr_1[:, 0], Sr_1[:, 1], 'k.-')
            #plt.xlabel(r'$\xi_l$')
            #plt.ylabel(r'$\dot{\xi}_l$')
            plt.figtext(0.5, 0.01, "Residual = {:.3f} , alpha = {:.3f} , beta = {:.3f} , delta = {:.3f}".format(Rk, alpha_k, beta_k, delta_k), wrap=True, horizontalalignment='center', fontsize=12)
            plt.tight_layout()
            ax = plt.gca()
            ax.axes.xaxis.set_visible(True)
            ax.axes.yaxis.set_visible(True)
            ax.axes.xaxis.set_ticks([])
            ax.axes.yaxis.set_ticks([])
            ax.axes.xaxis.set_ticklabels([])
            ax.axes.yaxis.set_ticklabels([])
            ax.set_facecolor('none')
            plt.grid(False)
            
            plt.show()

            

        
<<<<<<< HEAD

=======
>>>>>>> 015aaac3d09192f1835a0386a42cbbca9f2f7c7a
        if (Rk < Rk_best) and ((d_1/g_1)>(cut_off)) and ((d_1/g_1)<(1/cut_off)):  # has improvement
            #if (Rk < Rk_best):  # has improvement
            # Record best
            iteration_best = iteration
            R_best = R
            Rk_best = Rk
            alpha_best = alpha_k
            beta_best = beta_k
            delta_best = delta_k
            sol_best = sol
            u0_best = u0
            pv_best = np.array([alpha_best, beta_best, delta_best])  # param vector

            # Compute gradients
            d_alpha = -np.dot((dX[:num_tsteps, 0] + dX[:num_tsteps, 1]), (L + E))
            d_beta = np.sum(
                L * (1 + np.square(X[:num_tsteps, 0])) * dX[:num_tsteps, 0]
                + E * (1 + np.square(X[:num_tsteps, 1])) * dX[:num_tsteps, 1]
            )
            d_delta = np.sum(0.5 * (X[:num_tsteps, 1] * E - X[:num_tsteps, 0] * L))
            dpv = np.array([d_alpha, d_beta, d_delta])  # param grad vector
            dpv = dpv / np.linalg.norm(dpv)  # normalize
            d_alpha, d_beta, d_delta = dpv

            # Update
            alpha, beta, delta = optim_grad_step(
                alpha, beta, delta, d_alpha, d_beta, d_delta, stepsize=0.1
            )
            # alpha, beta, delta = optim_adapt_step(
            # alpha, beta, delta, d_alpha, d_beta, d_delta, default_step=0.1
            # )
            iteration += 1
            # logger.info(
            #     f"[{patience:d}:{iteration:d}] IMPROV: alpha = {alpha:.4f}   beta = {beta:.4f}   "
            #     f"delta = {delta:.4f}"
            # )
        else:  # no improvement
            patience = patience + 1

            # Compute conjugate gradients
            dpv = np.array([d_alpha, d_beta, d_delta])  # param grad vector
            dpv = dpv / np.linalg.norm(dpv)  # normalize
            ov = np.random.randn(len(dpv))  # orthogonal vector
            ov = ov - (np.dot(ov, dpv) / np.dot(dpv, dpv)) * dpv  # orthogonalize
            ov = ov / np.linalg.norm(ov)  # normalize
            d_alpha, d_beta, d_delta = ov

            # Reverse previous update & update in conjugate direction
            alpha, beta, delta = optim_grad_step(
                alpha_best, beta_best, delta_best, d_alpha, d_beta, d_delta, stepsize=0.1,
            )
            # alpha, beta, delta = optim_adapt_step(
            #     alpha_best,
            #     beta_best,
            #     delta_best,
            #     d_alpha,
            #     d_beta,
            #     d_delta,
            #     default_step=0.1,
            # )
            iteration += 1
            # logger.info(
            #     f"[{patience:d}:{iteration:d}] NO IMPROV: alpha = {alpha:.4f}   beta = {beta:.4f}   "
            #     f"delta = {delta:.4f}"
            # )

        while (alpha <= 0.01) or (beta <= 0.01) or (delta <= 0.01):  # if param goes below 0
            if_adjust = 1
            rv = np.random.randn(len(pv_best))  # radius
            rv = rv / np.linalg.norm(rv)  # normalize to 1
            pv = pv_best + 0.01 * rv  # perturb within a 0.01 radius ball
            alpha, beta, delta = pv
        if if_adjust:
            # logger.info(
            #     f"[{patience:d}:{iteration:d}] ADJUST: alpha = {alpha:.4f}   beta = {beta:.4f}   "
            #     f"delta = {delta:.4f}"
            # )
            if_adjust = 0

        
    best_results["iteration"].append(iteration_best)
    best_results["R"].append(R_best)
    best_results["Rk"].append(Rk_best)
    best_results["alpha"].append(alpha_best)
    best_results["beta"].append(beta_best)
    best_results["delta"].append(delta_best)
    best_results["sol"].append(sol_best)
    best_results["u0"].append(u0_best)
    if verbose==1:
            print(f"BEST@{iteration_best:d}: L2 Residual = {Rk_best:.4f} | alpha = {alpha_best:.4f}   "
            f"beta = {beta_best:.4f}   delta = {delta_best:.4f}")
    
    l = np.linspace(-5,5,100)
    p,r1,i1,r2,i2 = sys_eigenvals(l,alpha_best,beta_best,delta_best)
    
    res = {
        'alpha':float(alpha_best),
        'beta':float(beta_best),
        'delta':float(delta_best),
        'Rk':float(Rk_best),
        'distanceRatio':float(d_1/g_1),
        'eigenreal1':float(r1),
        'eigenreal2':float(r2),
        'eigensign':int(np.sign(r1*r2)),
        'timestamp': datetime.datetime.now().isoformat(),
<<<<<<< HEAD
=======
        'dae_time': dae_t,
        'ode_time': ode_t,
>>>>>>> 015aaac3d09192f1835a0386a42cbbca9f2f7c7a
    }

       # NOTE: If you want to plot glottal flow, estimatted glottal flow and residual
    plt.figure()
    fig, ax = plt.subplots(figsize=(20,3)) 
    plt.plot(sol_best[:, 0], glottal_flow, "k.-")
    plt.plot(sol_best[:, 0], u0_best, "b.-")
    #plt.plot(sol_best[:, 0], R_best, "r.-")
    plt.xlabel("t")
    plt.legend(["glottal flow", "estimated glottal flow", "residual"])
    plt.figure()
    plt.subplot(121)
    plt.plot(sol_best[:, 1], sol_best[:, 3], "b.-")
    plt.xlabel(r"$\xi_r$")
    plt.ylabel(r"$\xi_l$")
    plt.subplot(122)
    plt.plot(sol_best[:, 2], sol_best[:, 4], "b.-")
    plt.xlabel(r"$\dot{\xi}_r$")
    plt.ylabel(r"$\dot{\xi}_l$")
    plt.tight_layout()
    plt.show()

    

    return res
