import numpy as np
import os
from matplotlib import pyplot as plt
import logging
from scipy.stats import chisquare
#from utils_odes import sys_eigenvals

from models.vocal_fold.vocal_fold_model_displacement import vdp_coupled, vdp_jacobian
from models.vocal_fold.adjoint_model_displacement import adjoint_model
from solvers.ode_solvers.ode_solver import ode_solver
from solvers.ode_solvers.dae_solver import dae_solver
#from utils_odes import sys_eigenvals

import librosa
import librosa.display

def plot_phasor(wav_file, wav_chunk, alpha, beta, delta, output_dir, g, sampling_rate):
    """
    Input: results, save_dir
    Output: save plot
    
   """
   
   # Initial conditions
    t_max = 500
    params = {'text.usetex': False, 'mathtext.fontset': 'stixsans'}
    plt.rcParams.update(params)
    
    vdp_init_t = 0.0
    vdp_init_state = [0.0, 0.1, 0.0, 0.1]  # (xr, dxr, xl, dxl), xl=xr=0

    # vdp_params = [0.64, 0.32, 0.16]  # normal
    # vdp_params = [0.64, 0.32, 1.6]  # torus
    # vdp_params = [0.7, 0.32, 1.6]  # two cycle
    # vdp_params = [0.8, 0.32, 1.6]  # one cycle
    vdp_params = [alpha, beta, delta]

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
    
    Rkk, length = calc_RK(wav_chunk, sampling_rate, g, alpha, beta, delta)
    Rkk=Rkk/length
    
    ch,p=chisquare(g,wav_chunk)

    # Plot states
    plt.figure()
    plt.subplot(121)
    plt.plot(Sl[:, 0], Sl[:, 1], 'k.-')
    #plt.xlabel(r'$\xi_r$')
    plt.ylabel('Left')

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
    #plt.xlabel(r'$\xi_l$')
    plt.ylabel('Right')
    plt.figtext(0.5, 0.01, "Rk = {:.5f}, alpha = {:.3f} , beta = {:.3f} , delta = {:.3f}".format(Rkk, alpha, beta, delta), wrap=True, horizontalalignment='center', fontsize=12)
    #plt.figtext(0.5, 0.01, "Residual", fontfamily="sans-serif" )
    

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

    #CISCO 
    plt.savefig(os.path.splitext(wav_file)[0] + "-plot.png", bbox_inches='tight',pad_inches = 0, transparent=True, edgecolor='none')

    plt.clf()
    plt.cla()
    plt.close()
    

    return Sr, Sl

def plot_mel(wav_file, sample_rate, g, g_est, plot_dir_mel_spectrogram_true, plot_dir_mel_spectrogram_est, results):
    """
    Input: wav_file: name of wav_file to be operated on
           sample_rate: sampling rate used
           g: glottal flow by IAIF
           g_est: estimated glottal flow by solving Adjoint model
           plot_dir_mel_spectrogram_true: IAIF glow flow mel spectrogram plot directory
           plot_dir_mel_spectrogram_est: estimated glottal flow spectrogram plot directory
    Output: saves plot
    """ 

    n_fft = 1024
    hop_length = 256
    n_mels = 128
    
    # plot time glottal waveform true
    plt.xlabel('time')
    plt.ylabel('Amplitude')
    plt.title('Glottal waveform')
    plt.tight_layout()
    plt.figtext(0.5, 0.01, "Residual = {:.3f} , alpha = {:.3f} , beta = {:.3f} , delta = {:.3f}".format(results["Rk"][-1], results["alpha"][-1], results["beta"][-1], results["delta"][-1]), wrap=True, horizontalalignment='center', fontsize=12)
    plt.plot(g)
    plt.savefig(plot_dir_mel_spectrogram_true + '/' + wav_file.replace('/', '_') + '_time_'+".png")
    plt.clf()
    plt.cla()
    plt.close()

    # plot mel glottal waveform true
    S = librosa.feature.melspectrogram(g, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    S_DB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_DB, sr=sample_rate, hop_length=hop_length, x_axis='time', y_axis='mel')
    plt.title('Mel Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.figtext(0.5, 0.01, "Residual = {:.3f} , alpha = {:.3f} , beta = {:.3f} , delta = {:.3f}".format(results["Rk"][-1], results["alpha"][-1], results["beta"][-1], results["delta"][-1]), wrap=True, horizontalalignment='center', fontsize=12)
    plt.tight_layout()
    plt.savefig(plot_dir_mel_spectrogram_true + '/' + wav_file.replace('/', '_') + '_mel_' + ".png")
    plt.clf()
    plt.cla()
    plt.close()

    # plot time glottal waveform estimated
    plt.xlabel('time')
    plt.ylabel('Amplitude')
    plt.title('Estimated Glottal waveform')
    plt.tight_layout()
    plt.figtext(0.5, 0.01, "Residual = {:.3f} , alpha = {:.3f} , beta = {:.3f} , delta = {:.3f}".format(results["Rk"][-1], results["alpha"][-1], results["beta"][-1], results["delta"][-1]), wrap=True, horizontalalignment='center', fontsize=12)
    plt.plot(g_est[0])
    plt.savefig(plot_dir_mel_spectrogram_est + '/' + wav_file.replace('/', '_') + '_time_'+".png")
    plt.clf()
    plt.cla()
    plt.close()

    # plot mel glottal waveform est
    S = librosa.feature.melspectrogram(g_est[0], sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    S_DB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_DB, sr=sample_rate, hop_length=hop_length, x_axis='time', y_axis='mel')
    plt.title('Mel Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.figtext(0.5, 0.01, "Residual = {:.3f} , alpha = {:.3f} , beta = {:.3f} , delta = {:.3f}".format(results["Rk"][-1], results["alpha"][-1], results["beta"][-1], results["delta"][-1]), wrap=True, horizontalalignment='center', fontsize=12)
    plt.tight_layout()
    plt.savefig(plot_dir_mel_spectrogram_est + '/' + wav_file.replace('/', '_') + '_mel_' + ".png")
    plt.clf()
    plt.cla()
    plt.close()
    
    
def calc_RK(wav_chunk, sample_rate, glottal_flow, alpha, beta, delta):
     
    logger = logging.getLogger('my_logger')
    logging.basicConfig(level=logging.DEBUG)

    logger.info(
            f"sample_rate {sample_rate:.4f} | alpha = {alpha:.4f}   "
            f"beta = {beta:.4f}   delta = {delta:.4f}"
        )
    
    # Set constants
    M = 0.5  # mass, g/cm^2
    B = 100  # damping, dyne s/cm^3
    d = 1.75  # length of vocal folds, cm
    x0 = 0.1  # half glottal width at rest position, cm
    tau = 1e-3  # time delay for surface wave to travel half glottal height, ms
    c = 5000  # air particle velocity, cm/s
    eta = 1.0  # nonlinear factor for energy dissipation at large amplitude

    vdp_init_t = 0.0
    vdp_init_state = [0.0, 0.1, 0.0, 0.1]  # (xr, dxr, xl, dxl), xl=xr=0
    num_tsteps = len(wav_chunk)  # total number of time steps
    T = len(wav_chunk) / float(sample_rate)  # total time, s

    K = B ** 2 / (beta ** 2 * M)
    Ps = (alpha * x0 * np.sqrt(M * K)) / tau
    time_scaling = np.sqrt(K / float(M))  # t -> s
    x_scaling = np.sqrt(eta)

    vdp_params = [alpha, beta, delta]
    
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
    
    if len(sol) > len(wav_chunk):
        sol = sol[:-1]   
    
    X = sol[:, [1, 3]]  # vocal fold displacement (right, left), cm
    dX = sol[:, [2, 4]]  # cm/s
    u0 = c * d * (np.sum(X, axis=1) + 2 * x0)  # volume velocity flow, cm^3/s
    u0 = u0 / np.linalg.norm(u0) * np.linalg.norm(glottal_flow)  # normalize

    logger.info(
            f"time_scaling {time_scaling:.4f} | T = {T:.4f}   "
            f"len(wav_chunk) = {len(wav_chunk):.4f}   len(glottal_flow) = {len(glottal_flow):.4f} len(u0) = {len(u0):4f}"
        )
    
    R = u0 - glottal_flow
    Rk = np.sqrt(np.sum(R ** 2))
    logger.info(f"Rk = {Rk:.4f}")
    
    
    return Rk, len(wav_chunk)
