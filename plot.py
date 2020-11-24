import numpy as np
from matplotlib import pyplot as plt

from models.vocal_fold.vocal_fold_model_displacement import vdp_coupled, vdp_jacobian
from models.vocal_fold.adjoint_model_displacement import adjoint_model
from solvers.ode_solvers.ode_solver import ode_solver
from solvers.ode_solvers.dae_solver import dae_solver

import librosa
import librosa.display

def plot_phasor(wav_file, results, output_dir):
    """
    Input: results, save_dir
    Output: save plot
    """
    # Initial conditions
    t_max = 500

    vdp_init_t = 0.0
    vdp_init_state = [0.0, 0.1, 0.0, 0.1]  # (xr, dxr, xl, dxl), xl=xr=0

    # vdp_params = [0.64, 0.32, 0.16]  # normal
    # vdp_params = [0.64, 0.32, 1.6]  # torus
    # vdp_params = [0.7, 0.32, 1.6]  # two cycle
    # vdp_params = [0.8, 0.32, 1.6]  # one cycle
    vdp_params = [results['alpha'][-1], results['beta'][-1], results['delta'][-1]]

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

    # Plot states
    plt.figure()
    plt.subplot(121)
    plt.plot(Sr[:, 0], Sr[:, 1], 'b.-')
    plt.xlabel(r'$\xi_r$')
    plt.ylabel(r'$\dot{\xi}_r$')
    plt.subplot(122)
    plt.plot(Sl[:, 0], Sl[:, 1], 'b.-')
    plt.xlabel(r'$\xi_l$')
    plt.ylabel(r'$\dot{\xi}_l$')
    plt.figtext(0.5, 0.01, "Residual = {:.3f} , alpha = {:.3f} , beta = {:.3f} , delta = {:.3f}".format(results["Rk"][-1], results["alpha"][-1], results["beta"][-1], results["delta"][-1]), wrap=True, horizontalalignment='center', fontsize=12)
    plt.tight_layout()
    # plt.show()
    plt.savefig(output_dir + '/' + wav_file.replace('/', '_') + ".png")
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