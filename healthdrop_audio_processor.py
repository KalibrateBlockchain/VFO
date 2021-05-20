from __future__ import unicode_literals
import numpy as np
import argparse
import logging
import pwd
import grp
import os
import time
import audio_parser
from glob import glob
import shutil
from utils import *
from glottal_flow_extractor import glottal_flow_extractor
from vocal_fold_estimator import vocal_fold_estimator
from plot import plot_phasor, plot_mel
from tqdm import tqdm
import pandas as pd
from scipy.io import wavfile
from math import floor, ceil
import json
from pydub import AudioSegment

import matplotlib.pyplot as plt




import pylab
from scipy.fftpack import fft
import utils_odes
import IPython 
from IPython import embed 
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
import time
import datetime 
import json
import matplotlib.pyplot as plt
from math import pi, sin, sqrt, pow, floor, ceil
from external.pypevoc.speech.glottal import iaif_ola, lpcc2pole
import pylab
from PIL import Image
from utils_odes import residual_ode, ode_solver, ode_sys, physical_props
from utils_odes import foo_main, sys_eigenvals, plot_solution
from models.vocal_fold.vocal_fold_model_displacement import vdp_coupled, vdp_jacobian
from solvers.ode_solvers.ode_solver import ode_solver_1
from fitter import vfo_fitter


def run_cww_vfo(working_audio_file):
    ### audio_file = os.path.splitext(args.audio_file)[0]

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
        print(fname)
        from google.colab import drive
        drive.mount('/content/drive')
        CantAnalyze=plt.imread("Sample_files/CantAnalyze.png")
        TooNoisy=plt.imread("Sample_files/TooNoisy.png")

    if mode_of_processing==2:
        fname=(working_audio_file)
        CantAnalyze=plt.imread("Sample_files/CantAnalyze.png")
        TooNoisy=plt.imread("Sample_files/TooNoisy.png")

    file_audio, s_rate = sf.read(fname)
    file_audio = (file_audio / pow(2, 15)).astype("float32") # Convert from to 16-bit int to 32-bit float

    if mode_of_processing==1:
        fig, ax = plt.subplots(figsize=(20,3)) #display raw_audio entire
        plt.title('File Audio')
        ax.plot(file_audio)

    rw_audio=file_audio

    '''
    #Here's the code to extract noise clip from animated coached video
    ns_audio=numpy.concatenate(([int(s_rate*1.6):int(s_rate*1.9)], rw_audio[int(s_rate*2.45):int(s_rate*2.75)], rw_audio[int(s_rate*3.05):int(s_rate*3.35)])
    '''

    # Here's the noise clip extracted from the raw_audio beginning at 0.3 seconds, and ending at 1.0 seconds
    ns_audio = rw_audio[int(s_rate*.3):int(s_rate*0.5)]
    mean_noise=np.mean(np.abs(ns_audio))
    max_noise=0.000001

    working_audio_plot_filename=os.path.splitext(working_audio_file)[0] + "-plot.png"

    if mean_noise>max_noise:
        if mode_of_processing==1:
            plt.imshow(TooNoisy)
        if mode_of_processing==2:
            plt.imsave(working_audio_plot_filename, TooNoisy)
        return

    if mode_of_processing==1:
        fig, ax = plt.subplots(figsize=(20,3)) #display noise
        plt.title('Noise')
        ax.plot(ns_audio)
 
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
    while avg_audio[index] > (threshold*.2):
        end=index
        index=index+1

    rwt_audio = rw_audio[start:end]

    if mode_of_processing==1:
        fig, ax = plt.subplots(figsize=(20,3)) 
        plt.title('Trimmed Audio')
        ax.plot(rwt_audio)

    # trim 0.1 seconds from beginning and end
    rwt_audio = rwt_audio[int(s_rate * 0.1): int(len(rwt_audio)-(s_rate * 0.1))]    

    if mode_of_processing==1:
        fig, ax = plt.subplots(figsize=(20,3)) #display trimmed audio
        plt.title('Second Trimmed Audio')
        ax.plot(rwt_audio)

    t_trimmed = np.arange(len(rwt_audio))/s_rate
   
    # filter glotal signal
    gl_audio, dg, vt, gf = iaif_ola(rwt_audio, Fs=s_rate , tract_order=2 * int(np.round(s_rate / 2000)) + 4 , glottal_order=2 * int(np.round(s_rate / 4000)))

    if mode_of_processing==1:
        fig, ax = plt.subplots(figsize=(20,3)) #display glottal audio
        plt.title('Glottal Audio')
        ax.plot(gl_audio)

    # trim audio signal
    rwt_audio=rwt_audio[int(s_rate*.3): int(s_rate*0.6)]

    if mode_of_processing==1:
        fig, ax = plt.subplots(figsize=(20,3)) #display trimmed glottal audio
        plt.title('Third Trimmed Audio')
        ax.plot(gl_audio)

    # trim glotal signal
    gl_audio=gl_audio[int(s_rate*.3): int(s_rate*0.6)]

    if mode_of_processing==1:
        fig, ax = plt.subplots(figsize=(20,3)) #display trimmed glottal audio
        plt.title('Third Trimmed Glottal Audio')
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


    numberOfPeriods = int(12)

    if len(period)<numberOfPeriods:
        if mode_of_processing==1:
            plt.imshow(CantAnalyze)
        if mode_of_processing==2:
            plt.imsave(working_audio_plot_filename, CantAnalyze)
        return

    res1=vfo_fitter(gl_audio, rwt_audio, s_rate, period, numberOfPeriods)
    res2=vfo_fitter(gl_audio, rwt_audio, s_rate, period, numberOfPeriods)
    res3=vfo_fitter(gl_audio, rwt_audio, s_rate, period, numberOfPeriods)

    res=res1

    if (res2['eigensign'] == res3['eigensign']):
        res=res2

    #start of plotting code    
    t_max = 1000
    params = {'text.usetex': False, 'mathtext.fontset': 'stixsans'}
    plt.rcParams.update(params)  
    vdp_init_t = 0.0
    vdp_init_state = [0.0, 0.1, 0.0, 0.1]  # (xr, dxr, xl, dxl), xl=xr=0
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

    color='w.-'

    if mode_of_processing==1:
        color='k.-'

    # Plot states
    plt.figure()
    plt.subplot(121)
    plt.plot(Sl[:, 0], Sl[:, 1], color)
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
    plt.plot(Sr[:, 0], Sr[:, 1], color)
    #plt.plot(Sr[:, 0], Sr[:, 1], 'k.-')
    plt.ylabel('Right  Vocal Fold, λ = {:.9f}'.format(res['eigenreal2']), fontsize=10)
    #plt.figtext(0.6, 0.01, eigen)
    plt.figtext(0.59, 0.01, res['timestamp'])
    
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

    # convert to PIL Image object

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    pil_img = Image.open(buf)
    pil_img.show()
    buf.close()
    
    if mode_of_processing==1:
        plt.show() 

    if mode_of_processing==2:
        plt.savefig(working_audio_plot_filename, bbox_inches='tight',pad_inches = 0, transparent=True, edgecolor='none')


    #plt.close()
    #end of plotting code

    noise_threshold = 0.00000003
    if ((res['eigensign'])==1) and (np.mean(np.abs(ns_audio)) > noise_threshold):
        if mode_of_processing==1:
            plt.imshow(TooNoisy)
        if mode_of_processing==2:
            plt.imsave(working_audio_plot_filename, TooNoisy)
        return

    results_file = open(os.path.splitext(working_audio_file)[0] + ".json", "w")
    json.dump(res, results_file)
    results_file.close()
    
    return 
    


def run_analysis(wav_path,wav_chunk, sampling_rate):
    # Rita Original Method
    return run_analysis_RITA(wav_path,wav_chunk, sampling_rate)
    # Andres Method
    #return run_analysis_Andres(wav_path,wav_chunk, sampling_rate)
    # Calvin
    #return run_cww_vfo(wav_path)

def run_analysis_RITA(wav_path,wav_chunk, sampling_rate):
    # Save sampled audio clip
    # sample_rate, wav = wavfile.read(wav_file)
    # wav_chunk = wav[start_index:end_index]

    # Estimate glottis from IAIF and use that to get the alpha, beta, delta values by training against it
    
    g = glottal_flow_extractor(os.path.dirname(wav_path),wav_chunk, sampling_rate,section = -1)
    results = vocal_fold_estimator(wav_path,wav_chunk, sampling_rate, g, logging, t_patience = 100, section = -1)

    # # From csv get if the wav_file person has covid or not
    # a = wav_file.replace('_','|').split('|')
    # x = wav_label[wav_label["file_name"].str.contains(a[2])]
    # label = x['label'].max()

    # Save phasor plot and mel spectrogram for IAIF and estimated one

    Sr, Sl = plot_phasor(wav_path, wav_chunk, results['alpha'][-1], results['beta'][-1], results['delta'][-1], "", g, sampling_rate)
    # S.append([wav_file, label, Sr, Sl, results["alpha"][-1], results["beta"][-1], results["delta"][-1]])
    # plot_mel(wav_file, sample_rate, g, results['u0'], plot_dir_mel_spectrogram_true, plot_dir_mel_spectrogram_est, results)

    return  {"glottal_waveform": g,
               "estimated_glottal_waveform": results["u0"],
               "Sr": Sr,
               "Sl": Sl,
               "alpha": results["alpha"][-1],
               "beta": results["beta"][-1],
               "delta": results["delta"][-1],
               # "R": results["R"][-1],
               "Rk": results["Rk"][-1]
              }

def run_analysis_Andres(wav_path,wav_chunk, sampling_rate):
    t0 = time.process_time() # Here start count time
    t, signal , glot_flow, sr = audio_parser.load_audio(wav_path)
    
    #CISCO  np.save(save_dir + "/" + wav_file.replace('/', '_')  + ".npy", g)
    #np.save(wav_file_path+os.path.splitext(os.path.basename(wav_file_path))[0] + ".npy", g)

    plt.plot(t, signal, 'r-', alpha=0.5, label="audio")
    plt.plot(t[:-1], glot_flow, 'b-', alpha=0.9, label="Glottal flow")
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.legend(loc='best')
    plt.ylabel('audio signal')
    plt.xlabel('$t$')
    plt.tight_layout()
    #plt.show()
    
    plt.savefig(os.path.splitext(wav_path)[0] + "-plot.png", bbox_inches='tight',pad_inches = 0, transparent=True, edgecolor='none')

    #print('audio signal',type(signal),signal.shape)
    #print('audio time',type(t),t.shape)
    
    #t1 = time.process_time() # Here end counting time
    
    #print("Elapsed time to solve: ",t1-t0)


def export_results(output_file, object):
    with open(output_file, "w") as f:
        np.save(output_file, object)

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

def run_export_analysis(args):
    # User Have a Folder in the data_dir/UID/TestID/
    # in folder created SampleX.WAV the standard WAV files
    # in folder created SampleXTrimmed.WAV the silence removed WAV files
    # Create SampleX.analysis.png the COVID pic

    working_filepath = args.data_dir+os.path.sep+args.user_id+os.path.sep

    sampling_rate, wav_sample = wavfile.read(working_filepath+os.path.splitext(args.audio_file)[0] + "-sample-trimmed.WAV")
    #len(wav_samples)-window_size
    if args.verbose_mode ==1:
        print('Running analysis')
    #wav_chunk = wav_sample[0:800]
    wav_chunk = wav_sample

    #ANALYSIS: NEW METHOD 
    #run_cww_vfo(working_filepath+os.path.splitext(args.audio_file)[0] + "-sample-trimmed.WAV")
        
    #ANALYSIS: OLD METHOD 
    analysis_result = run_analysis(working_filepath+os.path.splitext(args.audio_file)[0] + "-sample-trimmed.WAV",wav_chunk, sampling_rate)
    output_filename = os.path.splitext(args.audio_file)[0] + ".npy"
    export_results(working_filepath+output_filename, analysis_result)

def run_spectrogram_generator(args):
    # User Have a Folder in the data_dir/UID/TestID/
    # in folder created SampleX.WAV the standard WAV files
    # Created SampleX.png the spectrogram of the WAV file
    working_filepath = args.data_dir+os.path.sep+args.user_id+os.path.sep

    sampling_rate, wav_sample = wavfile.read(working_filepath+os.path.splitext(args.audio_file)[0] + "-sample-trimmed.WAV")
    #Check if wave file is 16bit or 32 bit. 24bit is not supported
    wav_data_type = wav_sample.dtype
    #We can convert our sound array to floating point values ranging from -1 to 1 as follows
    wav_sample = wav_sample / (2.**15)

    #Check sample points and sound channel for duel channel(5060, 2) or  (5060, ) for mono channel
    sound_shape = wav_sample.shape
    sample_points = float(wav_sample.shape[0])

    #Get duration of sound file
    signal_duration =  wav_sample.shape[0] / sampling_rate

    #If two channels, then select only one channel
    #wav_sample_one_channel = wav_sample[0::2]
    wav_sample_one_channel = wav_sample
    

    #Plotting the tone
    #We can represent sound by plotting the pressure values against time axis.
    #Create an array of sample point in one dimension
    time_array = np.arange(0, sample_points, 1)
    time_array = time_array / sampling_rate

    #Scale to milliSeconds
    time_array = time_array * 1000 

    #Plot the tone
    plt.plot(time_array, wav_sample_one_channel, color='w')
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    ax.set_facecolor('none')
    plt.grid(False)
    #plt.xlabel('')
    #plt.ylabel('')
    output_filename = os.path.splitext(args.audio_file)[0] + ".png"
    plt.savefig(working_filepath+output_filename, bbox_inches='tight',pad_inches = 0, transparent=True, edgecolor='none')
    plt.clf()
    plt.cla()
    plt.close()


def run_convert_audio_file(args):
    # User Have a Folder in the data_dir/UID/TestID/
    # app recording SampleX.3pg the standard for android
    # app recording SampleX.caf the standard for iOS
    # Create SampleX.WAV the standard WAV files
    working_filepath = args.data_dir+os.path.sep+args.user_id+os.path.sep
    if args.audio_file.endswith(".3gp"):
        os.system("ffmpeg -i "+working_filepath+args.audio_file+" -y -ss 1 -t 1 -ab 256k -ar 16k "+working_filepath+os.path.splitext(args.audio_file)[0] + "-sample.WAV") #ffmpeg to wav
        if args.verbose_mode ==1:
            print('')
            print('Convted file {}'.format(args.audio_file))

    if args.audio_file.endswith(".caf"):
        os.system("ffmpeg -i "+working_filepath+args.audio_file+" -y -ss 1 -t 1 -ac 1 -ab 256k -ar 16k "+working_filepath+os.path.splitext(args.audio_file)[0] + "-sample.WAV") #ffmpeg to wav
        if args.verbose_mode ==1:
            print('')
            print('Convted file {}'.format(args.audio_file))

    trimm_audio = False
    if trimm_audio:
        # This area is going to read the sample, and create a trimmed for silence
        sound = AudioSegment.from_file(working_filepath+os.path.splitext(args.audio_file)[0] + "-sample.WAV", format="wav")
        start_trim = detect_leading_silence(sound)
        end_trim = detect_leading_silence(sound.reverse())

        duration = len(sound)
        if duration-end_trim-start_trim > 2000:
            end_trim=duration-start_trim-2000
        trimmed_sound = sound[start_trim:duration-end_trim]
        trimmed_sound.export(working_filepath+os.path.splitext(args.audio_file)[0] + "-sample-trimmed.WAV", format="wav")
    else:
        shutil.copyfile(working_filepath+os.path.splitext(args.audio_file)[0] + "-sample.WAV",working_filepath+os.path.splitext(args.audio_file)[0] + "-sample-trimmed.WAV")

def run_set_scan_process(args):
    working_filepath = '/var/www/html/process_samples/'+args.user_id+'.'+os.path.splitext(args.audio_file)[0]+'.job'
    #args.data_dir ='/var/www/html/process_samples' 
    #Save the process job file
    with open(working_filepath, 'wt') as outfile:
    	json.dump(vars(args), outfile)

def process_file(args):
    logging.debug('process_file')
    if args.verbose_mode == 1:
        print('Processing Audio File')

    if args.mode == '0':
        #Convert file to WAV
        # ex: Sample[1-(NumberOfTests)].3gp (android) -> 16bit WAV
        run_convert_audio_file(args)

        #Create Spectrogram of WAV
        run_spectrogram_generator(args)

    if args.mode == '1':
        #Analyze WAV
        run_export_analysis(args)

    if args.mode == '2':
        logging.debug('Conver Audio Spectrogram and Analysis Mode:2')
        logging.debug('Ownership folder fixing:'+args.data_dir+os.path.sep+args.user_id)
        #uid=pwd.getpwnam('cisco').pw_uid
        #gid=grp.getgrnam('www-data').gr_gid
        #os.chown(args.data_dir+os.path.sep+args.user_id,uid,gid)
        #logging.debug('Ownership folder fixed')
        #logging.debug('Ownership file fixing inside:'+args.data_dir+os.path.sep+args.user_id)
        #shutil.chown(args.data_dir+os.path.sep+args.user_id,group='www-data',recursive=True)
        os.system('sudo -u root chown -R www-data:www-data '+args.data_dir+os.path.sep+args.user_id)
        logging.debug('Ownership files fixed')
        logging.debug('run_convert_audio_file(args)')
        run_convert_audio_file(args)
        logging.debug('run_spectrogram_generator(args)')
        run_spectrogram_generator(args)
        logging.debug('run_export_analysis(args)')
        run_export_analysis(args)

    if args.mode == '4':
        logging.debug('Conver Audio Spectrogram and Analysis Mode:4')
        logging.debug('Ownership folder fixing:'+args.data_dir+os.path.sep+args.user_id)
        os.system('sudo -u root chown -R www-data:www-data '+args.data_dir+os.path.sep+args.user_id)
        logging.debug('Ownership files fixed')
        run_convert_audio_file(args)
        run_spectrogram_generator(args)
        run_set_scan_process(args) 


if __name__ == '__main__':
    ###os.system('cd /home/cisco/VFO')
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
    process_file(args)
