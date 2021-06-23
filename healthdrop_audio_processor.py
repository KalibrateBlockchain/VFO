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
from CWWmain import CWWmain



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
    CWWmain(working_audio_file,2)
    return 
    


def run_analysis(wav_path,wav_chunk, sampling_rate):
    # Rita Original Method
    #return run_analysis_RITA(wav_path,wav_chunk, sampling_rate)
    # Andres Method
    #return run_analysis_Andres(wav_path,wav_chunk, sampling_rate)
    # Calvin
    #print (wav_path)
    return CWWmain(wav_path,2)

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

    #sampling_rate, wav_sample = wavfile.read(working_filepath+os.path.splitext(args.audio_file)[0] + "-sample-trimmed.WAV")
    #len(wav_samples)-window_size
    #if args.verbose_mode ==1:
    #    print('Running analysis')
    #wav_chunk = wav_sample[0:800]
    #wav_chunk = wav_sample

    #ANALYSIS: NEW METHOD 
    #run_cww_vfo(working_filepath+os.path.splitext(args.audio_file)[0] + "-sample-trimmed.WAV")
        
    #ANALYSIS: OLD METHOD 
    #run_analysis(working_filepath+os.path.splitext(args.audio_file)[0] + "-sample.WAV",None, None)
    run_analysis(working_filepath+args.audio_file,None, None)

    #output_filename = os.path.splitext(args.audio_file)[0] + ".npy"
    #export_results(working_filepath+output_filename, analysis_result)

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
    return

    
    working_filepath = args.data_dir+os.path.sep+args.user_id+os.path.sep
    if args.audio_file.endswith(".mp4"):
        os.system("ffmpeg -i "+working_filepath+args.audio_file+" -y -ab 256k -ar 16k "+working_filepath+os.path.splitext(args.audio_file)[0] + ".wav") #ffmpeg to wav
        if args.verbose_mode ==1:
            print('')
            print('Convted file {}'.format(args.audio_file))
        args.audio_file = os.path.splitext(args.audio_file)[0] + ".wav"


    if args.audio_file.endswith(".3gp"):
        os.system("ffmpeg -i "+working_filepath+args.audio_file+" -y -ab 256k -ar 16k "+working_filepath+os.path.splitext(args.audio_file)[0] + ".wav") #ffmpeg to wav
        if args.verbose_mode ==1:
            print('')
            print('Convted file {}'.format(args.audio_file))
        args.audio_file = os.path.splitext(args.audio_file)[0] + ".wav"

    if args.audio_file.endswith(".caf"):
        os.system("ffmpeg -i "+working_filepath+args.audio_file+" -y -ac 1 -ab 256k -ar 16k "+working_filepath+os.path.splitext(args.audio_file)[0] + ".wav") #ffmpeg to wav
        if args.verbose_mode ==1:
            print('')
            print('Convted file {}'.format(args.audio_file))
        args.audio_file = os.path.splitext(args.audio_file)[0] + ".wav"

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
        logging.debug('Converting Audofile')
        run_convert_audio_file(args)
        logging.debug('Generating Spectrogram')
        run_spectrogram_generator(args)
        logging.debug('Adding to the Scan Queue')
        run_set_scan_process(args) 

    if args.mode == '5':
        logging.debug('Analysis File Direct Mode:5')
        logging.debug('Ownership folder fixing:'+args.data_dir+os.path.sep+args.user_id)
        os.system('sudo -u root chown -R www-data:www-data '+args.data_dir+os.path.sep+args.user_id)
        logging.debug('Ownership files fixed')
        logging.debug('Converting Audofile')
        run_convert_audio_file(args)
        logging.debug('Adding to the Scan Queue')
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
