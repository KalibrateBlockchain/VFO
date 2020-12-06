import numpy as np
import argparse
import logging
import os
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

import matplotlib.pyplot as plt
import pylab
from scipy.fftpack import fft


def run_analysis(wav_path,wav_chunk, sampling_rate):
    # Save sampled audio clip
    # sample_rate, wav = wavfile.read(wav_file)
    # wav_chunk = wav[start_index:end_index]

    # Estimate glottis from IAIF and use that to get the alpha, beta, delta values by training against it
    g = glottal_flow_extractor(wav_path,wav_chunk, sampling_rate,section = -1)
    results = vocal_fold_estimator(wav_path,wav_chunk, sampling_rate, g, logging, t_patience = 100, section = -1)

    # # From csv get if the wav_file person has covid or not
    # a = wav_file.replace('_','|').split('|')
    # x = wav_label[wav_label["file_name"].str.contains(a[2])]
    # label = x['label'].max()

    # Save phasor plot and mel spectrogram for IAIF and estimated one
    Sr, Sl = plot_phasor(wav_path, results, "")
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

def export_results(output_file, object):
    with open(output_file, "w") as f:
        np.save(output_file, object)

def run_export_analysis(args):
    # User Have a Folder in the data_dir/UID/TestID/
    # in folder created SampleX.WAV the standard WAV files
    # Create SampleX.analysis.png the COVID pic

    working_filepath = args.data_dir+os.path.sep+args.user_id+os.path.sep+args.test_id+os.path.sep
    sampling_rate, wav_sample = wavfile.read(working_filepath+os.path.splitext(args.audio_file)[0] + ".WAV")
    #len(wav_samples)-window_size
    if args.verbose_mode ==1:
        print('Running analysis')
    #wav_chunk = wav_sample[0:800]
    wav_chunk = wav_sample
    analysis_result = run_analysis(working_filepath+os.path.splitext(args.audio_file)[0] + ".WAV",wav_chunk, sampling_rate)

    # in folder created SampleX.npy the results of the analysis
    output_filename = os.path.splitext(args.audio_file)[0] + ".npy"
    export_results(working_filepath+output_filename, analysis_result)

def run_spectrogram_generator(args):
    # User Have a Folder in the data_dir/UID/TestID/
    # in folder created SampleX.WAV the standard WAV files
    # Created SampleX.png the spectrogram of the WAV file
    working_filepath = args.data_dir+os.path.sep+args.user_id+os.path.sep+args.test_id+os.path.sep

    sampling_rate, wav_sample = wavfile.read(working_filepath+os.path.splitext(args.audio_file)[0] + ".WAV")
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
    working_filepath = args.data_dir+os.path.sep+args.user_id+os.path.sep+args.test_id+os.path.sep
    if args.audio_file.endswith(".3gp"):
        os.system("ffmpeg -i "+working_filepath+args.audio_file+" -y -ss 1 -t 1 -ab 256k -ar 16k "+working_filepath+os.path.splitext(args.audio_file)[0] + ".WAV") #ffmpeg to wav
        if args.verbose_mode ==1:
            print('')
            print('Convted file {}'.format(args.audio_file))



def process_file(args):
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
        run_convert_audio_file(args)
        run_spectrogram_generator(args)
        run_export_analysis(args)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--user_id", type=str,required=True)
    parser.add_argument("--test_id", type=str,required=True)
    parser.add_argument("--audio_file", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True)
    parser.add_argument("--verbose_mode", type=str, required=False)
    args = parser.parse_args()
    process_file(args)
