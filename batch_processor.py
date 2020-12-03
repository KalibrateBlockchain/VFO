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

def run_analysis(wav_file_path,wav_chunk, sampling_rate):
    # Save sampled audio clip
    # sample_rate, wav = wavfile.read(wav_file)
    # wav_chunk = wav[start_index:end_index]

    # Estimate glottis from IAIF and use that to get the alpha, beta, delta values by training against it
    print(wav_file_path)
    print(sampling_rate)
    print(wav_chunk)
    g = glottal_flow_extractor(wav_file_path,wav_chunk, sampling_rate, section = -1)
    results = vocal_fold_estimator(wav_file_path,wav_chunk, sampling_rate, g, logging, t_patience = 100, section = -1)

    # # From csv get if the wav_file person has covid or not
    # a = wav_file.replace('_','|').split('|')
    # x = wav_label[wav_label["file_name"].str.contains(a[2])]
    # label = x['label'].max()

    # Save phasor plot and mel spectrogram for IAIF and estimated one
    #CISCO Sr, Sl = plot_phasor(wav_chunk, results, "")
    Sr, Sl = plot_phasor(wav_file_path, results, "")
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

def export_results(output_filepath, object):
    with open(output_filepath, "w") as f:
        np.save(output_filepath, object)

def run_export_analysis(wav_file_path, window_size, overlap):
    all_results = {}
    print("Processing file: {}".format(wav_file_path))
    sampling_rate, wav_samples = wavfile.read(wav_file_path)

    for start_index in range(0, len(wav_samples)-window_size, overlap):
        print("Chunk percentage: {}".format(start_index/len(wav_samples)))
        wav_chunk = wav_samples[start_index:start_index+window_size]
        # CISCO results = run_analysis(wav_chunk, sampling_rate)
        results = run_analysis(os.path.splitext(os.path.basename(wav_file_path))[0]+"_"+str(start_index)+".wav",wav_chunk, sampling_rate)
        all_results["{}-{}".format(start_index, start_index+window_size)] = results
        # break

    output_filename = os.path.splitext(os.path.basename(wav_file_path))[0] + ".npy"
    output_filepath = os.path.join(os.path.dirname(wav_file_path), output_filename)
    export_results(output_filepath, all_results)

def process_files(args):
    for root, dirs, files in os.walk(args.data_dir):
        for file in files:
            if file.endswith("_split.wav"):
                print("Found: {}".format(file))
               	run_export_analysis(os.path.join(root, file), args.window_size, args.overlap)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--window_size", type=int, default=400)
    parser.add_argument("--overlap", type=int, default=200)
    args = parser.parse_args()
    process_files(args)
