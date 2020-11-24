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

if __name__ == '__main__':
    """
    Input: data_dir, exp_name, output_dir
    Return: saves phase plot to corresponding output directory
            saves .csv file with directory of wav file, alpha, beta, delta]
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_dir", "-data_dir", type=str, default="data")
    parser.add_argument("-exp_name", "-exp_name", type=str, required=True)
    parser.add_argument("-output_dir", "-output_dir", type=str, default="output")
    parser.add_argument("-section", '-section', type=int, default = 1)
    args = parser.parse_args()

    # Setting up logging
    output_dir = args.output_dir + '/' + args.exp_name
    create_logging(output_dir + '/logs', filemode = 'w')
    logging.info('logging started for experiment = {}'.format(args.exp_name))

    # Get all wav_files in list
    # wav_files = [f for f in glob(args.data_dir+'/**', recursive = True) if os.path.isfile(f)]
    header_list = ["file_name", "label"]
    wav_files = [f for f in glob(args.data_dir + '/**', recursive = True) if os.path.isfile(f)]
    wav_files = [f for f in wav_files if f.split('.')[-1] == 'wav']
    wav_label = pd.read_csv("data/all_file_aug_22_label.csv", names = header_list)

    # Directory path for saving extracted glottal waveform in .npy format
    glottal_dir = output_dir + '/extracted_glottal_flow'
    plot_dir_covid = output_dir + '/plots_phasor_covid'
    plot_dir_normal = output_dir + '/plots_phasor_normal'
    plot_dir_mel_spectrogram_true = output_dir + '/plots_mel_spectrogram_true'
    plot_dir_mel_spectrogram_est = output_dir + '/plots_mel_spectrogram_est'
    proc_data_dir = output_dir + '/proc_data'

    create_folder_rm(glottal_dir)
    create_folder_rm(plot_dir_covid)
    create_folder_rm(plot_dir_normal)
    create_folder_rm(plot_dir_mel_spectrogram_true)
    create_folder_rm(plot_dir_mel_spectrogram_est)
    create_folder_rm(proc_data_dir)

    results_arr = []
    S = []
    for i, wav_file in tqdm(enumerate(wav_files)):

        # Save sampled audio clip
        sample_rate, wav = wavfile.read(wav_file)
        if args.section == 1:
            wav = wav[floor(len(wav)/2): ceil(len(wav)/2 + sample_rate)]
        wavfile.write(proc_data_dir + '/' + wav_file.replace('/', '_') + ".wav", sample_rate, wav)

        # Estimate glottis from IAIF and use that to get the alpha, beta, delta values by training against it
        g = glottal_flow_extractor(wav_file, glottal_dir, section = args.section)
        results = vocal_fold_estimator(wav_file, g, logging, t_patience = 100, section = args.section)

        # From csv get if the wav_file person has covid or not
        a = wav_file.replace('_','|').split('|')
        x = wav_label[wav_label["file_name"].str.contains(a[2])]
        label = x['label'].max()

        # Save phasor plot and mel spectrogram for IAIF and estimated one
        Sr, Sl = plot_phasor(wav_file, results, plot_dir_covid if label else plot_dir_normal)
        S.append([wav_file, label, Sr, Sl, results["alpha"][-1], results["beta"][-1], results["delta"][-1]])
        plot_mel(wav_file, sample_rate, g, results['u0'], plot_dir_mel_spectrogram_true, plot_dir_mel_spectrogram_est, results)

        # Append the results and dump in a .csv
        results_arr.append([
            wav_file, results["iteration"][-1], results["Rk"][-1],
            results["alpha"][-1], results["beta"][-1], results["delta"][-1],
            label
            # 1 if '_c' in label else 0
        ])
        np.savetxt(output_dir+"/results.csv", np.asarray(results_arr), delimiter=",", fmt='%s', header="file_name,iteration,R,alpha,beta,delta,is_covid")
        break

    np.savetxt(output_dir + '/timeseries.npy', np.array(S), fmt='%s')
    print('.:: Complete ::.')
