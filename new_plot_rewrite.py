import numpy as np
import argparse
import logging
import os
from glob import glob
import shutil
from utils import *
import pandas as pd
from shutil import copyfile

if __name__ == '__main__':

    data_dir = 'output/exp_covid_vowel_section_1_with_melplots'
    covid_out_est_covid_dir = 'output/exp_covid_vowel_section_1_with_melplots/covid_plots_mel_spectrogram_est'
    normal_out_est_covid_dir = 'output/exp_covid_vowel_section_1_with_melplots/normal_plots_mel_spectrogram_est'
    covid_out_true_covid_dir = 'output/exp_covid_vowel_section_1_with_melplots/covid_plots_mel_spectrogram_true'
    normal_out_true_covid_dir = 'output/exp_covid_vowel_section_1_with_melplots/normal_plots_mel_spectrogram_true'

    for name in [covid_out_est_covid_dir, normal_out_est_covid_dir, covid_out_true_covid_dir, normal_out_true_covid_dir]:
        create_folder_rm(name)

    results = pd.read_csv(data_dir + '/results.csv')
    re_covid = []
    re_normal = []
    for i in range(0, results.shape[0]):
        if results.iloc[i,6]:
            re_covid.append(results.iloc[i,0].split('_UPID-')[1].split('.wav')[0])
        else:
            re_normal.append(results.iloc[i,0].split('_UPID-')[1].split('.wav')[0])

    for file_name in glob(data_dir + '/plots_mel_spectrogram_est/*.png'):
        id = file_name.split('/')[-1].split('_UPID-')[-1].split('.wav')[0]
        if id in re_covid:
            copyfile(file_name, file_name.replace('plots_mel_spectrogram_est', 'covid_plots_mel_spectrogram_est'))
        else:
            copyfile(file_name, file_name.replace('plots_mel_spectrogram_est', 'normal_plots_mel_spectrogram_est'))

    for file_name in glob(data_dir + '/plots_mel_spectrogram_true/*.png'):
        id = file_name.split('/')[-1].split('_UPID-')[-1].split('.wav')[0]
        if id in re_covid:
            copyfile(file_name, file_name.replace('plots_mel_spectrogram_true', 'covid_plots_mel_spectrogram_true'))
        else:
            copyfile(file_name, file_name.replace('plots_mel_spectrogram_true', 'normal_plots_mel_spectrogram_true'))
