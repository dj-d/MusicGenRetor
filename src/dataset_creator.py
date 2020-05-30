"""
This script create the dataset needed to ML Algorithms
"""

import os
import sys

import pandas as pd
from pip._vendor.distlib.compat import raw_input

from src.audio_features import AudioFeatures

# TODO: create config file
script_path = os.path.dirname(os.path.realpath(__file__))
dir_path = os.path.abspath(script_path + '/..') + '/songs'
attrs = ['Title', 'bpm', 'zero_crossing_rate', 'audio_time_series', 'genre']
datasets_path = 'Datasets' + '/'
dataset_base_name = datasets_path + 'Dataset'

actual_id = 0
df = pd.DataFrame(columns=attrs)
recreate = True
upgrade = False

# Bulk manager
bulk_save_number = 300
save = 'save'
load = 'load'


def bulk_manager(mode):
    global df, actual_id
    i = 1

    while os.path.exists(dataset_base_name + '_' + str(i)):
        i += 1

    if mode == save:
        if upgrade or recreate:
            df.to_pickle(dataset_base_name + '_' + str(i - 1))
        else:
            df.to_pickle(dataset_base_name + '_' + str(i))
        df = pd.DataFrame(columns=attrs)
    elif mode == load:
        df = pd.read_pickle(dataset_base_name + '_' + str(i - 1))
        actual_id = len(df.index)


def populate(data_path):
    print('Loading data from:\t' + data_path)
    for root, directories, files in os.walk(data_path):
        for directory in directories:
            subdir_path = os.path.join(data_path, directory)
            os.walk(subdir_path)
            print('---------- ' + directory + ' ----------')
            files = os.listdir(subdir_path)
            for file in files:
                file_path = os.path.join(subdir_path, file)
                file_name = os.path.basename(file_path)
                title = file_name.split('.')[0]
                print(title)
                if title not in df[attrs[0]].values:
                    load_song(file_path)
                    if (actual_id % bulk_save_number) == 0:
                        bulk_manager(save)
                else:
                    print('Record exist: skipping ...')


def load_song(path):
    global df, actual_id, attrs
    audio_features = AudioFeatures(path)

    # Following attrs list
    file_name = os.path.basename(path)
    title = file_name.split('.')[0]
    bpm = audio_features.get_bpm()
    zero_crossing_rate = audio_features.get_zero_crossing_rate()
    audio_time_series = audio_features.get_audio_time_series()
    genre = os.path.basename(os.path.dirname(path))

    # Adding song to Dataset
    song = pd.Series([title, bpm, zero_crossing_rate, audio_time_series, genre], index=attrs)
    df = df.append(song, ignore_index=True)

    actual_id = actual_id + 1


# Starting Dataset Creation
if os.path.exists(dataset_base_name + '_1'):
    print('Existing dataset/s')
    recreate = False
    sys.stdout.write('Re-create new datasets? (Actual files will be lost) ' + '[y/N]')
    choice = raw_input().lower()
    if choice == 'Y' or choice == 'y':
        recreate = True
    else:
        upgrade = False
        sys.stdout.write('Upgrade the dataset? ' + '[y/N]')
        choice = raw_input().lower()
        if choice == 'Y' or choice == 'y':
            upgrade = True
            bulk_manager(load)

if recreate or upgrade:
    # Populating Dataset
    populate(dir_path)
    # Save Dataset
    bulk_manager(save)
    print('Total records:\t' + str(actual_id))
