"""
This script create the dataset needed to ML Algorithms
"""

import os
import sys

import pandas as pd
from pip._vendor.distlib.compat import raw_input

from src.audio_features import AudioFeatures

script_path = os.path.dirname(os.path.realpath(__file__))
dir_path = os.path.abspath(script_path + '/..') + '/songs'

actual_id = 0
attrs = ['Title', 'bpm', 'zero_crossing_rate', 'audio_time_series', 'genre']
df = pd.DataFrame(columns=attrs)

recreate = True
dataset_name = 'Dataset'


def populate(data_path):
    global actual_id
    print('Loading data from:\t' + data_path)
    for root, directories, files in os.walk(data_path):
        for directory in directories:
            subdir_path = os.path.join(data_path, directory)
            os.walk(subdir_path)
            print('---------- ' + directory + ' ----------')
            files = os.listdir(subdir_path)
            for file in files:
                print(file)
                file_path = os.path.join(subdir_path, file)
                load_song(file_path)


def load_song(path):
    global df, actual_id, attrs
    audio_features = AudioFeatures(path)

    # Following attrs list
    title = os.path.basename(path)
    bpm = audio_features.get_bpm()
    zero_crossing_rate = audio_features.get_zero_crossing_rate()
    audio_time_series = audio_features.get_audio_time_series()
    genre = os.path.basename(os.path.dirname(path))

    # Adding song to Dataset
    song = pd.Series([title, bpm, zero_crossing_rate, audio_time_series, genre], index=attrs)
    df = df.append(song, ignore_index=True)

    actual_id = actual_id + 1


# Starting Dataset Creation
if os.path.exists(dataset_name):
    print('Existing dataset')
    recreate = False
    sys.stdout.write('Create new dataset? ' + '[y/N]')
    choice = raw_input().lower()
    if choice == 'Y' or choice == 'y':
        recreate = True

    df = pd.read_pickle(dataset_name)

if recreate:
    # Populating Dataset
    populate(dir_path)
    # Save Dataset
    df.to_pickle(dataset_name)
    print('Total records:\t' + str(actual_id))

print(df)
