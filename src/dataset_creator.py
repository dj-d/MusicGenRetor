"""
This script create the dataset needed to ML Algorithms
"""

import os

import librosa

script_path = os.path.dirname(os.path.realpath(__file__))
dir_path = os.path.abspath(script_path + '/..') + '/songs'


def populate(data_path):
    print('Loading data from:\t' + data_path)
    for root, directories, files in os.walk(data_path):
        for directory in directories:
            subdir_path = os.path.join(data_path, directory)
            os.walk(subdir_path)
            print('---------- ' + directory + ' ----------')
            files = os.listdir(subdir_path)
            for file in files:
                # print(os.path.join(str(subdir_path), file))
                print(file)


def load_song(path):
    librosa.load(path)


populate(dir_path)
