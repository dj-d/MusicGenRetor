from os import path, walk, listdir
from sys import stdout

import pandas as pd
from pip._vendor.distlib.compat import raw_input

from src.audio_features import AudioFeatures

DEFAULT_FOLDER_NAME = 'songs'
DEFAULT_INDEX = ['Title', 'bpm', 'zero_crossing_rate', 'audio_time_series']
DEFAULT_DATASET_NAME = 'dataset'


class DatasetCreator:
    def __init__(self, folder_name=DEFAULT_FOLDER_NAME, index=None, dataset_name=DEFAULT_DATASET_NAME):
        if index is None:
            index = DEFAULT_INDEX

        script_path = path.dirname(path.realpath(__file__))
        self.dir_path = path.abspath(script_path + '/..') + '/' + folder_name

        self.index = index
        self.data_frame = pd.DataFrame(columns=index)

        self.dataset_name = dataset_name
        self.id = 0

    def load_song(self, path):
        audio_features = AudioFeatures(path)

        title = path.basename(path)
        bpm = audio_features.get_bpm()
        zero_crossing_rate = audio_features.get_zero_crossing_rate()
        audio_time_series = audio_features.get_audio_time_series()

        song = pd.Series([title, bpm, zero_crossing_rate, audio_time_series], index=self.index)
        self.data_frame = self.data_frame.append(song, ignore_index=True)

        self.id += 1

    def populate(self, data_path):
        print('Loading data from: ' + data_path)

        for root, directories, files in walk(data_path):
            for directory in directories:
                subdir_path = path.join(data_path, directory)

                walk(subdir_path)

                print('\n---------- ' + directory + ' ----------\n')

                files = listdir(subdir_path)

                for file in files:
                    print(file)

                    file_path = path.join(subdir_path, file)
                    self.load_song(file_path)

    def main(self):
        recreate = True

        if path.exists(self.dataset_name):
            print('Existing dataset')

            recreate = False

            stdout.write('Create new dataset? [y/N]')
            choice = raw_input().lower()

            if choice == 'Y' or choice == 'y':
                recreate = True

            self.data_frame = pd.read_pickle(self.dataset_name)

        if recreate:
            self.populate(self.dir_path)

            self.data_frame.to_pickle(self.dataset_name)

            print('Total records: ' + str(self.id))

            print(self.data_frame)
