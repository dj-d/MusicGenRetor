"""
This script create the dataset needed to ML Algorithms
"""

import os
import random
import sys

import pandas as pd
from pip._vendor.distlib.compat import raw_input

from src.audio_features import AudioFeatures

# TODO: create config file
script_path = os.path.dirname(os.path.realpath(__file__))
dir_path = os.path.abspath(script_path + '/..') + '/songs'
attrs = ['Title', 'bpm', 'zero_crossing_rate', 'audio_time_series', 'genre']
datasets_directory = 'Datasets' + '/'
training_directory = 'Training' + '/'
training_datasets_path = datasets_directory + training_directory
test_directory = 'Testing' + '/'
test_datasets_path = datasets_directory + test_directory
dataset_base_name = 'Dataset'

training_records = 0
test_records = 0
df_training = pd.DataFrame(columns=attrs)
df_test = pd.DataFrame(columns=attrs)
recreate = False
upgrade = False
none_exist = False
training_percentual = 0.8
training_type = 'Training'
test_type = 'Testing'

# Bulk manager
bulk_save_number = 150
save = 'save'
load = 'load'


class DatasetsCreator:

    def __init__(self):
        # Starting Dataset Creation
        global recreate, upgrade, none_exist
        if os.path.exists(training_datasets_path + dataset_base_name + '_1'):
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
                    self.bulk_manager(load, training_type)
                    self.bulk_manager(load, test_type)
        else:
            none_exist = True
        if recreate or upgrade or none_exist:
            # Populating Dataset
            self.populate(dir_path)
            # Save Dataset
            self.bulk_manager(save, training_type)
            self.bulk_manager(save, test_type)
            print('Train records:\t' + str(training_records))
            print('Test records:\t' + str(test_records))

    @staticmethod
    def bulk_manager(mode, dataset_type):
        global df_training, df_test, training_records, test_records
        dataset_path = datasets_directory + dataset_type + '/' + dataset_base_name

        existing_datasets = 1
        while os.path.exists(dataset_path + '_' + str(existing_datasets)):
            existing_datasets += 1

        if mode == save:
            if upgrade or recreate:
                existing_datasets -= 1
            if dataset_type == training_type:
                df_training.to_pickle(dataset_path + '_' + str(existing_datasets))
                df_training = pd.DataFrame(columns=attrs)
            elif dataset_type == test_type:
                df_test.to_pickle(dataset_path + '_' + str(existing_datasets))
                df_test = pd.DataFrame(columns=attrs)
        elif mode == load:
            existing_datasets -= 1
            if dataset_type == training_type:
                df_training = pd.read_pickle(dataset_path + '_' + str(existing_datasets))
                training_records = len(df_training.index)
            elif dataset_type == test_type:
                df_test = pd.read_pickle(dataset_path + '_' + str(existing_datasets))
                test_records = len(df_test.index)

    @staticmethod
    def load_song(path, df):
        global attrs
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
        return df

    @staticmethod
    def split_dataset(elements_number, split_percentual):
        numbers = range(0, elements_number)
        return random.sample(numbers, int(elements_number * split_percentual))

    def populate(self, data_path):
        global df_training, df_test, training_records, test_records
        print('Loading data from:\t' + data_path)
        for root, directories, files in os.walk(data_path):
            for directory in directories:
                subdir_path = os.path.join(data_path, directory)
                os.walk(subdir_path)
                print('---------- ' + directory + ' ----------')
                files = os.listdir(subdir_path)
                training_drawns = self.split_dataset(len(files), training_percentual)
                for file in files:
                    file_path = os.path.join(subdir_path, file)
                    file_name = os.path.basename(file_path)
                    title = file_name.split('.')[0]
                    print('Analyzing song: \t' + title)
                    if title not in df_training[attrs[0]].values and title not in df_training[attrs[0]].values:
                        total_records = training_records + test_records
                        if total_records in training_drawns:
                            df_training = self.load_song(file_path, df_training)
                            training_records += 1
                            if (training_records % bulk_save_number) == 0:
                                self.bulk_manager(save, training_type)
                        else:
                            df_test = self.load_song(file_path, df_test)
                            test_records += 1
                            if (test_records % bulk_save_number) == 0:
                                self.bulk_manager(save, test_type)
                    else:
                        print('Record exist: skipping ...')
