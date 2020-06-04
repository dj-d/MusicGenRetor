import os
import random
import sys

from conf import constants as cn

import pandas as pd
from pip._vendor.distlib.compat import raw_input

from src.audio_features import AudioFeatures


class DatasetsCreator:
    def __init__(self):
        script_path = os.path.dirname(os.path.realpath(__file__))  # TODO: To change
        self.dir_path = os.path.abspath(script_path + '/..') + '/songs'  # TODO: To change

        self.dataset_base_name = cn.DATASET_BASE_NAME

        self.datasets_directory = cn.DATASET_DIRECTORY
        self.training_datasets_path = cn.TRAINING_DATASET_PATH
        self.testing_datasets_path = cn.TESTING_DATASET_PATH

        self.attrs = cn.ATTRS

        self.training_percentage = 0.8

        self.training_type = 'Training'
        self.testing_type = 'Testing'

        self.training_records = 0
        self.testing_records = 0

        self.df_training = pd.DataFrame(columns=self.attrs)
        self.df_testing = pd.DataFrame(columns=self.attrs)

        self.recreate = False
        self.upgrade = False
        self.none_exist = False

        self.bulk_save_number = 150
        self.save = "save"
        self.load = "load"

        # TODO: It shouldn't be here
        # Starting Dataset Creation
        if os.path.exists(self.training_datasets_path + self.dataset_base_name + '_1'):
            print('Existing dataset/s')

            # TODO: It could be removed from here
            self.recreate = False

            sys.stdout.write('Re-create new datasets? (Actual files will be lost) ' + '[y/N]')
            choice = raw_input().lower()

            if choice == 'Y' or choice == 'y':
                self.recreate = True
            else:
                # TODO: It could be removed from here
                self.upgrade = False

                sys.stdout.write('Upgrade the dataset? ' + '[y/N]')
                choice = raw_input().lower()

                if choice == 'Y' or choice == 'y':
                    self.upgrade = True

                    self.bulk_manager(self.load, self.training_type)
                    self.bulk_manager(self.load, self.testing_type)
        else:
            self.none_exist = True

        if self.recreate or self.upgrade or self.none_exist:
            # Populating Dataset
            self.populate(self.dir_path)

            # Save Dataset
            self.bulk_manager(self.save, self.training_type)
            self.bulk_manager(self.save, self.testing_type)

            print('Train records:\t' + str(self.training_records))
            print('Test records:\t' + str(self.testing_records))

    def bulk_manager(self, mode, dataset_type):
        dataset_path = self.datasets_directory + dataset_type + '/' + self.dataset_base_name

        existing_datasets = 1

        while os.path.exists(dataset_path + '_' + str(existing_datasets)):
            existing_datasets += 1

        if mode == self.save:
            if self.upgrade or self.recreate:
                existing_datasets -= 1
                self.upgrade = False
                self.recreate = False

            if dataset_type == self.training_type:
                self.df_training.to_pickle(dataset_path + '_' + str(existing_datasets))
                self.df_training = pd.DataFrame(columns=self.attrs)
            elif dataset_type == self.testing_type:
                self.df_testing.to_pickle(dataset_path + '_' + str(existing_datasets))
                self.df_testing = pd.DataFrame(columns=self.attrs)
        elif mode == self.load:
            existing_datasets -= 1

            if dataset_type == self.training_type:
                self.df_training = pd.read_pickle(dataset_path + '_' + str(existing_datasets))
                self.training_records = len(self.df_training.index)
            elif dataset_type == self.testing_type:
                self.df_testing = pd.read_pickle(dataset_path + '_' + str(existing_datasets))
                self.testing_records = len(self.df_testing.index)

    def load_song(self, path, df):
        audio_features = AudioFeatures(path)

        # Following attrs list
        file_name = os.path.basename(path)
        title = file_name.split('.')[0]
        bpm = audio_features.get_bpm()
        zero_crossing_rate = audio_features.get_zero_crossing_rate()
        audio_time_series = audio_features.get_audio_time_series()
        genre = os.path.basename(os.path.dirname(path))

        # Adding song to Dataset
        song = pd.Series([title, bpm, zero_crossing_rate, audio_time_series, genre], index=self.attrs)
        df = df.append(song, ignore_index=True)

        return df

    @staticmethod
    def split_dataset(elements_number, split_percentual):
        numbers = range(0, elements_number)

        return random.sample(numbers, int(elements_number * split_percentual))

    def populate(self, data_path):
        print('Loading data from:\t' + data_path)

        for root, directories, files in os.walk(data_path):
            for directory in directories:
                subdir_path = os.path.join(data_path, directory)
                os.walk(subdir_path)

                print('---------- ' + directory + ' ----------')

                files = os.listdir(subdir_path)
                training_drawns = self.split_dataset(len(files), self.training_percentage)
                genre_records = 0

                for file in files:
                    file_path = os.path.join(subdir_path, file)
                    file_name = os.path.basename(file_path)
                    title = file_name.split('.')[0]

                    print('Analyzing song: \t' + title)

                    if title not in self.df_training[self.attrs[0]].values and title not in self.df_training[self.attrs[0]].values:
                        if genre_records in training_drawns:
                            self.df_training = self.load_song(file_path, self.df_training)
                            self.training_records += 1

                            if (self.training_records % self.bulk_save_number) == 0:
                                self.bulk_manager(self.save, self.training_type)
                        else:
                            self.df_testing = self.load_song(file_path, self.df_testing)

                            self.testing_records += 1

                            if (self.testing_records % self.bulk_save_number) == 0:
                                self.bulk_manager(self.save, self.testing_type)

                        genre_records += 1
                    else:
                        print('Record exist: skipping ...')
