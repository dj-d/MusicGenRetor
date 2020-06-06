import os
import sys

import numpy as np
import pandas as pd
from pip._vendor.distlib.compat import raw_input
from sklearn.metrics import mean_squared_error

from conf import constants as cn
from src.audio_features import AudioFeatures


class Testing:
    def __init__(self):
        self.dataset_base_name = cn.DATASET_BASE_NAME

        self.datasets_directory = cn.DATASET_DIRECTORY
        self.testing_datasets_path = cn.TESTING_DATASET_PATH

        self.datasets_path = self.testing_datasets_path + self.dataset_base_name

        self.models_path = cn.MODELS_PATH

        self.genres = cn.GENRES
        self.attrs = cn.ATTRS

        self.sr = cn.SR

        self.rows = cn.ROWS
        self.columns = cn.COLUMNS

        # TODO: It shouldn't be here
        if os.path.exists(self.testing_datasets_path + self.dataset_base_name + '_1'):
            sys.stdout.write('Perform tests on the models? ' + '[y/N]')
            choice = raw_input().lower()

            if choice == 'Y' or choice == 'y':
                self.testing()

    @staticmethod
    def image_compare(image_a, image_b):
        # TODO: check if need of a filter
        # image_a = image_a.iloc[:, 1:12]
        # image_b = image_b.iloc[:, 1:12]
        # AudioFeatures().plot_perform_mfcc_by_values(image_a, sr)
        # AudioFeatures().plot_perform_mfcc_by_values(image_b, sr)

        return mean_squared_error(image_a, image_b)

    def compare_song_by_path(self, song_path):
        audio_features = AudioFeatures(song_path)
        series = audio_features.get_audio_time_series()

        return self.compare_song(series)
        # TODO: compare song by series not path

    def compare_song(self, series):
        series.sort(axis=0)

        # mfcc = AudioFeatures.get_perform_mfcc(series, sr)
        mfcc = AudioFeatures().get_perform_mfcc(series, self.sr)
        song = pd.DataFrame(np.zeros((self.rows, self.columns)))

        for i in range(self.rows):
            for j in range(self.columns):
                song.iloc[i, j] += mfcc[i, j]

        # song_image = audio_features.plot_perform_mfcc_by_values(models, sr)

        result = {}

        for genre in self.genres:
            model = pd.read_pickle(self.models_path + 'ImageModel_' + genre)
            compare_value = self.image_compare(song, model)
            result[genre] = compare_value

        # return result
        return sorted(result.items(), key=lambda kv: kv[1])

    def testing(self):
        existing_datasets = 0
        total_accuracy = 0
        total_records = 0

        while os.path.exists(self.datasets_path + '_' + str(existing_datasets + 1)):
            existing_datasets += 1

        for n in range(1, existing_datasets + 1):
            df_test = pd.read_pickle(self.datasets_path + '_' + str(n))

            for song in range(len(df_test)):
                print('#----- Testing -----#')
                print(df_test.loc[song, ['Title', 'genre']])

                genre = df_test.loc[song, 'genre']
                series = df_test.loc[song, self.attrs[len(self.attrs) - 2]]

                # TODO: compare song by series not path
                result = self.compare_song(series)

                print('-- Result --')
                print(result)
                print('-- Real genre --')
                print(genre)

                # for res in result:
                if result[0] == genre:
                    # record_accuracy = len(genre) - list(result).index(res)
                    # total_accuracy += record_accuracy
                    total_accuracy += 1
                    # print('Song Accuracy:\t' + str(record_accuracy))

                total_records += 1

        print('Accuracy:\t' + str(total_accuracy) + '\tMax Accuracy:\t' + str(total_records * len(self.genres)))
