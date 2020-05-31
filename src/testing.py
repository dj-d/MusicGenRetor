import os
import sys

import numpy as np
import pandas as pd
from pip._vendor.distlib.compat import raw_input
from sklearn.metrics import mean_squared_error

from src.audio_features import AudioFeatures

datasets_directory = 'Datasets' + '/'
test_directory = 'Testing' + '/'
test_datasets_path = datasets_directory + test_directory
dataset_base_name = 'Dataset'
dataset_path = test_datasets_path + dataset_base_name
genres = ['Blues', 'Electronic', 'Classical', 'Pop', 'Rock', 'Jazz']
attrs = ['Title', 'bpm', 'zero_crossing_rate', 'audio_time_series', 'genre']
models_path = 'Models' + '/'
rows = 20
columns = 49
sr = 22100


class Testing:
    def __init__(self):
        if os.path.exists(test_datasets_path + dataset_base_name + '_1'):
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
        #     self.compare_song(series)
        # TODO: compare song by series not path
        # def compare_song(self, series):
        series.sort(axis=0)

        # mfcc = AudioFeatures.get_perform_mfcc(series, sr)
        mfcc = audio_features.get_perform_mfcc(series, sr)
        song = pd.DataFrame(np.zeros((rows, columns)))
        for i in range(rows):
            for j in range(columns):
                song.iloc[i, j] += mfcc[i, j]
        # song_image = audio_features.plot_perform_mfcc_by_values(models, sr)

        result = {}
        for genre in genres:
            model = pd.read_pickle(models_path + 'ImageModel_' + genre)
            compare_value = self.image_compare(song, model)
            result[genre] = compare_value

        return sorted(result.items(), key=lambda kv: kv[1])

    def testing(self):
        existing_datasets = 0
        while os.path.exists(dataset_path + '_' + str(existing_datasets)):
            existing_datasets += 1

        for n in range(1, existing_datasets):
            df_test = pd.read_pickle(dataset_path + '_' + str(n))

            for song in range(len(df_test)):
                print('#----- Testing -----#')
                print(df_test.loc[song, ['Title', 'genre']])
                genre = df_test.loc[song, 'genre']
                series = df_test.loc[song, attrs[len(attrs) - 2]]

                # TODO: compare song by series not path
                result = self.compare_song(series)
                print('-- Result --')
                print(result)
                print('-- Real genre --')
                print(genre)
