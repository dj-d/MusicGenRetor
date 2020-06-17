import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pip._vendor.distlib.compat import raw_input
from skimage.measure import compare_ssim as ssim
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import minmax_scale

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

        self.ignored = 0

        # TODO: It shouldn't be here
        if os.path.exists(self.testing_datasets_path + self.dataset_base_name + '_1'):
            sys.stdout.write('Perform tests on the models? ' + '[y/N]')
            choice = raw_input().lower()

            if choice == 'Y' or choice == 'y':
                self.testing()

    @staticmethod
    def mse(image_a, image_b):
        # TODO: check if need of a filter
        return mean_squared_error(image_a, image_b)

    def compare_song_by_path(self, song_path):
        audio_features = AudioFeatures(song_path)
        series = audio_features.get_audio_time_series()

        return self.compare_song(series)
        # TODO: compare song by series not path

    def compare_song(self, series):
        mfcc = AudioFeatures().get_perform_mfcc(series, self.sr)
        print(str(len(mfcc[0])))
        if len(mfcc[0]) < self.columns:
            print('Skipping: too short')
            self.ignored += 1
        else:
            song = pd.DataFrame(np.zeros((self.rows, self.columns)))
            for i in range(self.rows):
                for j in range(self.columns):
                    song.iloc[i, j] += mfcc[i, j]

            result = {}
            ssim_res = []

            for genre in self.genres:
                model = pd.read_pickle(self.models_path + 'ImageModel_' + genre)
                # Fix dimensions
                model = model.iloc[:self.rows, :self.columns]
                ssim_value = ssim(song.to_numpy(), model.to_numpy())
                ssim_res.append(ssim_value)

            ssim_res = minmax_scale(ssim_res)

            for genre in self.genres:
                i = self.genres.index(genre)
                result[genre] = ssim_res[i]

            # Plots
            sorted_res = sorted(result.items(), key=lambda kv: kv[1])

            return sorted_res

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

                # Minmax
                series = minmax_scale(series)

                # TODO: compare song by series not path
                result = self.compare_song(series)

                if result is not None:
                    print('-- Result --')
                    print(result)
                    print('-- Real genre --')
                    print(genre)

                    # if result[0][0] == genre:
                    if result[len(result) - 1][0] == genre:
                        print("---------- Good ----------")
                        total_accuracy += 1
                    else:
                        print("---------- Bad ----------")
                        # Check second position
                        if result[len(result) - 2][0] == genre:
                            print("---------- Good ----------")
                            total_accuracy += 1

                    total_records += 1

        print('Accuracy:\t' + str(total_accuracy) + '\tMax Accuracy:\t' + str(total_records))
        print('Ignored:\t' + str(self.ignored))

    def compare_images(self, image_a, image_b):
        print('Compare Image')

        m = self.mse(image_a, image_b)
        s = ssim(image_a, image_b)

        fig = plt.figure('Compare')
        plt.suptitle('MSE: %.2f, SSIM: %.2f' % (m, s))

        ax = fig.add_subplot(1, 2, 1)
        plt.imshow(image_a)
        plt.axis('off')

        ax = fig.add_subplot(1, 2, 2)
        plt.imshow(image_b)
        plt.axis('off')

        plt.show()
