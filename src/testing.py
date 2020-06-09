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

        # TODO: It shouldn't be here
        if os.path.exists(self.testing_datasets_path + self.dataset_base_name + '_1'):
            sys.stdout.write('Perform tests on the models? ' + '[y/N]')
            choice = raw_input().lower()

            if choice == 'Y' or choice == 'y':
                self.testing()

    @staticmethod
    def mse(image_a, image_b):
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
        # series.sort(axis=0) #REMOVED

        # mfcc = AudioFeatures.get_perform_mfcc(series, sr)
        mfcc = AudioFeatures().get_perform_mfcc(series, self.sr)
        song = pd.DataFrame(np.zeros((self.rows, self.columns)))

        for i in range(self.rows):
            for j in range(self.columns):
                song.iloc[i, j] += mfcc[i, j]

        # song_image = audio_features.plot_perform_mfcc_by_values(models, sr)

        result = {}
        score = []
        # mse_res = []
        ssim_res = []

        # SONG_FILE_PATH = 'song.jpeg'
        # im = Image.fromarray(song.to_numpy()).convert('L')
        # im.save(SONG_FILE_PATH)

        for genre in self.genres:
            model = pd.read_pickle(self.models_path + 'ImageModel_' + genre)
            # compare_value = self.mse(song, model)
            mse_value = self.mse(song, model)
            ssim_value = ssim(song.to_numpy(), model.to_numpy())
            # result[genre] = {mse_value, ssim_value}
            # score.append(mse_value)
            # score.append(ssim_value)
            # score.append({mse_value, ssim_value})
            # mse_res.append(mse_value)
            ssim_res.append(ssim_value)

            # TODO insert in training
            # MODEL_FILE_PATH = 'model.jpeg'
            # im = Image.fromarray(model.to_numpy()).convert('L')
            # im.save(MODEL_FILE_PATH)

            # cv = CompareImage(SONG_FILE_PATH, MODEL_FILE_PATH)
            # compare_score = cv.compare_image()
            # print(compare_score)

        # score = minmax_scale(score)
        # mse_res = minmax_scale(mse_res)
        ssim_res = minmax_scale(ssim_res)

        for genre in self.genres:
            i = self.genres.index(genre)
            # result[genre] = score[i * 2] + score[(i * 2) + 1]
            # result[genre] = score[i]
            # result[genre] = mse_res[i] + ssim_res[i]
            result[genre] = ssim_res[i]

        # Plots
        sorted_res = sorted(result.items(), key=lambda kv: kv[1])
        # model = pd.read_pickle(self.models_path + 'ImageModel_' + sorted_res[0][0])
        # self.compare_images(song.to_numpy(), model.to_numpy())

        # return result
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
                    # print(result)

                    # if res[0] == genre:
                    #     record_accuracy = len(genre) - list(result).index(res)
                    #     total_accuracy += record_accuracy
                    #     print('Song Accuracy:\t' + str(record_accuracy))

                total_records += 1

        print('Accuracy:\t' + str(total_accuracy) + '\tMax Accuracy:\t' + str(total_records))

    def compare_images(self, imageA, imageB):
        print('Compare Image')

        m = self.mse(imageA, imageB)
        s = ssim(imageA, imageB)

        fig = plt.figure('Compare')
        plt.suptitle('MSE: %.2f, SSIM: %.2f' % (m, s))

        ax = fig.add_subplot(1, 2, 1)
        plt.imshow(imageA)
        plt.axis('off')

        ax = fig.add_subplot(1, 2, 2)
        plt.imshow(imageB)
        plt.axis('off')

        plt.show()
