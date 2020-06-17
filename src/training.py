import os
import sys

import numpy as np
import pandas as pd
from pip._vendor.distlib.compat import raw_input
from sklearn.preprocessing import minmax_scale

from conf import constants as cn
from src.audio_features import AudioFeatures


# TODO: change the script name to a more meaningful one


class Training:
    def __init__(self):
        self.dataset_base_name = cn.DATASET_BASE_NAME

        self.training_datasets_path = cn.TRAINING_DATASET_PATH

        self.models_path = cn.MODELS_PATH

        self.genres = cn.GENRES
        self.attrs = cn.ATTRS

        self.sr = cn.SR

        self.rows = cn.ROWS
        self.columns = cn.COLUMNS

        self.df_training = pd.DataFrame(columns=self.attrs)

        self.genres_dfs = {}
        self.init_genres_dfs()

        if os.path.exists(self.models_path + "ImageModel_Pop"):
            sys.stdout.write('Create new models? ' + '[y/N]')
            choice = raw_input().lower()

            if choice == 'Y' or choice == 'y':
                self.generate_models()

    def init_genres_dfs(self):
        for genre in self.genres:
            self.genres_dfs[genre] = {
                'models': pd.DataFrame(np.zeros((self.rows, self.columns))),
                'n_songs': 0
            }

    def load_dataset(self, n):
        # TODO: create bulk load function - ONLY TRAINING
        return pd.read_pickle(self.training_datasets_path + self.dataset_base_name + '_' + str(n))

    def load_image_model(self, model_name):
        model = pd.read_pickle(model_name)
        AudioFeatures().plot_perform_mfcc_by_values(model, self.sr)

    def plot_models(self):
        for genre in self.genres:
            model_file = self.models_path + 'ImageModel_' + genre

            if os.path.exists(model_file):
                self.load_image_model(model_file)
            else:
                print('No models file for ' + genre)

    def generate_models(self):
        ignored = 0
        # Define number of existing datasets
        n_datasets = 1

        while os.path.exists(self.training_datasets_path + self.dataset_base_name + '_' + str(n_datasets)):
            n_datasets += 1

        n_datasets -= 1

        for n in range(1, n_datasets + 1):
            dataset = self.load_dataset(n)

            for song in range(len(dataset)):
                print('#----- Training -----#')
                print(dataset.loc[song, ['Title', 'genre']])

                genre = dataset.loc[song, 'genre']
                series = dataset.loc[song, self.attrs[len(self.attrs) - 2]]
                mfcc = AudioFeatures().get_perform_mfcc(series, self.sr)

                # Normalize
                mfcc = minmax_scale(mfcc)
                model = self.genres_dfs[genre]['models']

                if len(mfcc[0]) >= self.columns:
                    for i in range(self.rows):
                        for j in range(self.columns):
                            model.iloc[i, j] += mfcc[i, j]
                    self.genres_dfs[genre]['n_songs'] += 1
                else:
                    print('Skipping: too short')
                    ignored += 1

        for genre in self.genres_dfs:
            model = self.genres_dfs[genre]['models']
            n_songs = self.genres_dfs[genre]['n_songs']
            model = model / n_songs

            if model.notnull().all().any():
                AudioFeatures().plot_perform_mfcc_by_values(model, self.sr)
                model.to_pickle(self.models_path + 'ImageModel_' + genre)
        print('Ignored:\t' + str(ignored))
