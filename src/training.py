import os

import numpy as np
import pandas as pd

from src.audio_features import AudioFeatures

# TODO: change the script name to a more meaningful one

# TODO: create a config file with attrs, dataset_name and genres
attrs = ['Title', 'bpm', 'zero_crossing_rate', 'audio_time_series', 'genre']
dataset_name = 'Dataset'
genres = ['Blues', 'Electronic', 'Classical', 'Pop', 'Rock', 'Jazz']
models_path = 'Models' + '/'
training_datasets_path = 'Datasets' + '/' 'Training' + '/'
genres_dfs = {}
df_training = pd.DataFrame(columns=attrs)
rows = 20
columns = 49
sr = 22100


class Training:

    def __init__(self):
        # Init dictionary
        for genre in genres:
            genres_dfs[genre] = {
                'models': pd.DataFrame(np.zeros((rows, columns))),
                'n_songs': 0
            }

    @staticmethod
    def load_dataset(n):
        global df_training
        # TODO: create bulk load function - ONLY TRAINING
        return pd.read_pickle(training_datasets_path + dataset_name + '_' + str(n))

    @staticmethod
    def load_image_model(model_name):
        model = pd.read_pickle(model_name)
        AudioFeatures().plot_perform_mfcc_by_values(model, sr)

    def plot_models(self):
        for genre in genres:
            model_file = models_path + 'ImageModel_' + genre
            if os.path.exists(model_file):
                self.load_image_model(model_file)
            else:
                print('No models file for ' + genre)

    def generate_models(self):
        # init()
        # Define number of existing datasets
        n_datasets = 1
        while os.path.exists(dataset_name + '_' + str(n_datasets)):
            n_datasets += 1

        for n in range(1, n_datasets):
            dataset = self.load_dataset(n)

            for song in range(len(dataset)):
                print('#----- Training -----#')
                print(dataset.loc[song, ['Title', 'genre']])
                genre = dataset.loc[song, 'genre']
                series = dataset.loc[song, attrs[len(attrs) - 2]]

                # Todo order in dataset creation
                series.sort(axis=0)

                mfcc = AudioFeatures().get_perform_mfcc(series, sr)
                model = genres_dfs[genre]['models']
                for i in range(rows):
                    for j in range(columns):
                        model.iloc[i, j] += mfcc[i, j]
                genres_dfs[genre]['n_songs'] += 1

        for genre in genres_dfs:
            model = genres_dfs[genre]['models']
            n_songs = genres_dfs[genre]['n_songs']
            for i in range(rows):
                for j in range(columns):
                    model.iloc[i, j] = model.iloc[i, j] / n_songs
            if model.notnull().all().any():
                AudioFeatures().plot_perform_mfcc_by_values(model, sr)
                model.to_pickle(models_path + 'ImageModel_' + genre)
