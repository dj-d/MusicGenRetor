import os

import numpy as np
import pandas as pd

from src.audio_features import AudioFeatures

# TODO: change the script name to a more meaningful one

# TODO: create a config file with attrs, dataset_name and genres
attrs = ['Title', 'bpm', 'zero_crossing_rate', 'audio_time_series', 'genre']
dataset_name = 'Dataset'
genres = ['Blues', 'Electronic', 'Classical', 'Pop', 'Rock', 'Jazz']
genres_dfs = {}
df = pd.DataFrame(columns=attrs)
rows = 20
columns = 49
sr = 22100


def load_dataset(n):
    global df
    # TODO: create bulk load function
    return pd.read_pickle(dataset_name + '_' + str(n))


def load_image_model(model_name):
    model = pd.read_pickle(model_name)
    AudioFeatures().plot_perform_mfcc_by_values(model, sr)


def init():
    # Init dictionary
    for genre in genres:
        genres_dfs[genre] = {
            'model': pd.DataFrame(np.zeros((rows, columns))),
            'n_songs': 0
        }


def generate_models():
    init()
    # Define number of existing datasets
    n_datasets = 1
    while os.path.exists(dataset_name + '_' + str(n_datasets)):
        n_datasets += 1

    for n in range(1, n_datasets):
        dataset = load_dataset(n)

        for song in range(len(dataset)):
            print(dataset.loc[song, ['Title', 'genre']])
            genre = dataset.loc[song, 'genre']
            series = dataset.loc[song, attrs[len(attrs) - 2]]

            # Todo order in dataset creation
            series.sort(axis=0)

            mfcc = AudioFeatures().get_perform_mfcc(series, sr)
            model = genres_dfs[genre]['model']
            for i in range(rows):
                for j in range(columns):
                    model.iloc[i, j] += mfcc[i, j]
            genres_dfs[genre]['n_songs'] += 1

    for genre in genres_dfs:
        model = genres_dfs[genre]['model']
        n_songs = genres_dfs[genre]['n_songs']
        for i in range(rows):
            for j in range(columns):
                model.iloc[i, j] = model.iloc[i, j] / n_songs
        if model.notnull().all().any():
            AudioFeatures().plot_perform_mfcc_by_values(model, sr)
            model.to_pickle('ImageModel_' + genre)


# load_image_model('ImageModel_Jazz')
generate_models()
