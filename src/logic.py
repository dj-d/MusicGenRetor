import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from src.audio_features import AudioFeatures

# TODO: change the script name to a more meaningful one

# TODO: create a config file with attrs, dataset_name and genres
attrs = ['Title', 'bpm', 'zero_crossing_rate', 'audio_time_series', 'genre']
dataset_name = 'Dataset'
genres = ['Blues', 'Electronic', 'Classical', 'Pop', 'Rock', 'Jazz']
models_path = 'Models' + '/'
datasets_path = 'Datasets' + '/'
genres_dfs = {}
df = pd.DataFrame(columns=attrs)
rows = 20
columns = 49
sr = 22100


def load_dataset(n):
    global df
    # TODO: create bulk load function
    return pd.read_pickle(datasets_path + dataset_name + '_' + str(n))


def load_image_model(model_name):
    model = pd.read_pickle(model_name)
    AudioFeatures().plot_perform_mfcc_by_values(model, sr)


def init():
    # Init dictionary
    for genre in genres:
        genres_dfs[genre] = {
            'models': pd.DataFrame(np.zeros((rows, columns))),
            'n_songs': 0
        }


def plot_models():
    for genre in genres:
        model_file = models_path + 'ImageModel_' + genre
        if os.path.exists(model_file):
            load_image_model(model_file)
        else:
            print('No models file for ' + genre)


def image_compare(image_a, image_b):
    return mean_squared_error(image_a, image_b)


def compare_song(song_path):
    audio_features = AudioFeatures(song_path)
    series = audio_features.get_audio_time_series()

    # Todo order in dataset creation
    series.sort(axis=0)

    mfcc = audio_features.get_perform_mfcc(series, sr)
    song = pd.DataFrame(np.zeros((rows, columns)))
    for i in range(rows):
        for j in range(columns):
            song.iloc[i, j] += mfcc[i, j]
    # song_image = audio_features.plot_perform_mfcc_by_values(models, sr)

    result = {}
    for genre in genres:
        model = pd.read_pickle(models_path + 'ImageModel_' + genre)
        compare_value = image_compare(song, model)
        result[genre] = compare_value

    return result


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


generate_models()
plot_models()

# print(compare_song())
