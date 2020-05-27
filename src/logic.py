import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import array
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.feature_extraction import image

# TODO: change the script name to a more meaningful one

# TODO: create a config file with attrs, dataset_name and genres
attrs = ['Title', 'bpm', 'zero_crossing_rate', 'audio_time_series', 'genre']
dataset_name = 'Dataset'
genres = ['Blues' 'Electronic' 'Classical' 'Pop' 'Rock' 'Jazz']

df = pd.DataFrame(columns=attrs)


def load_dataset():
    global df
    # TODO: create bulk load function
    return pd.read_pickle(dataset_name + '_1')


def clustering():
    df = load_dataset()

    ids = df[attrs[0]]
    genres_true = df[attrs[len(attrs) - 1]]
    dataset = df.loc[0:, attrs[1]:attrs[len(attrs) - 2]]
    # dataset.loc[1:, attrs[1]:attrs[len(attrs) - 3]] = preprocessing.normalize(dataset.loc[1:, attrs[1]:attrs[len(attrs) - 3]])
    # dataset = preprocessing.normalize(dataset)

    kmeans = KMeans(n_clusters=len(genres), random_state=0).fit(dataset)
    # kmeans.labels_
    print(kmeans.labels_)
    kmeans.predict()
    # kmeans.cluster_centers_
    print(kmeans.cluster_centers_)


def spectral_analisys(img):
    mask = img.astype(bool)
    img = img.astype(float)
    # Convert the image into a graph with the value of the gradient on the
    # edges.
    graph = image.img_to_graph(img, mask=mask)

    # Take a decreasing function of the gradient: we take it weakly
    # dependent from the gradient the segmentation is close to a voronoi
    graph.data = np.exp(-graph.data / graph.data.std())

    # Force the solver to be arpack, since amg is numerically
    # unstable on this example
    # labels = spectral_clustering(graph, n_clusters=4, eigen_solver='arpack')
    # label_im = np.full(mask.shape, -1.)
    # label_im[mask] = labels

    plt.matshow(img)
    # plt.matshow(label_im)
    plt.show()


def array_to_img(arr):
    # x = 1000
    x = int(math.sqrt(len(arr)))
    to_fill = x - (len(arr) % x)

    for i in range(to_fill):
        arr = np.append(arr, [0])
    # print(len(arr))
    n_array = array(arr)
    matrix = n_array.reshape((x, -1))
    # matrix = preprocessing.normalize(matrix)
    matrix = preprocessing.minmax_scale(matrix)
    plt.imshow(matrix, cmap='Spectral')
    plt.colorbar()
    plt.show()


dataset = load_dataset()
song = 10
print(dataset.loc[song, ['Title', 'genre']])
array_to_img(dataset.loc[song, attrs[len(attrs) - 2]])
# spectral_analisys()
# clustering()
