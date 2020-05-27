import pandas as pd
from sklearn.cluster import KMeans

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
    dataset = load_dataset()
    kmeans = KMeans(n_clusters=len(genres), random_state=0).fit(dataset)
    # kmeans.labels_
    print(kmeans.labels_)
    kmeans.predict()
    # kmeans.cluster_centers_
    print(kmeans.cluster_centers_)


clustering()
