import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from conf import constants as cn


class RandomForest:

    def __init__(self):
        self.dataset_base_name = cn.DATASET_BASE_NAME

        self.datasets_directory = cn.DATASET_DIRECTORY
        self.testing_datasets_path = cn.TESTING_DATASET_PATH
        self.training_datasets_path = cn.TRAINING_DATASET_PATH

        self.training_datasets_path = self.training_datasets_path + self.dataset_base_name
        self.testing_datasets_path = self.testing_datasets_path + self.dataset_base_name

        # self.models_path = cn.MODELS_PATH
        #
        # self.genres = cn.GENRES
        self.attrs = cn.ATTRS

        # self.sr = cn.SR
        #
        # self.rows = cn.ROWS
        # self.columns = cn.COLUMNS

        self.training = pd.DataFrame(columns=self.attrs)
        self.testing = pd.DataFrame(columns=self.attrs)

    # def analyzing(self):
    #     plt.figure(figsize=(12, 5))
    #     sns.heatmap(self.datasets.corr(), annot=True)
    #
    #     # Count plot for target which displays total zeros and ones
    #     sns.countplot(x='genre', data=self.datasets)

    def perform_test(self):
        existing_datasets = 0

        while os.path.exists(self.testing_datasets_path + '_' + str(existing_datasets + 1)):
            existing_datasets += 1

        for n in range(1, existing_datasets + 1):
            df_test = pd.read_pickle(self.testing_datasets_path + '_' + str(n))
            df_test = self.to_2d(df_test)

            self.testing = pd.concat([self.testing, df_test])

        existing_datasets = 0

        while os.path.exists(self.training_datasets_path + '_' + str(existing_datasets + 1)):
            existing_datasets += 1

        for n in range(1, existing_datasets + 1):
            df_train = pd.read_pickle(self.training_datasets_path + '_' + str(n))
            df_train = self.to_2d(df_train)

            self.training = pd.concat([self.training, df_train])

            # for song in range(len(df_test)):
            #     print('#----- Testing -----#')
            #     print(df_test.loc[song, ['Title', 'genre']])
            #
            #     genre = df_test.loc[song, 'genre']
            #     # series = df_test.loc[song, self.attrs[len(self.attrs) - 2]]

        # define dataset
        # x_train, y_train = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=3)

        x_train = self.training.loc[:, 'bpm':'chroma_frequencies']
        y_train = self.training['genre']

        x_test = self.testing.loc[:, 'bpm':'chroma_frequencies']
        y_test = self.testing['genre']

        # Perform PCA
        # x_train = self.perform_pca(x_train)
        # y_train = self.perform_pca(y_train)
        # x_test = self.perform_pca(x_test)
        # y_test = self.perform_pca(y_test)

        # define the model
        model = RandomForestClassifier()
        # evaluate the model
        # cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        # n_scores = cross_val_score(model, x_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')

        print(self.training)

        # cv = RepeatedStratifiedKFold(n_splits=4, n_repeats=3, random_state=1)
        # n_scores = cross_val_score(model, x_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')

        rf_model = RandomForestClassifier(n_estimators=200)
        rf_model.fit(x_train, y_train)
        RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                               max_depth=None, max_features='auto', max_leaf_nodes=None,
                               min_impurity_decrease=0.0, min_impurity_split=None,
                               min_samples_leaf=1, min_samples_split=2,
                               min_weight_fraction_leaf=0.0, n_estimators=200, n_jobs=None,
                               oob_score=False, random_state=None, verbose=0,
                               warm_start=False)

        # Prediction using Random Forest Model
        rf_prediction = rf_model.predict(x_test)
        # Evaluations
        print('Classification Report: \n')
        print(classification_report(y_test, rf_prediction))
        print('\nConfusion Matrix: \n')
        print(confusion_matrix(y_test, rf_prediction))

        # report performance
        # print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

    def to_2d(self, df):
        df_2d = df.unstack(1)
        df_2d = df_2d.to_frame()
        # df_2d.columns = df_2d.columns.droplevel(0)
        return df_2d

        # data = x - np.mean(x, axis=0)
        # scatter_matrix = np.dot(data, data.T)
        # eig_val, eig_vec = np.linalg.eig(scatter_matrix)
        # new_reduced_data = np.sqrt(eig_val[0]) * eig_vec.T[0].reshape(-1, 1)
        # pca = decomposition.PCA(n_components=3)
        # pca.fit(new_reduced_data)
        # return pca.transform(new_reduced_data)
