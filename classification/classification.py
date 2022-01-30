import math

import numpy as np
import pandas
import pandas as pd
from IPython.core.display import display
from sklearn.preprocessing import StandardScaler
from spotipy_section.graphPlaylist import get_all_music_features, ALL_FEATURE_LABELS, view_scatter_graph
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge


class Classifier:
    def __init__(self, data, emotion):
        self.user_data = data
        self.EMOTION = emotion

    TRAIN_DATA_PROPORTION = 0.7
    LOG_THRESHOLD = 0.3
    DATA_PRESERVED = 0.95

    train_lbl = pandas.Series
    test_lbl = pandas.Series
    split_pos = 0
    train_data = pandas.Series
    test_data = pandas.Series

    # General PCA
    # def standardizer(df):
    #     x = df.loc[:, ALL_FEATURE_LABELS].values
    #     x = StandardScaler().fit_transform(x)
    #     return x
    #
    #
    # def princomp(data, target):
    #     pca = PCA(n_components=2)
    #     principal_components = pca.fit_transform(data)
    #     principal_df = pd.DataFrame(data=principal_components, columns=['pc1', 'pc2'])
    #     final_df = pd.concat([principal_df, target], axis=1)
    #     return final_df

    # --- formatting training data and test data for modeling ---
    def prep_data(self):
        # global train_data, test_data, test_lbl, train_lbl
        # Gets the list of tracks from the user
        track_list = self.user_data['track_id'].tolist()
        # Gets the all the musical features of each song
        track_features = get_all_music_features(track_list)
        self.user_data = pd.concat([self.user_data, track_features], axis=1)

        # Randomizes rows
        shuffled_data = self.user_data.sample(frac=1)

        # Accesses and sets the data and labels from main df
        self.set_train_test_data(shuffled_data.iloc[:, 9:])
        self.set_train_test_labels(shuffled_data.iloc[:, 2:9])
        # standardizes the data
        self.train_data, self.test_data = self.standardizer()

    def set_train_test_data(self, music_data):
        # global split_pos, train_data, test_data
        self.split_pos = math.ceil((len(music_data)) * self.TRAIN_DATA_PROPORTION)
        self.train_data = music_data.iloc[:self.split_pos, :]
        self.test_data = music_data.iloc[self.split_pos:, :]

    def set_train_test_labels(self, labels):
        # global train_lbl, test_lbl, split_pos
        labels = labels[self.EMOTION]
        self.train_lbl = labels[:self.split_pos]
        self.test_lbl = labels[self.split_pos:]

    def get_log_data_labels(self, label):
        log_label = [int(x >= self.LOG_THRESHOLD) for x in label.tolist()]
        return log_label

    # --- PCA (for use on ML techniques) ---
    def standardizer(self):
        # get data
        # global train_data, test_data

        # define scaler
        scaler = StandardScaler()

        # fit scalar to training data only
        scaler.fit(self.train_data)

        # Transform both datasets
        self.train_data = scaler.transform(self.train_data)
        self.test_data = scaler.transform(self.test_data)
        return self.train_data, self.test_data

    # PCA function
    def prin_comp(self):
        # global train_data, test_data
        # Keeps the relevant number of components to ensure 95% of original data is preserved
        pca = PCA(self.DATA_PRESERVED)
        pca.fit(self.train_data)

        # Transform both data
        self.train_data = pca.transform(self.train_data)
        self.test_data = pca.transform(self.test_data)

    # --- Different models ---
    # Logistic regression modeling
    def log_reg_func(self):
        # global train_data, train_lbl
        log_reg_train_lbl = self.get_log_data_labels(self.train_lbl)
        logistic_reg = LogisticRegression(solver='lbfgs', fit_intercept=False)
        logistic_reg.fit(self.train_data, log_reg_train_lbl)
        return logistic_reg

    # Linear regression modeling
    def lin_reg_func(self):
        # global train_data
        linear_reg = LinearRegression(positive=True, fit_intercept=False)
        linear_reg.fit(self.train_data, self.train_lbl)
        return linear_reg

    # Ridge
    def ridge_reg_func(self):
        # global train_data
        ridge_reg = Ridge(positive=True, fit_intercept=False)
        ridge_reg.fit(self.train_data, self.train_lbl)
        return ridge_reg

    # def classify(user_df):
    #     # Gets the list of tracks from the user
    #     track_list = user_df['track_id'].tolist()
    #
    #     # Gets the all the musical features of each song
    #     track_features = get_all_music_features(track_list)
    #
    #     # standardizes the data
    #     std_data = standardizer(track_features)
    #     target = user_df[['joy', 'sadness']]
    #
    #     # does PCA on the standardized data
    #     pc_data = princomp(std_data, target)
    #
    #     # Graph data
    #     view_scatter_graph(pc_data)

    # --- Testing different classification methods ---
    def log_reg_classifier(self):
        # applying logistic regression
        log_reg_model = self.log_reg_func()
        # predict all values
        print("Logistic Regression:")
        print("Actual:")
        print(self.get_log_data_labels(self.test_lbl))
        print("Predicted:")
        print(log_reg_model.predict(self.test_data))
        print("")

    def lin_reg_classifier(self):
        # global test_data
        lin_reg_model = self.lin_reg_func()
        # predicting all values
        print("Linear Regression:")
        print("Actual:")
        print(np.around(self.test_lbl.tolist(), 3))
        print("Predicted:")
        print(np.around(lin_reg_model.predict(self.test_data), 3))
        print("")

    def ridge_reg_classifier(self):
        # global test_data
        ridge_reg_model = self.ridge_reg_func()
        # predict all values
        print("Ridge Regression:")
        print("Actual:")
        print(np.around(self.test_lbl.tolist(), 3))
        print("Predicted:")
        print(np.around(ridge_reg_model.predict(self.test_data), 3))
        print("")

    # --- Runs classification, choosing appropriate method
    def classify(self):
        # Gets formatted data for training and testing
        self.prep_data()
        # Does PCA on data
        self.prin_comp()

        self.log_reg_classifier()
        self.lin_reg_classifier()
        self.ridge_reg_classifier()
