import math

import numpy as np
import pandas
import pandas as pd
from IPython.core.display import display
from sklearn.preprocessing import StandardScaler
from spotipy_section.graphPlaylist import get_all_music_features, ALL_FEATURE_LABELS, view_scatter_graph
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge

TRAIN_DATA_PROPORTION = 0.7

train_lbl = pandas.Series
test_lbl = pandas.Series

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
def prep_data(user_df):
    # Gets the list of tracks from the user
    track_list = user_df['track_id'].tolist()
    # Gets the all the musical features of each song
    track_features = get_all_music_features(track_list)

    # standardizes the data
    train_data, test_data = standardizer(track_features)
    # does PCA on the standardized data
    return prin_comp(train_data, test_data)


# --- PCA (for use on ML techniques) ---
def standardizer(df):
    # get data
    train_data = df.iloc[:10, :]
    test_data = df.iloc[10:, :]

    # define scaler
    scaler = StandardScaler()

    # fit scalar to training data only
    scaler.fit(train_data)

    # Transform both datasets
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)
    return train_data, test_data


# PCA function
def prin_comp(train_data, test_data):
    # Keeps the relevant number of components to ensure 95% of original data is preserved
    pca = PCA(0.95)
    pca.fit(train_data)
    # Transform both data
    train_data = pca.transform(train_data)
    test_data = pca.transform(test_data)
    return train_data, test_data


# --- Different models ---
# Logistic regression modeling
def log_reg_func(train_data):
    log_reg_train_lbl = [0, 1, 1, 1, 0, 0, 0, 1, 0, 0]
    logistic_reg = LogisticRegression(solver='lbfgs', fit_intercept=False)
    logistic_reg.fit(train_data, log_reg_train_lbl)
    return logistic_reg


# Linear regression modeling
def lin_reg_func(train_data):
    linear_reg = LinearRegression(positive=True, fit_intercept=False)
    linear_reg.fit(train_data, train_lbl)
    return linear_reg


# Ridge
def ridge_reg_func(train_data):
    ridge_reg = Ridge(positive=True, fit_intercept=False)
    ridge_reg.fit(train_data, train_lbl)
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

def set_train_test_labels(labels):
    global train_lbl, test_lbl
    split_pos = math.ceil((len(labels)) * TRAIN_DATA_PROPORTION)
    train_lbl = labels[:split_pos]
    test_lbl = labels[split_pos:]


# --- Testing different classification methods ---
def log_reg_classifier(train_data, test_data):
    # applying logistic regression
    log_reg_model = log_reg_func(train_data)
    # predict all values
    print("Logistic Regression:")
    print("Actual:")
    print([0, 1, 0, 0])
    print("Predicted:")
    print(log_reg_model.predict(test_data))
    print("")


def lin_reg_classifier(train_data, test_data):
    lin_reg_model = lin_reg_func(train_data)
    # predicting all values
    print("Linear Regression:")
    print("Actual:")
    print(test_lbl.to_numpy())
    print("Predicted:")
    print(lin_reg_model.predict(test_data))
    print("")


def ridge_reg_classifier(train_data, test_data):
    ridge_reg_model = ridge_reg_func(train_data)
    # predict all values
    print("Ridge Regression:")
    print("Actual:")
    print(test_lbl.to_numpy())
    print("Predicted:")
    print(ridge_reg_model.predict(test_data))
    print("")


# --- Runs classification, choosing appropriate method
def classifier(user_df):
    # Gets formatted data for training and testing
    train_data, test_data = prep_data(user_df)
    # Set labels
    set_train_test_labels(user_df['joy'])

    log_reg_classifier(train_data, test_data)
    lin_reg_classifier(train_data, test_data)
    ridge_reg_classifier(train_data, test_data)
