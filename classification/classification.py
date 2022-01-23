import pandas as pd
from IPython.core.display import display
from sklearn.preprocessing import StandardScaler
from spotipy_section.graphPlaylist import get_all_music_features, ALL_FEATURE_LABELS, view_scatter_graph
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression


# General PCA
def standardizer(df):
    x = df.loc[:, ALL_FEATURE_LABELS].values
    x = StandardScaler().fit_transform(x)
    return x


def princomp(data, target):
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(data)
    principal_df = pd.DataFrame(data=principal_components, columns=['pc1', 'pc2'])
    final_df = pd.concat([principal_df, target], axis=1)
    return final_df


# Application on ML
def standardizer2(df):
    # get data
    train_data = df.iloc[:9, :]
    test_data = df.iloc[10:, :]

    # define scaler
    scaler = StandardScaler()

    # fit scalar to training data only
    scaler.fit(train_data)

    # Transform both datasets
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)
    return train_data, test_data


def princomp2(train_data, test_data):
    pca = PCA(0.95)
    pca.fit(train_data)
    # Transform both data
    train_data = pca.transform(train_data)
    test_data = pca.transform(test_data)
    return train_data, test_data


def log_reg(train_data):
    train_lbl = [0, 1, 1, 1, 0, 0, 0, 1, 0]
    # train_lbl = pd.DataFrame({'label': joy_label_list})
    logistic_regr = LogisticRegression(solver='lbfgs')
    logistic_regr.fit(train_data, train_lbl)
    return logistic_regr


def classify(user_df):
    # Gets the list of tracks from the user
    track_list = user_df['track_id'].tolist()

    # Gets the all the musical features of each song
    track_features = get_all_music_features(track_list)

    # standardizes the data
    std_data = standardizer(track_features)
    target = user_df[['joy', 'sadness']]

    # does PCA on the standardized data
    pc_data = princomp(std_data, target)

    # Graph data
    view_scatter_graph(pc_data)


def classify2(user_df):
    # Gets the list of tracks from the user
    track_list = user_df['track_id'].tolist()

    # Gets the all the musical features of each song
    track_features = get_all_music_features(track_list)

    # standardizes the data
    train_data, test_data = standardizer2(track_features)
    # does PCA on the standardized data
    train_data, test_data = princomp2(train_data, test_data)
    # logistic regression
    log_reg_model = log_reg(train_data)

    # predict one value
    print(log_reg_model.predict(test_data[0:4]))

    # Graph data
    # view_scatter_graph(pc_data)
