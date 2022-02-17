import math
import seaborn as sns
import numpy as np
import pandas
import pandas as pd
from IPython.core.display import display
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

from spotipy_section.graphPlaylist import get_all_music_features, ALL_FEATURE_LABELS, view_scatter_graph, \
    get_song_list_ids
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge


class LinLogReg:
    def __init__(self, data, emotion):
        self.user_data = data
        self.EMOTION = emotion

    TRAIN_DATA_PROPORTION = 0.7
    LOG_THRESHOLD = 0.5
    DATA_PRESERVED = 0.95

    train_lbl = pandas.Series
    test_lbl = pandas.Series
    split_pos = 0
    train_data = pandas.Series
    test_data = pandas.Series

    scalar = None
    pca_model = None
    predict_playlist_data = None

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
        # get_playlist_data()
        # Gets the list of tracks from the user
        track_list = self.user_data['track_id'].tolist()

        # Gets the all the musical features of each song
        track_features = get_all_music_features(track_list)

        # Merges track features to the track and other data
        self.user_data = pd.concat([self.user_data, track_features], axis=1)
        # self.user_data.to_csv('userdata.csv')

        # Gets the size of the overall dataframe
        df_size = self.user_data.shape[1]

        # Randomizes rows
        shuffled_data = self.user_data.sample(frac=1)

        # Sets music components as data (and includes lyrical data if any is present)
        # (10 represents the number of music features)
        self.set_train_test_data(shuffled_data.iloc[:, df_size-10:])

        # Sets music data to also include lyrical sentiment
        # self.set_train_test_data(shuffled_data.iloc[:, df_size-17:])

        # and anger to tentative as labels
        self.set_train_test_labels(shuffled_data.iloc[:, 2:9])

    def set_train_test_data(self, music_data):
        self.split_pos = math.ceil((len(music_data)) * self.TRAIN_DATA_PROPORTION)
        self.train_data = music_data.iloc[:self.split_pos, :]
        self.test_data = music_data.iloc[self.split_pos:, :]

    def set_train_test_labels(self, labels):
        labels = labels[self.EMOTION]
        self.train_lbl = labels[:self.split_pos]
        self.test_lbl = labels[self.split_pos:]

    def get_log_data_labels(self, label):
        log_label = [int(x >= self.LOG_THRESHOLD) for x in label.tolist()]
        return log_label

    # --- PCA (for use on ML techniques) ---
    def standardizer(self):
        # define scalar
        scalar = StandardScaler()

        # fit scalar to training data only
        scalar.fit(self.train_data)
        self.scalar = scalar

        # Transform both datasets
        self.train_data = scalar.transform(self.train_data)
        self.test_data = scalar.transform(self.test_data)

    # PCA function
    def prin_comp(self):
        # Keeps the relevant number of components to ensure 95% of original data is preserved
        pca = PCA(self.DATA_PRESERVED)
        pca.fit(self.train_data)

        self.pca_model = pca

        # Transform both data
        self.train_data = pca.transform(self.train_data)
        self.test_data = pca.transform(self.test_data)
        # print("Originality per dimension increase")
        # print(np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4) * 100))

    # --- Different models ---
    # Logistic regression modeling
    def log_reg_func(self):
        log_reg_train_lbl = self.get_log_data_labels(self.train_lbl)
        logistic_reg = LogisticRegression(solver='lbfgs', fit_intercept=False)
        logistic_reg.fit(self.train_data, log_reg_train_lbl)
        return logistic_reg

    # Linear regression modeling
    def lin_reg_func(self):
        linear_reg = LinearRegression(positive=True, fit_intercept=False)
        linear_reg.fit(self.train_data, self.train_lbl)
        return linear_reg

    # Ridge
    def ridge_reg_func(self):
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
        lin_reg_model = self.lin_reg_func()
        # predicting all values
        print("Linear Regression:")
        print("Actual:")
        print(np.around(self.test_lbl.tolist(), 3))
        print("Predicted:")
        print(np.around(lin_reg_model.predict(self.test_data), 3))
        print("")

    def ridge_reg_classifier(self):
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
        # standardizes the data
        self.standardizer()
        # Does PCA on data
        self.prin_comp()

        self.log_reg_classifier()
        self.lin_reg_classifier()
        self.ridge_reg_classifier()

    # Evaluating metrics for models using predictions
    def evaluate_model(self, test_preds):
        # Calculates mean squared error on test data
        mse = mean_squared_error(self.test_lbl, test_preds)
        rmse = math.sqrt(mse)

        # Generates pair-wise Pearsons correlation
        corr_matrix = self.user_data.corr()

        r2 = r2_score(self.test_lbl, test_preds)

        print("RSME:")
        print(rmse)
        print("")

        print("R2 Score")
        print(r2)
        print("")

        print("Correlation matrix")
        display(corr_matrix[self.EMOTION])
        print("")


class KernelSVC(LinLogReg):
    def drive(self):
        self.prep_data()
        self.standardizer()
        self.prin_comp()
        y_pred = self.kernel("gaus")
        self.eval(y_pred)

    def kernel(self, k_type):
        if k_type == "poly":
            sv_classifier = SVC(kernel='poly', degree=8)
        elif k_type == "gaus":
            sv_classifier = SVC(kernel='rbf')
        elif k_type == "sigmoid":
            sv_classifier = SVC(kernel='sigmoid')
        else:
            print("Used Default")
            sv_classifier = SVC(kernel='poly', degree=8)
        sv_classifier.fit(self.train_data, self.get_log_data_labels(self.train_lbl))
        y_pred = sv_classifier.predict(self.test_data)
        return y_pred
        # print("ACTUAL")
        # print(self.get_log_data_labels(self.test_lbl))
        # print("PREDICTION")
        # print(y_pred)

    def eval(self, y_pred):
        print("Values")
        print(self.get_log_data_labels(self.test_lbl))
        print(y_pred)
        print(confusion_matrix(self.get_log_data_labels(self.test_lbl), y_pred))
        print(classification_report(self.get_log_data_labels(self.test_lbl), y_pred))


# TODO
class KernelSVR(LinLogReg):
    def drive(self):
        pass


class KNeighborRegressor(LinLogReg):
    def drive(self):
        print("KNR ", self.EMOTION, ":")
        self.prep_data()
        self.standardizer()
        self.prin_comp()
        best_params = self.grid_search()
        bagging_model = self.knr_bagging(best_params['n_neighbors'], best_params['leaf_size'], best_params['weights'])
        return bagging_model, self.scalar, self.pca_model
        # self.correlation_graph()

    def knr_bagging(self, k_num, leaf_sz, weight):
        # Initialises KNR using optimal parameters
        bagged_knr = KNeighborsRegressor(n_neighbors=k_num, leaf_size=leaf_sz, weights=weight)
        print("k: ", k_num, " leaf: ", leaf_sz, " weights: ", weight)
        # Uses bagging to improve model
        bagging_model = BaggingRegressor(bagged_knr, n_estimators=1000)
        # Fits the bagged model to the training data with labels
        bagging_model.fit(self.train_data, self.train_lbl)
        # Makes predictions on test data using fitted model
        test_preds = bagging_model.predict(self.test_data)
        # Evaluates model using function
        self.evaluate_model(test_preds)
        # make prediction on playlist data
        return bagging_model
        # self.knr_boosting()

    # Boosting regressor instead of KNR and Bagging
    # def knr_boosting(self):
    #     print("Boost model:")
    #     boost_model = GradientBoostingRegressor(random_state=0)
    #     boost_model.fit(self.train_data, self.train_lbl)
    #     test_preds = boost_model.predict(self.test_data)
    #     mse = mean_squared_error(self.test_lbl, test_preds)
    #     rmse = math.sqrt(mse)
    #     print(rmse)

    # Testing to see correlation for energy and loudness and prediction on colorbar
    def plot_model(self, test_preds):
        cmap = sns.cubehelix_palette(as_cmap=True)
        f, ax = plt.subplots()
        points = ax.scatter(self.test_data['energy'], self.test_data['loudness'], c=test_preds, s=50, cmap=cmap)
        f.colorbar(points)
        plt.show()

    # Tuning
    def grid_search(self):
        max_n = math.floor(len(self.train_data)*(4/5))
        # Define range to test for parameters
        parameters = {"n_neighbors": range(2, max_n), 'weights': ['uniform', 'distance'], "leaf_size": range(15, 30)}
        gridsearch = GridSearchCV(KNeighborsRegressor(), parameters)
        # Find the parameters for the given data
        gridsearch.fit(self.train_data, self.train_lbl)
        # Returns the optimal parameters for the training data
        return gridsearch.best_params_


class DecisionTree(LinLogReg):
    def drive(self):
        print("Dec Tree Reg ", self.EMOTION, ":")
        self.prep_data()
        self.standardizer()
        self.prin_comp()
        dec_tree = self.decision_tree_regressor()
        return dec_tree, self.scalar, self.pca_model
        # self.correlation_graph()

    def decision_tree_regressor(self):
        dec_tree = tree.DecisionTreeRegressor()
        dec_tree = dec_tree.fit(self.train_data, self.train_lbl)
        test_preds = dec_tree.predict(self.test_data)
        self.evaluate_model(test_preds)
        return dec_tree
