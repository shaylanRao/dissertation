from matplotlib import pyplot as plt
import numpy as np

from classification.classification import KNeighborRegressor, DecisionTree
from spotipy_section.graphPlaylist import get_song_list_ids, get_all_music_features


# An object containing the model, scalar and pca used to fit the specified emotion
class EmotionModel:
    def __init__(self, emotion, data_to_graph):
        self.emotion = emotion
        # self.model, self.scalar, self.pca = KNeighborRegressor(data_to_graph, emotion).drive()
        self.model, self.scalar, self.pca = DecisionTree(data_to_graph, emotion).drive()

    # Makes predictions for the playlist chosen
    def predict(self):
        return self.model.predict(get_playlist_data(self))


# Gets prepared data from playlist in this method
def get_playlist_data(emotion_model):
    track_list = get_song_list_ids('0IAG5sPikOCo5nvyKJjCYo')
    tracks_features = get_all_music_features(track_list)

    tracks_features = emotion_model.scalar.transform(tracks_features)
    predict_playlist_data = emotion_model.pca.transform(tracks_features)
    return predict_playlist_data


class Prediction:
    def __init__(self, data_to_graph):
        self.anger = EmotionModel("anger", data_to_graph)
        self.fear = EmotionModel("fear", data_to_graph)
        self.joy = EmotionModel("joy", data_to_graph)
        self.sadness = EmotionModel("sadness", data_to_graph)

    anger_pred = []
    fear_pred = []
    joy_pred = []
    sadness_pred = []

    def drive(self):
        self.anger_pred = self.anger.predict()
        self.fear_pred = self.fear.predict()
        self.joy_pred = self.joy.predict()
        self.sadness_pred = self.sadness.predict()

        print("Playlist prediction:")
        self.graph()

    def graph(self):
        # TODO clean up limit of graph x axis
        x = range(1, len(get_song_list_ids('0IAG5sPikOCo5nvyKJjCYo')) + 1)
        j, f, a, s = map(list, zip(*sorted(zip(self.joy_pred, self.fear_pred, self.anger_pred, self.sadness_pred))))
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(x, a, label="Anger", c='r')
        ax.plot(x, f, label="Fear", c='m')
        ax.plot(x, j, label="Joy", c='y')
        ax.plot(x, s, label="Sadness", c='b')
        plt.legend()
        plt.ylim(0, 1)
        plt.show()

    def verbal_classifier(self):
        pass
