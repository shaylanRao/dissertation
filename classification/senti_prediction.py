from classification.classification import KNeighborRegressor
from spotipy_section.graphPlaylist import get_song_list_ids, get_all_music_features


# An object containing the model, scalar and pca used to fit the specified emotion
class EmotionModel:
    def __init__(self, emotion, data_to_graph):
        self.emotion = emotion
        self.model, self.scalar, self.pca = KNeighborRegressor(data_to_graph, emotion)


# Gets prepared data from playlist in this method
class Prediction:
    def __init__(self, data_to_graph):
        self.anger = EmotionModel("anger", data_to_graph)
        self.fear = EmotionModel("fear", data_to_graph)
        self.joy = EmotionModel("joy", data_to_graph)
        self.sadness = EmotionModel("sadness", data_to_graph)

    def get_playlist_data(self):
        track_list = get_song_list_ids('7d6WFDrKCCz4veVu0p7PVt')
        tracks_features = get_all_music_features(track_list)

        tracks_features = self.joy.scalar.transform(tracks_features)
        predict_playlist_data = self.joy.pca.transform(tracks_features)
        return predict_playlist_data

    def drive(self):
        playlist_pred = self.joy.model.predict(self.get_playlist_data())
        print("Playlist prediction:")
        print(playlist_pred)

    def verbal_classifier(self):
        pass
