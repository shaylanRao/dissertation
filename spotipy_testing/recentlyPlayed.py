import spotipy
from spotipy.oauth2 import SpotifyOAuth
import datetime

scope = "user-read-recently-played playlist-read-private"

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id='ed61ad2eb1ac48f38a5971328cec9f01',
                                               client_secret='3977cda8a7a14e63b5bdf985c0a5b440', scope=scope,
                                               redirect_uri='http://localhost:8080', show_dialog=True))


def get_recently_played():
    results = sp.current_user_recently_played(limit=50)
    track_ids = []

    for idx, item in enumerate(results['items']):
        # track = item['track']
        # track_ids.append(item['track']['uri'])
        # time = item['played_at'].replace("T", "   ")
        # print(idx, item['track']['uri'], " – ", track['name'])
        # print(time)
        song_id = item['track']['uri'].split(":")
        track_ids.append(song_id[2])

    return track_ids


# print(results['items'])
# print(get_recently_played())
# print("Current-time: ", datetime.datetime.now())
