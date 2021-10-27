import spotipy
from spotipy.oauth2 import SpotifyOAuth
import datetime

scope = "user-read-recently-played"

sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id='ed61ad2eb1ac48f38a5971328cec9f01', client_secret='3977cda8a7a14e63b5bdf985c0a5b440', scope=scope, redirect_uri = 'http://localhost:8080', show_dialog = True))


results = sp.current_user_recently_played(limit=10)
tids = []


for idx, item in enumerate(results['items']):
    track = item['track']
    tids.append(item['track']['uri'])
    # print(idx, item['track']['uri'], " – ", track['name'])

features = sp.audio_features(tids)

print(features[0])
print("Current-time: ", datetime.datetime.now())