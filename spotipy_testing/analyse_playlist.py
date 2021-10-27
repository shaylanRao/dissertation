import spotipy
from spotipy.oauth2 import SpotifyOAuth
import datetime
import matplotlib.pyplot as plt
import numpy as np

# Define scope and link to app
scope = "playlist-read-private"
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id='ed61ad2eb1ac48f38a5971328cec9f01',
                                               client_secret='3977cda8a7a14e63b5bdf985c0a5b440', scope=scope,
                                               redirect_uri='http://localhost:8080', show_dialog=True))

vary = 'energy'
varx = 'valence'


def get_attribute(featuresp, attribute):
    return [featuresp[i][attribute] for i in range(len(features))]


# Get playlist 'Happy playlist' via id
pl_id = '0IAG5sPikOCo5nvyKJjCYo'

offset = 0

#  Get a list of all tracks on playlist
response = sp.playlist_items(pl_id,
                             offset=offset,
                             fields='items.track.id,total',
                             additional_types=['track'])

song_list_ids = []
for song in (response['items']):
    song_list_ids.append(song['track']['id'])

# Get features for each song in list
features = sp.audio_features(song_list_ids)

y1 = get_attribute(features, vary)
x1 = get_attribute(features, varx)


#########################################################################

pl_id = '3rsdTIkZaDaZT5gL5oPPo1'

offset = 0

#  Get a list of all tracks on playlist
response = sp.playlist_items(pl_id,
                             offset=offset,
                             fields='items.track.id,total',
                             additional_types=['track'])

song_list_ids = []
for song in (response['items']):
    song_list_ids.append(song['track']['id'])

# Get features for each song in list
features = sp.audio_features(song_list_ids)

y2 = get_attribute(features, vary)
x2 = get_attribute(features, varx)

plt.plot(x1, y1, 'bo')
plt.plot(x2, y2, 'bo')
plt.xlabel(varx)
plt.ylabel(vary)
plt.plot(range(1))
plt.yticks(np.arange(0, 1, 0.1))
plt.xlim(0, 1)
plt.ylim(0, 1)

plt.show()

print(features[2])
