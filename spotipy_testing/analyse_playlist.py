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


def get_attribute(featuresp, attribute):
    return [featuresp[i][attribute] for i in range(len(featuresp))]

# Returns attributes x, y, z given playlist id
def get_x_y_z(pl_id):
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

    print(features[0])

    y = get_attribute(features, vary)
    x = get_attribute(features, varx)
    z = get_attribute(features, varz)

    return x, y, z


def show_graph():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(x1, y1, z1, c='y', marker='o')
    ax.scatter(x2, y2, z2, c='b', marker='o')
    ax.scatter(x3, y3, z3, c='r', marker='o')


    ax.set_xlabel(varx)
    ax.set_ylabel(vary)
    ax.set_zlabel(varz)

    plt.show()


# Define the 3 variables
varx = 'valence'
vary = 'energy'
varz = 'speechiness'

x1, y1, z1 = get_x_y_z('0IAG5sPikOCo5nvyKJjCYo')
x2, y2, z2 = get_x_y_z('78FHjijA1gBLuVx4qmcHq6')
x3, y3, z3 = get_x_y_z('3aBeWOxyVcFupF8sKMm2k7')


show_graph()

