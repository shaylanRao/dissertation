import spotipy
from cffi.backend_ctypes import xrange
from spotipy.oauth2 import SpotifyOAuth
import datetime
import recentlyPlayed
import matplotlib.pyplot as plt
import numpy as np

# Define scope and link to app
scope = "user-read-recently-played playlist-read-private"
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id='ed61ad2eb1ac48f38a5971328cec9f01',
                                               client_secret='3977cda8a7a14e63b5bdf985c0a5b440', scope=scope,
                                               redirect_uri='http://localhost:8080', show_dialog=True))


def get_attribute(featuresp, attribute):
    return [featuresp[i][attribute] for i in range(len(featuresp))]


def get_song_list_ids(pl_id):
    offset = 0
    response = sp.playlist_items(pl_id,
                                 offset=offset,
                                 fields='items.track.id,total',
                                 additional_types=['track'])
    song_list_ids = []
    for song in (response['items']):
        song_list_ids.append(song['track']['id'])

    return song_list_ids


# Returns attributes x, y, z given playlist id
def get_x_y_z(song_list_ids, varw, varx, vary, varz):
    # Get features for each song in list
    features = sp.audio_features(song_list_ids)

    # print(features[0])
    w = get_attribute(features, varw)
    y = get_attribute(features, vary)
    x = get_attribute(features, varx)
    z = get_attribute(features, varz)

    return w, x, y, z


def show_graph_sample(varx, vary, varz, x1, x2, x3, y1, y2, y3, z1, z2, z3):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(x1, y1, z1, c='y', marker='o')
    ax.scatter(x2, y2, z2, c='b', marker='o')
    ax.scatter(x3, y3, z3, c='r', marker='o')

    ax.set_xlabel(varx)
    ax.set_ylabel(vary)
    ax.set_zlabel(varz)

    plt.show()


def graph_one_playlist(varx, vary, varz, w, x, y, z):
    fig1 = plt.figure()
    ax = fig1.add_subplot(projection='3d')

    # img = ax.scatter(x1, y1, z1, c=w1, cmap=plt.hot(), marker=",")
    img = ax.scatter(x, y, z, c=w, cmap=plt.hot(), marker="v")
    # img = ax.scatter(x3, y3, z3, c=w3, cmap=plt.hot(), marker="o")

    ax.set_xlabel(varx)
    ax.set_ylabel(vary)
    ax.set_zlabel(varz)

    fig1.colorbar(img)

    plt.show()


def graph_two_playlist(varx, vary, varz, w1, x1, y1, z1, w2, x2, y2, z2):
    fig2 = plt.figure()
    ax = fig2.add_subplot(projection='3d')

    img = ax.scatter(x1, y1, z1, c=w1, cmap=plt.hot(), marker=",")
    img = ax.scatter(x2, y2, z2, c=w2, cmap=plt.hot(), marker="v")

    ax.set_xlabel(varx)
    ax.set_ylabel(vary)
    ax.set_zlabel(varz)

    # for ii in range(0, 360, 1):
    #     ax.view_init(elev=10., azim=ii)
    #     plt.savefig("movie%d.png" % ii)

    fig2.colorbar(img)

    plt.show()


# Define the 3 variables
# varx = 'valence'
# vary = 'energy'
# varz = 'speechiness'
#
#
# x1, y1, z1 = get_x_y_z(get_song_list_ids('0IAG5sPikOCo5nvyKJjCYo'))
# x2, y2, z2 = get_x_y_z(get_song_list_ids('78FHjijA1gBLuVx4qmcHq6'))
# # x3, y3, z3 = get_x_y_z(spotipyDocCode.get_recently_played())
#
# x3, y3, z3 = get_x_y_z(get_song_list_ids('3aBeWOxyVcFupF8sKMm2k7'))
#
# show_graph()


# -----------------------------------------------------------------------------

def main():
    vw = 'acousticness'
    vx = 'valence'
    vy = 'energy'
    vz = 'speechiness'

    w1, x1, y1, z1 = get_x_y_z(get_song_list_ids('0IAG5sPikOCo5nvyKJjCYo'), vw, vx, vy, vz)
    w2, x2, y2, z2 = get_x_y_z(get_song_list_ids('78FHjijA1gBLuVx4qmcHq6'), vw, vx, vy, vz)
    w3, x3, y3, z3 = get_x_y_z(get_song_list_ids('3aBeWOxyVcFupF8sKMm2k7'), vw, vx, vy, vz)
    # w3, x3, y3, z3 = get_x_y_z(spotipyDocCode.get_recently_played())

    show_graph_sample(vx, vy, vz, x1, x2, x3, y1, y2, y3, z1, z2, z3)


print("Running")
