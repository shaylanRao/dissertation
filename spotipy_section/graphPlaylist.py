import matplotlib as matplotlib
import spotipy
from IPython.core.display import display
from cffi.backend_ctypes import xrange
from spotipy.oauth2 import SpotifyOAuth
from IPython import get_ipython
# get_ipython().magic('matplotlib inline')
import datetime
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib import cm

from mpl_toolkits.mplot3d import Axes3D
import scipy
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import numpy as np

# 3D Heatmap in Python using matplotlib

# importing required libraries
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from pylab import *

# get_ipython().run_line_magic('matplotlib', 'inline')

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
def get_w_to_z(song_list_ids, varw, varx, vary, varz):
    # Get features for each song in list
    features = sp.audio_features(song_list_ids)

    # print(features[0])
    w = get_attribute(features, varw)
    y = get_attribute(features, vary)
    x = get_attribute(features, varx)
    z = get_attribute(features, varz)

    return w, x, y, z

def get_x_y_z(song_list_ids, varx, vary, varz):
    # Get features for each song in list
    features = sp.audio_features(song_list_ids)

    # print(features[0])
    y = get_attribute(features, vary)
    x = get_attribute(features, varx)
    z = get_attribute(features, varz)

    return x, y, z


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


def graph_one_playlist(song_list_graph_one):
    print("GRAPH 1: ", song_list_graph_one)
    vw = 'acousticness'
    vx = 'valence'
    vy = 'energy'
    vz = 'speechiness'

    w, x, y, z = get_w_to_z(song_list_graph_one, vw, vx, vy, vz)
    fig1 = plt.figure()

    ax = fig1.add_subplot(projection='3d', xlim=(0, 1), ylim=(0, 1), zlim=(0, 1))

    # 4-dimensions
    img = ax.scatter(x, y, z, c=w, cmap=plt.hot(), marker=".")
    fig1.colorbar(img)

    # 3-dimensions
    # img = ax.scatter(x, y, z, marker=".")

    ax.set_xlabel(vx)
    ax.set_ylabel(vy)
    ax.set_zlabel(vz)
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


# -----------------------------------------------------------------------------

# TODO add error detection (track_id)
def label_heatmap(song_label_df):
    # test_3d()
    # test_2d_heatmap()
    example_user_name = song_label_df['user_name'][1]
    song_label_df = song_label_df[song_label_df['user_name'] == example_user_name]
    song_label_df = song_label_df[song_label_df['anger'].notna()]
    track_list = song_label_df['track_id'].tolist()
    vx = 'valence'
    vy = 'energy'
    vz = 'speechiness'

    x, y, z = get_x_y_z(track_list, vx, vy, vz)

    # Define sentiment (label)
    label_name = 'sadness'
    label = song_label_df[label_name]
    surf_plot_2(x, y, label, label_name)

    label_name = 'joy'
    label = song_label_df[label_name]
    surf_plot_2(x, y, label, label_name)


def surf_plot_2(x, y, label, label_name):
    data = list(zip(x, y, label))
    x, y, label = zip(*data)
    grid_x, grid_y = np.mgrid[0:1:100j, 0:1:100j]
    grid_z = griddata((x, y), label, (grid_x, grid_y), method='cubic')
    grid_z[np.isnan(grid_z)] = 0
    fig = plt.figure()
    # ax = fig.gca(projection='3d')
    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(grid_x, grid_y, grid_z, cmap=plt.cm.plasma)

    # Fixed label names for x and y
    ax.set_xlabel('Valence')
    ax.set_ylabel('Energy')
    ax.set_zlabel(label_name)

    ax.set_zlim(0, 1)
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.show()


def main():
    # w2, x2, y2, z2 = get_x_y_z(get_song_list_ids('78FHjijA1gBLuVx4qmcHq6'), vw, vx, vy, vz)
    # w3, x3, y3, z3 = get_x_y_z(get_song_list_ids('3aBeWOxyVcFupF8sKMm2k7'), vw, vx, vy, vz)
    # w3, x3, y3, z3 = get_x_y_z(recentlyPlayed.get_recently_played())

    # w1, x1, y1, z1 = get_x_y_z(get_song_list_ids('4ghvB1pIW4LTUn0RYrfuD5'), vw, vx, vy, vz)
    # graph_one_playlist(get_song_list_ids('0IAG5sPikOCo5nvyKJjCYo'))

    graph_one_playlist(get_song_list_ids('78FHjijA1gBLuVx4qmcHq6'))

    graph_one_playlist(get_song_list_ids('3tpc6g7KWkUF5TVt0zT8q6'))

    # w1, x1, y1, z1 = get_x_y_z(get_song_list_ids('3aBeWOxyVcFupF8sKMm2k7'), vw, vx, vy, vz)
    # graph_one_playlist(vx, vy, vz, w1, x1, y1, z1)
    # show_graph_sample(vx, vy, vz, x1, x2, x3, y1, y2, y3, z1, z2, z3)

# main()
