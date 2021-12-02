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


def graph_one_playlist(song_list_graph_one):
    print("GRAPH 1: ", song_list_graph_one)
    vw = 'acousticness'
    vx = 'valence'
    vy = 'energy'
    vz = 'speechiness'

    w, x, y, z = get_x_y_z(song_list_graph_one, vw, vx, vy, vz)
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

# TODO add error detection (track_id)
def label_heatmap(song_label_df):
    # test_3d()
    # test_2d_heatmap()
    example_user_name = song_label_df['user_name'][1]
    song_label_df = song_label_df[song_label_df['user_name'] == example_user_name]
    song_label_df = song_label_df[song_label_df['anger'].notna()]
    track_list = song_label_df['track_id'].tolist()

    surf_plot_2(track_list, song_label_df)
    # interactive_graph()

    # vw = 'acousticness'
    # vx = 'valence'
    # vy = 'energy'
    # vz = 'speechiness'
    #
    # w, x, y, z = get_x_y_z(track_list, vw, vx, vy, vz)
    # fig1 = plt.figure()
    #
    # w = song_label_df['sadness']
    #
    # ax = fig1.add_subplot(projection='3d', xlim=(0, 1), ylim=(0, 1), zlim=(0, 1))
    #
    # # 4-dimensions
    # img = ax.scatter(x, y, z, c=w, cmap='RdBu', vmin=0, vmax=1, marker=".")
    # fig1.colorbar(img)
    #
    # # 3-dimensions
    # # img = ax.scatter(x, y, z, marker=".")
    #
    # ax.set_xlabel(vx)
    # ax.set_ylabel(vy)
    # ax.set_zlabel(vz)
    # plt.show()


def test_2d_heatmap(track_list, song_label_df):
    vw = 'acousticness'
    vx = 'valence'
    vy = 'energy'
    vz = 'speechiness'

    w, x, y, z = get_x_y_z(track_list, vw, vx, vy, vz)

    w = song_label_df['sadness']

    heatmap, xedges, yedges = np.histogram2d(x, y, bins=100)
    extent = [0, 1, 0, 1]

    plt.clf()
    plt.imshow(heatmap.T, extent=extent, origin='lower')
    # plt.xlim(0, 1), ylim(0, 1)
    plt.show()


def surface_plot(track_list, song_label_df):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    import numpy as np

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    vw = 'acousticness'
    vx = 'valence'
    vy = 'energy'
    vz = 'speechiness'

    w, x, y, z = get_x_y_z(track_list, vw, vx, vy, vz)
    w = song_label_df['sadness']
    print(w)
    # Make data.
    xlist = [0.876, 0.461, 0.601, 0.488, 0.403, 0.315, 0.711, 0.462, 0.467, 0.439, 0.647, 0.403, 0.31, 0.494]
    ylist = [0.861, 0.886, 0.63, 0.748, 0.842, 0.424, 0.711, 0.644, 0.749, 0.639, 0.987, 0.842, 0.616, 0.833]

    xlist = sorted(xlist)
    ylist = sorted(ylist)
    # xlist = [0.876, 0.461, 0.601, 0.488, 0.403, 0.315, 0.5]
    # ylist = [0.861, 0.886, 0.63, 0.748, 0.842, 0.424]

    Z = np.random.rand(14, 14)

    X, Y = np.meshgrid(xlist, ylist)

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.plasma,
                           linewidth=0, vmin=0, vmax=1, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(0, 1), ylim(0, 1), xlim(0, 1)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


def surf_plot_2(track_list, song_label_df):
    vw = 'acousticness'
    vx = 'valence'
    vy = 'energy'
    vz = 'speechiness'

    w, x, y, z = get_x_y_z(track_list, vw, vx, vy, vz)

    w = song_label_df['sadness']

    data = list(zip(x, y, w))
    x, y, z = zip(*data)
    # z = list(map(float, w))
    grid_x, grid_y = np.mgrid[min(x):max(x):100j, min(y):max(y):100j]
    grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')
    grid_z[np.isnan(grid_z)] = 0
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(grid_x, grid_y, grid_z, cmap=plt.cm.plasma)

    ax.set_xlabel(vx)
    ax.set_ylabel(vy)
    ax.set_zlabel('sadness')

    ax.set_zlim(0, 1)
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.show()


def test_3d():
    # creating a dummy dataset
    x = np.random.randint(low=100, high=500, size=(1000,))
    y = np.random.randint(low=300, high=500, size=(1000,))
    z = np.random.randint(low=200, high=500, size=(1000,))
    colo = [x + y + z]

    # creating figures
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # setting color bar
    color_map = cm.ScalarMappable()
    color_map.set_array([colo])

    # creating the heatmap
    img = ax.scatter(x, y, z, marker='s',
                     s=200, color='green')
    plt.colorbar(color_map)

    # adding title and labels
    ax.set_title("3D Heatmap")
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

    # displaying plot
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
