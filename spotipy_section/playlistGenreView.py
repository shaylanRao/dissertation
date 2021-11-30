import graphPlaylist
from tweepy import hashtagSearch

vw = 'acousticness'
vx = 'valence'
vy = 'energy'
vz = 'speechiness'

song_ids1 = (graphPlaylist.get_song_list_ids('37i9dQZF1DWXRqgorJj26U'))
song_ids2 = (graphPlaylist.get_song_list_ids('37i9dQZF1DXbITWG1ZJKYt'))
song_id_twitter = hashtagSearch.get_twitter_song_list()

# w1, x1, y1, z1 = graphPlaylist.get_x_y_z(song_ids1, vw, vx, vy, vz)
# w2, x2, y2, z2 = graphPlaylist.get_x_y_z(song_ids2, vw, vx, vy, vz)
wt, xt, yt, zt = graphPlaylist.get_x_y_z(song_id_twitter, vw, vx, vy, vz)

# graphPlaylist.graph_one_playlist(vx, vy, vz, w1, x1, y1, z1)
# graphPlaylist.graph_two_playlist(vx, vy, vz, w1, x1, y1, z1, w2, x2, y2, z2)

graphPlaylist.graph_one_playlist(vx, vy, vz, wt, xt, yt, zt)

