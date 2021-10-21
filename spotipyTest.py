import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
from IPython.display import display
from tabulate import tabulate

spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(client_id='ed61ad2eb1ac48f38a5971328cec9f01', client_secret='3977cda8a7a14e63b5bdf985c0a5b440'))

artist_name = []
track_name = []
popularity = []
track_id = []

for i in range(1):
    track_results = spotify.search(q='year:2021', type='track', limit=20, offset=i)
    for i, t in enumerate(track_results['tracks']['items']):
        artist_name.append(t['artists'][0]['name'])
        track_name.append(t['name'])
        popularity.append(t['popularity'])

track_dataframe = pd.DataFrame({'artist_name' : artist_name})

# print(tabulate(track_dataframe, headers = 'keys', tablefmt = 'psql'))

print(track_dataframe.info)




