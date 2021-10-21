import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(client_id='ed61ad2eb1ac48f38a5971328cec9f01', client_secret='3977cda8a7a14e63b5bdf985c0a5b440'))

lz_uri = 'spotify:artist:36QJpDe2go2KgaRleHCDTp'
results = spotify.artist_top_tracks(lz_uri)

for track in results['tracks'][:10]:
    print('track    : ' + track['name'])
    print('audio    : ' + track['preview_url'])
    print('cover art: ' + track['album']['images'][0]['url'])
    print()

