import requests
from IPython.core.display import display
from bs4 import BeautifulSoup
import re
import unidecode

from sentiment.sentiment_analyser import get_lyr_senti


def get_lyrics_senti(song_name, artist_name):
    song_name = clean_name(song_name)
    artist_name = clean_name(artist_name)
    page_name = 'https://genius.com/' + artist_name + '-' + song_name + '-' + 'lyrics'
    print(page_name)
    page = requests.get(page_name)
    soup = BeautifulSoup(page.content, 'lxml')
    lyrics = ""

    for tag in soup.select('div[class^="Lyrics__Container"], .song_body-lyrics p'):
        text = tag.get_text(strip=True, separator='\n')
        if text:
            lyrics += re.sub("[\(\[].*?[\)\]]", "", text)

    return get_lyr_senti(lyrics)


def clean_name(name):
    name = name.replace(" ", "-")
    name = name.replace("'", "")
    name = re.sub("[\(\[].*?[\)\]]", "", name)
    name = re.sub(r'-$','', name)
    name = unidecode.unidecode(name)
    return name.lower()
