import json
import re
from datetime import timedelta
from string import punctuation

import tweepy
import innit_tweepy
import pandas as pd
import numpy as np
from IPython.display import display

from sentiment.sentimentAnalyser import get_senti
from spotipy_section.graphPlaylist import graph_one_playlist

api = innit_tweepy.getTweepyApi()

choice = '"open.spotify.com/track" lang:en exclude:replies -filter:retweets'
song_list = []
user_screen_name_list = []
all_tweets = []

column_names = ["user_name", "text", "track_id", "tweet_id", "time"]
all_s_tweets = pd.DataFrame(columns=column_names)

label_df = pd.DataFrame()

black_list = ['BBCR6MusicBot', 'BBC2MusicBot', 'KiddysplaceMx', 'Spotweefy', 'JohnOxley777', 'bieberonspotify',
              'LiveMixPlay', 'CAA_Official', 'fabclaxton', 'THXJRT', 'moevazquez']

num_users = 10
max_song_tweets = 20
num_before_tweets = 3
s_tweet_minimum = 6

# Gets recent tweets which include spotify links,  .items(n) -> how many different users will be searched
recent_s_tweets = tweepy.Cursor(api.search_tweets, q=choice, result_type='recent').items(num_users)


# Gets the Spotify song id
def song_id_in_url(urls):
    try:
        url = urls[0]['expanded_url']
    except IndexError:
        url = ""

    if url[0:31] == "https://open.spotify.com/track/":
        url = (url[31:53])
        return url

    return ""


# Adds to pre-defined list of users, the usernames from tweets
def get_user_list():
    for tweet in recent_s_tweets:
        user_screen_name_list.append(tweet.user.screen_name)


# Gets only spotify tweets from a user - passed as string
def get_users_spotify_tweets(screen_name):
    query = '"open.spotify.com/track" lang:en exclude:replies -filter:retweets' + " " + screen_name
    # Gets specified number of tweets that include songs in the tweets
    spotify_tweets = tweepy.Cursor(api.search_tweets, q=query, result_type='recent').items(max_song_tweets)
    return spotify_tweets


# Cleans text data - passed as string
def clean_text(message):
    # Removes space, flattens text
    tweet_text = message.replace('\n', '')

    # Removes urls
    tweet_text = re.sub(r'http\S+', '', tweet_text)

    # Removes any 'special' characters
    tweet_text = re.sub("[^0-9a-zA-Z{} ]+".format(punctuation), "", tweet_text)
    return tweet_text


# Gets tweet message and song url (if there is one)
def get_s_tweet_data(tweet):
    tweet_text = clean_text(tweet.text)
    urls = tweet.entities["urls"]
    song_url = song_id_in_url(urls)
    return tweet_text, song_url


# Gets the size of how many tweets are in the search
def get_size_of_search(tweet_search_items):
    counter = 0
    for i in tweet_search_items:
        counter += 1

    return counter


# Function for counting number of elements
def count_iterable(i):
    return sum(1 for e in i)


# Puts passed data into dataframe
def tabulate_s_tweets(user_name, text, track_id, tweet_id, time):
    df = {'user_name': user_name, 'text': text, 'track_id': track_id, 'tweet_id': tweet_id, 'time': time}
    global all_s_tweets
    all_s_tweets = all_s_tweets.append(df, ignore_index=True)
    return None


# Creates a song list for each user from the dataframe (all_s_tweets)
def create_song_lists():
    all_user_lists = []
    # iterates through each user within s_tweets
    for user in all_s_tweets['user_name'].unique():
        user_song_list = []
        for row in all_s_tweets[all_s_tweets['user_name'] == user].iterrows():
            # Gets track ID from tweet
            if row[1][2] != "":
                user_song_list.append(row[1][2])

        if user_song_list:
            all_user_lists.append(user_song_list)
    return all_user_lists


# TODO appending dataframe of labels to each song
def add_song_label(messages):
    global label_df
    label = get_senti(messages)
    print("Label:")
    display(label)
    print("-------------------------")
    df = pd.DataFrame({label})
    df.transpose()
    display(df)
    label_df.append(df)
    label_df = label_df.append(df)


def get_before_s_tweets():
    # example_user = all_s_tweets.iloc[0]
    # print("user_name:   ", example_user['user_name'])
    count = 0

    # For each user in the dataframe
    for user in all_s_tweets['user_name'].unique():
        print(user, ": ")
        # For each spotify tweet that user has made
        for s_tweet in all_s_tweets[all_s_tweets['user_name'] == user].iterrows():
            messages = ""

            # Gets date (YYY-MM-DD) of tweet - use to limit tweets only going back 7 days - only to keep tweets
            # within bound
            until_date = s_tweet[1][4].date()

            # Increments 1 to account for the current tweet
            until_date += timedelta(days=1)

            # Gets tweet_id
            tweet_id = s_tweet[1][3]

            # Query for tweets from user
            query = 'lang:en exclude:replies -filter:retweets ' + user

            # Gets (upto number declared) tweets from user - until: searches tweets BEFORE given date
            before_s_tweet = tweepy.Cursor(api.search_tweets,
                                           q=query,
                                           result_type='recent',
                                           max_id=tweet_id,
                                           until=until_date
                                           ).items(num_before_tweets)
            for tweet in before_s_tweet:
                # If the tweet is not the spotify tweet
                if tweet.id != tweet_id:
                    # And does not have any other spotify song associated with it
                    if song_id_in_url(tweet.entities["urls"]) == "":
                        messages = '\n'.join([messages, clean_text(tweet.text)])
                    else:
                        # Go to next tweet
                        break
                else:
                    messages = '\n'.join([messages, clean_text(tweet.text)])

            # Gets overall sentiment from past tweets together (rounded sentiment leading upto song)
            #     Acts as label for song
            # print(get_senti(messages))
            add_song_label(messages)


# Remove any blacklisted accounts
def rem_blacklist():
    for user_name in black_list:
        try:
            user_screen_name_list.remove(user_name)
        except ValueError:
            pass


def create_all_s_tweets():
    # Creates list of users who have posted using a spotify link in their tweet
    get_user_list()

    # Removes users on blacklist
    rem_blacklist()
    for user in user_screen_name_list:
        # If there are more than 2 tweets that the user has made which includes a spotify track, (DIS-COUNTS USERS
        # WITH LESS - hence not always selected number of users shown in table
        if count_iterable(get_users_spotify_tweets(user)) > s_tweet_minimum:

            # For each tweet, extract each component and collate it in a dataframe
            for tweet in get_users_spotify_tweets(user):
                text, song_id = get_s_tweet_data(tweet)
                tabulate_s_tweets(user_name=user, text=text, track_id=song_id, tweet_id=tweet.id, time=tweet.created_at)


def read_all_s_tweets():
    global all_s_tweets
    dtypes = {'user_name': 'str', 'text': 'str', 'track_id': 'str', 'tweet_id': 'int', 'time': 'str'}
    parse_date = ['time']
    all_s_tweets = pd.read_csv("s_tweets_trial.csv", index_col=0, dtype=dtypes, parse_dates=parse_date)


def _main_():
    global all_s_tweets
    global label_df
    # create_all_s_tweets()

    # Displays whole table of all users and corresponding spotify tweets
    # display(all_s_tweets)

    # Open csv and put into s_tweets
    read_all_s_tweets()
    display(all_s_tweets)

    # all_s_tweets.to_csv("s_tweets_trial.csv")

    # Gets a couple of previous tweets from a user before they posted a specific song
    display(get_before_s_tweets())

    display(label_df)
    all_s_tweets = pd.concat([all_s_tweets, label_df], axis=1)
    # Gets a list of songs
    all_song_lists = create_song_lists()

    # Gets the largest list of songs
    max_list = max(x for x in all_song_lists)

    # Graphs the largest song list
    graph_one_playlist(max_list)

    # Saves the tweets related to a track as a csv file [user id, text, track (if there), time]
    all_s_tweets.to_csv("s_tweets_trial.csv")

    display(all_s_tweets)

    # Outdated
    # Gets all tweets from a user
    # get_all_users_tweets()


_main_()
